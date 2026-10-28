//! Connection-pooled synchronous client for local and remote HyperspaceDB instances.

use std::{collections::HashMap, time::Duration};

use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use serde::{Serialize, de::DeserializeOwned};
use ureq::{
    Agent, Body,
    http::Response,
    unversioned::{
        resolver::DefaultResolver,
        transport::{Connector, DefaultConnector},
    },
};

use crate::{
    config::Config,
    error::Error,
    transport::ResumeReads,
    types::{
        BatchInsertRequest, CountRequest, CountResponse, CreateCollectionRequest, GraphNode,
        GraphTraverseRequest, HttpFilter, InduceGraphRequest, InsertPayload, LinkBatchRequest,
        LinkEdgeRequest, MetadataInductionRule, PointRecord, ScrollRequest, SearchRequest,
        SearchResultHit, UnlinkEdgeRequest, VerifyGenerationRequest, VerifyGenerationResponse,
    },
};

/// Crate version sent in the `User-Agent` header.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Maximum vectors sent in a single `/insert/batch` HTTP request.
pub const MAX_UPSERT_BATCH_SIZE: usize = 1_000;

/// Maximum points fetched per `/scroll` page during `VACUUM` / bulk scans.
const SCROLL_PAGE_SIZE: usize = 1_000;

/// Defensive upper bound on `/scroll` pages during a single `list_all_points` call.
const MAX_SCROLL_PAGES: usize = 50_000;

/// Characters percent-encoded when interpolating collection names into URL paths.
const PATH_SEGMENT: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'/')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'\\')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}');

fn encode_segment(segment: &str) -> impl std::fmt::Display + '_ {
    utf8_percent_encode(segment, PATH_SEGMENT)
}

fn default_user_agent() -> String {
    format!("pg_hyperspace-client/{VERSION} (rust; postgres-am)")
}

/// Synchronous, connection-pooled client for HyperspaceDB.
///
/// Uses `ureq`'s internal keep-alive connection pool combined with [`ResumeReads`] so
/// repeated queries from the same PostgreSQL backend reuse open TCP sockets with sub-millisecond
/// local IPC latency and automatic `EINTR` signal recovery.
pub struct Client {
    agent: Agent,
    config: Config,
    sleep: Box<dyn Fn(Duration) + Send + Sync>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("endpoint", &self.config.endpoint)
            .field("user_id", &self.config.user_id)
            .finish_non_exhaustive()
    }
}

impl Client {
    /// Creates a new `Client` with the given [`Config`].
    #[must_use]
    pub fn new(config: Config) -> Self {
        Self::with_sleep(config, std::thread::sleep)
    }

    /// Creates a new `Client` with a custom backoff sleep function (useful for deterministic tests).
    #[must_use]
    pub fn with_sleep(config: Config, sleep: impl Fn(Duration) + Send + Sync + 'static) -> Self {
        let user_agent = config.user_agent.clone().unwrap_or_else(default_user_agent);
        let agent_config = Agent::config_builder()
            .timeout_global(Some(config.timeout))
            .http_status_as_error(false)
            .user_agent(user_agent)
            .build();
        let connector = DefaultConnector::default().chain(ResumeReads);
        Self {
            agent: Agent::with_parts(agent_config, connector, DefaultResolver::default()),
            config,
            sleep: Box::new(sleep),
        }
    }

    /// Returns a reference to the client's [`Config`].
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// Checks whether the HyperspaceDB server is online (`GET /health`).
    pub fn health(&self) -> Result<bool, Error> {
        let url = format!("{}/health", self.config.endpoint);
        let resp = self.get_with_retry(&url)?;
        Ok(resp.status().as_u16() == 200)
    }

    /// Checks if a collection exists (`GET /api/collections/{name}/exists`).
    pub fn collection_exists(&self, collection: &str) -> Result<bool, Error> {
        let url = format!(
            "{}/api/collections/{}/exists",
            self.config.endpoint,
            encode_segment(collection)
        );
        let mut resp = self.get_with_retry(&url)?;
        match resp.status().as_u16() {
            200 => {
                #[derive(serde::Deserialize)]
                struct ExistsResp {
                    #[serde(default)]
                    exists: bool,
                }
                let parsed: ExistsResp = read_json(&mut resp)?;
                Ok(parsed.exists)
            }
            404 => Ok(false),
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Creates a collection (`POST /api/collections`).
    ///
    /// If the collection already exists, this method succeeds idempotently.
    pub fn create_collection(&self, req: &CreateCollectionRequest) -> Result<(), Error> {
        let url = format!("{}/api/collections", self.config.endpoint);
        let mut resp = self.post_json_with_retry(&url, req)?;
        let status = resp.status().as_u16();
        if status == 200 || status == 201 {
            return Ok(());
        }
        let body = read_body_string(&mut resp);
        if status == 400 && body.to_ascii_lowercase().contains("already exists") {
            return Ok(());
        }
        Err(Error::HttpStatus {
            status,
            message: body,
        })
    }

    /// Deletes a collection (`DELETE /api/collections/{name}`).
    ///
    /// Succeeds idempotently if the collection does not exist (`404`).
    pub fn delete_collection(&self, collection: &str) -> Result<(), Error> {
        let url = format!(
            "{}/api/collections/{}",
            self.config.endpoint,
            encode_segment(collection)
        );
        let mut resp = self.delete_with_retry(&url)?;
        match resp.status().as_u16() {
            200 | 204 | 404 => Ok(()),
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Inserts or updates a single vector point (`POST /api/collections/{name}/insert`).
    pub fn insert_point(&self, collection: &str, payload: &InsertPayload) -> Result<(), Error> {
        let url = format!(
            "{}/api/collections/{}/insert",
            self.config.endpoint,
            encode_segment(collection)
        );
        let mut resp = self.post_json_with_retry(&url, payload)?;
        match resp.status().as_u16() {
            200 | 201 => Ok(()),
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Batch-inserts vectors (`POST /api/collections/{name}/insert/batch`), automatically chunking
    /// at [`MAX_UPSERT_BATCH_SIZE`]. Returns total vectors inserted.
    pub fn insert_batch(
        &self,
        collection: &str,
        vectors: &[InsertPayload],
    ) -> Result<usize, Error> {
        if vectors.is_empty() {
            return Ok(0);
        }
        let url = format!(
            "{}/api/collections/{}/insert/batch",
            self.config.endpoint,
            encode_segment(collection)
        );
        let mut total = 0;
        for chunk in vectors.chunks(MAX_UPSERT_BATCH_SIZE) {
            let body = BatchInsertRequest {
                vectors: chunk.to_vec(),
            };
            let mut resp = self.post_json_with_retry(&url, &body)?;
            match resp.status().as_u16() {
                200 | 201 => total += chunk.len(),
                status => return Err(read_status_error(status, &mut resp)),
            }
        }
        Ok(total)
    }

    /// Deletes a single point by its 32-bit ID (`DELETE /api/collections/{name}/points/{id}`).
    pub fn delete_point(&self, collection: &str, id: u32) -> Result<(), Error> {
        let url = format!(
            "{}/api/collections/{}/points/{id}",
            self.config.endpoint,
            encode_segment(collection)
        );
        let mut resp = self.delete_with_retry(&url)?;
        match resp.status().as_u16() {
            200 | 204 | 404 => Ok(()),
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Deletes a batch of point IDs.
    pub fn delete_points_batch(&self, collection: &str, ids: &[u32]) -> Result<usize, Error> {
        let mut deleted = 0;
        for &id in ids {
            self.delete_point(collection, id)?;
            deleted += 1;
        }
        Ok(deleted)
    }

    /// Executes a nearest-neighbor vector search (`POST /api/collections/{name}/search`).
    pub fn search(
        &self,
        collection: &str,
        req: &SearchRequest,
    ) -> Result<Vec<SearchResultHit>, Error> {
        let url = format!(
            "{}/api/collections/{}/search",
            self.config.endpoint,
            encode_segment(collection)
        );
        let mut resp = self.post_json_with_retry(&url, req)?;
        match resp.status().as_u16() {
            200 => read_json(&mut resp),
            404 => Ok(Vec::new()),
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Scrolls through points in a collection (`POST /api/collections/{name}/scroll`).
    pub fn scroll(&self, collection: &str, req: &ScrollRequest) -> Result<Vec<PointRecord>, Error> {
        let url = format!(
            "{}/api/collections/{}/scroll",
            self.config.endpoint,
            encode_segment(collection)
        );
        let mut resp = self.post_json_with_retry(&url, req)?;
        match resp.status().as_u16() {
            200 => read_json(&mut resp),
            404 => Ok(Vec::new()),
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Lists all `(point_id, metadata)` pairs in `collection` for `VACUUM` dead-tuple sweeps.
    pub fn list_all_points(
        &self,
        collection: &str,
    ) -> Result<Vec<(u32, HashMap<String, String>)>, Error> {
        let mut out = Vec::new();
        let mut offset = 0;
        for _ in 0..MAX_SCROLL_PAGES {
            let page = self.scroll(
                collection,
                &ScrollRequest {
                    limit: Some(SCROLL_PAGE_SIZE),
                    offset: Some(offset),
                    filters: None,
                },
            )?;
            let count = page.len();
            for pt in page {
                out.push((pt.id, pt.metadata));
            }
            if count < SCROLL_PAGE_SIZE {
                break;
            }
            offset += count;
        }
        Ok(out)
    }

    /// Counts points in a collection matching optional filters (`POST /api/collections/{name}/count`).
    pub fn count(
        &self,
        collection: &str,
        filters: Option<Vec<HttpFilter>>,
    ) -> Result<usize, Error> {
        let url = format!(
            "{}/api/collections/{}/count",
            self.config.endpoint,
            encode_segment(collection)
        );
        let req = CountRequest { filters };
        let mut resp = self.post_json_with_retry(&url, &req)?;
        match resp.status().as_u16() {
            200 => {
                let parsed: CountResponse = read_json(&mut resp)?;
                Ok(parsed.count)
            }
            404 => Ok(0),
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Links two nodes in the Native Vector-Graph (KAG) (`POST /api/collections/{name}/graph/link`).
    pub fn link_edge(&self, collection: &str, req: &LinkEdgeRequest) -> Result<(), Error> {
        let url = format!(
            "{}/api/collections/{}/graph/link",
            self.config.endpoint,
            encode_segment(collection)
        );
        let mut resp = self.post_json_with_retry(&url, req)?;
        match resp.status().as_u16() {
            200 | 201 => Ok(()),
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Batch-links directed edges in the KAG graph (`POST /api/collections/{name}/graph/links`).
    pub fn link_batch(&self, collection: &str, edges: &[LinkEdgeRequest]) -> Result<usize, Error> {
        if edges.is_empty() {
            return Ok(0);
        }
        let url = format!(
            "{}/api/collections/{}/graph/links",
            self.config.endpoint,
            encode_segment(collection)
        );
        let req = LinkBatchRequest {
            edges: edges.to_vec(),
        };
        let mut resp = self.post_json_with_retry(&url, &req)?;
        match resp.status().as_u16() {
            200 | 201 => {
                #[derive(serde::Deserialize)]
                struct LinkBatchResp {
                    #[serde(default)]
                    edges_linked: usize,
                }
                let parsed: LinkBatchResp = read_json(&mut resp)?;
                Ok(parsed.edges_linked)
            }
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Removes a directed edge from the KAG graph (`DELETE /api/collections/{name}/graph/link`).
    pub fn unlink_edge(&self, collection: &str, req: &UnlinkEdgeRequest) -> Result<bool, Error> {
        let url = format!(
            "{}/api/collections/{}/graph/link",
            self.config.endpoint,
            encode_segment(collection)
        );
        let mut resp = self.delete_json_with_retry(&url, req)?;
        match resp.status().as_u16() {
            200 => {
                #[derive(serde::Deserialize)]
                struct UnlinkResp {
                    #[serde(default)]
                    status: String,
                }
                let parsed: UnlinkResp = read_json(&mut resp)?;
                Ok(parsed.status == "OK")
            }
            404 => Ok(false),
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Induces KAG edges declaratively from metadata (`POST /api/collections/{name}/graph/induce`).
    pub fn induce_graph(
        &self,
        collection: &str,
        rules: &[MetadataInductionRule],
    ) -> Result<usize, Error> {
        let url = format!(
            "{}/api/collections/{}/graph/induce",
            self.config.endpoint,
            encode_segment(collection)
        );
        let req = InduceGraphRequest {
            rules: rules.to_vec(),
        };
        let mut resp = self.post_json_with_retry(&url, &req)?;
        match resp.status().as_u16() {
            200 => {
                #[derive(serde::Deserialize)]
                struct InduceResp {
                    #[serde(default)]
                    edges_created: usize,
                }
                let parsed: InduceResp = read_json(&mut resp)?;
                Ok(parsed.edges_created)
            }
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Performs sub-millisecond KAG BFS graph traversal (`POST /api/collections/{name}/graph/traverse`).
    pub fn traverse_graph(
        &self,
        collection: &str,
        req: &GraphTraverseRequest,
    ) -> Result<Vec<GraphNode>, Error> {
        let url = format!(
            "{}/api/collections/{}/graph/traverse",
            self.config.endpoint,
            encode_segment(collection)
        );
        let mut resp = self.post_json_with_retry(&url, req)?;
        match resp.status().as_u16() {
            200 => read_json(&mut resp),
            404 => Ok(Vec::new()),
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    /// Verifies an LLM generation against context chunks via the Tribunal Router
    /// (`POST /api/collections/{name}/verify-generation`).
    pub fn verify_generation(
        &self,
        collection: &str,
        req: &VerifyGenerationRequest,
    ) -> Result<VerifyGenerationResponse, Error> {
        let url = format!(
            "{}/api/collections/{}/verify-generation",
            self.config.endpoint,
            encode_segment(collection)
        );
        let mut resp = self.post_json_with_retry(&url, req)?;
        match resp.status().as_u16() {
            200 => read_json(&mut resp),
            status => Err(read_status_error(status, &mut resp)),
        }
    }

    // -----------------------------------------------------------------------
    // Internal retry & header helpers
    // -----------------------------------------------------------------------

    fn get_with_retry(&self, url: &str) -> Result<Response<Body>, Error> {
        self.execute_with_retry(|| {
            let mut req = self.agent.get(url);
            if let Some(ref key) = self.config.api_key {
                req = req.header("x-api-key", key.as_str());
            }
            if let Some(ref uid) = self.config.user_id {
                req = req.header("x-hyperspace-user-id", uid.as_str());
            }
            req.call().map_err(|e| Error::Transport(e.to_string()))
        })
    }

    fn delete_with_retry(&self, url: &str) -> Result<Response<Body>, Error> {
        self.execute_with_retry(|| {
            let mut req = self.agent.delete(url);
            if let Some(ref key) = self.config.api_key {
                req = req.header("x-api-key", key.as_str());
            }
            if let Some(ref uid) = self.config.user_id {
                req = req.header("x-hyperspace-user-id", uid.as_str());
            }
            req.call().map_err(|e| Error::Transport(e.to_string()))
        })
    }

    fn delete_json_with_retry<T: Serialize>(
        &self,
        url: &str,
        body: &T,
    ) -> Result<Response<Body>, Error> {
        let payload = serde_json::to_vec(body).map_err(|e| Error::Serialization(e.to_string()))?;
        self.execute_with_retry(|| {
            let mut req = self
                .agent
                .delete(url)
                .force_send_body()
                .header("content-type", "application/json");
            if let Some(ref key) = self.config.api_key {
                req = req.header("x-api-key", key.as_str());
            }
            if let Some(ref uid) = self.config.user_id {
                req = req.header("x-hyperspace-user-id", uid.as_str());
            }
            req.send(&payload[..])
                .map_err(|e| Error::Transport(e.to_string()))
        })
    }

    fn post_json_with_retry<T: Serialize>(
        &self,
        url: &str,
        body: &T,
    ) -> Result<Response<Body>, Error> {
        let payload = serde_json::to_vec(body).map_err(|e| Error::Serialization(e.to_string()))?;
        self.execute_with_retry(|| {
            let mut req = self
                .agent
                .post(url)
                .header("content-type", "application/json");
            if let Some(ref key) = self.config.api_key {
                req = req.header("x-api-key", key.as_str());
            }
            if let Some(ref uid) = self.config.user_id {
                req = req.header("x-hyperspace-user-id", uid.as_str());
            }
            req.send(&payload[..])
                .map_err(|e| Error::Transport(e.to_string()))
        })
    }

    fn execute_with_retry(
        &self,
        mut op: impl FnMut() -> Result<Response<Body>, Error>,
    ) -> Result<Response<Body>, Error> {
        let max_attempts = self.config.max_retries.saturating_add(1).max(1);
        let mut attempt = 0;
        loop {
            attempt += 1;
            match op() {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let transient_http = matches!(status, 408 | 429 | 502 | 503 | 504);
                    if transient_http && attempt < max_attempts {
                        let backoff = Duration::from_millis(50 * (1_u64 << attempt.min(5)));
                        (self.sleep)(backoff);
                        continue;
                    }
                    return Ok(resp);
                }
                Err(err) if err.is_transient() && attempt < max_attempts => {
                    let backoff = Duration::from_millis(50 * (1_u64 << attempt.min(5)));
                    (self.sleep)(backoff);
                }
                Err(err) => return Err(err),
            }
        }
    }
}

fn read_json<T: DeserializeOwned>(resp: &mut Response<Body>) -> Result<T, Error> {
    resp.body_mut()
        .read_json::<T>()
        .map_err(|e| Error::Serialization(e.to_string()))
}

fn read_body_string(resp: &mut Response<Body>) -> String {
    resp.body_mut()
        .read_to_string()
        .unwrap_or_else(|_| "<unreadable response body>".to_string())
}

fn read_status_error(status: u16, resp: &mut Response<Body>) -> Error {
    let message = read_body_string(resp);
    Error::HttpStatus { status, message }
}
