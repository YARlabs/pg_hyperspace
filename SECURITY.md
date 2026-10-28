# Security Policy

## Supported Versions

We actively maintain and provide security updates for the latest minor release of `pg_hyperspace` across PostgreSQL 16, 17, and 18.

| Version | PostgreSQL Versions | Supported | Status |
| :--- | :--- | :---: | :--- |
| `0.1.x` | PostgreSQL 16, 17, 18 | :white_check_mark: | Active (Current) |

## Reporting a Vulnerability

We take the security of PostgreSQL backends and HyperspaceDB clusters seriously. If you discover a security vulnerability in `pg_hyperspace`, please **do not disclose it publicly** in GitHub issues.

**Please report vulnerabilities directly via email to:**
`hi@yar.ink`

Include:
1. A clear description of the vulnerability and its impact (e.g., privilege escalation, memory corruption, credential leakage).
2. PostgreSQL version (`pg16`, `pg17`, or `pg18`), OS/architecture, and `pg_hyperspace` version.
3. Minimal SQL reproduction steps.

We will acknowledge receipt within **48 hours** and provide an estimated timeline for a patch and coordinated disclosure.

## Security Architecture & Guarantees

1. **Superuser-Restricted Credential GUCs (`GucFlags::SUPERUSER_ONLY`):**
   - `hyperspace.api_key` is registered with `GucFlags::SUPERUSER_ONLY`, preventing unprivileged database roles from inspecting (`SHOW hyperspace.api_key`) or exfiltrating cluster API keys.
2. **Locked Catalog & Outbox Tables (`REVOKE ALL ... FROM PUBLIC`):**
   - Catalog tables (`hyperspace.tables`, `hyperspace.outbox`, `hyperspace.edges_outbox`) revoke read/write privileges from `PUBLIC` at extension bootstrap time so per-table API keys and queued vectors cannot be read by untrusted roles.
3. **Hardened `SECURITY DEFINER` Event Trigger:**
   - The `hyperspace._on_sql_drop()` event trigger executes with an explicit `SET search_path = pg_catalog, pg_temp` to prevent schema-shadowing and search-path hijacking attacks during `DROP TABLE` / `DROP INDEX`.
4. **SQL Injection Immunity via Identifier Quoting & Parameterized SPI:**
   - All internal SPI queries use bound positional parameters (`$1`, `$2`, ...) or `pg_catalog.quote_ident` / `pgrx::spi::quote_qualified_identifier`.
5. **TOAST & Memory Context Isolation:**
   - Detoasted `varlena` buffers in `parse_vector_f64` are freed immediately via `pg_sys::pfree`, and scan state allocations use `PgMemoryContexts::CurrentMemoryContext.leak_and_drop_on_delete(...)`, preventing memory exhaustion or use-after-free when a PostgreSQL query is cancelled (`SIGINT` / `ERROR` `longjmp`).
6. **Zero-Knowledge Encryption Compatibility:**
   - `pg_hyperspace` supports HyperspaceDB v4.1+ Zero-Knowledge Encrypted Collections (`encrypted: true`) with isometric orthogonal/Lorentz transformations (`Vec2Text` inversion defense) and mTLS (`https://`) endpoints via `rustls`.

