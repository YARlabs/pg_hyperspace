//! Signal-resilient transport wrapper for running HTTP inside PostgreSQL backend processes.
//!
//! PostgreSQL backends frequently receive `SIGUSR1` (latch wakeups, shared invalidation messages,
//! recovery conflict probes). On Linux and macOS, a blocking `read`/`poll` syscall with a socket
//! timeout can return `EINTR` (`io::ErrorKind::Interrupted`). This wrapper transparently resumes
//! interrupted reads within the remaining deadline budget so in-flight vector queries never fail
//! spuriously under heavy Postgres concurrency.

use std::{io, time::Instant};

use ureq::{
    Error,
    unversioned::transport::{
        Buffers, ConnectionDetails, Connector, NextTimeout, Transport, time::Duration,
    },
};

/// Connector adapter that wraps underlying transports in [`Resuming`].
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ResumeReads;

impl Connector<Box<dyn Transport>> for ResumeReads {
    type Out = Resuming;

    fn connect(
        &self,
        _: &ConnectionDetails,
        chained: Option<Box<dyn Transport>>,
    ) -> Result<Option<Resuming>, Error> {
        Ok(chained.map(Resuming))
    }
}

/// Transport wrapper that retries `ErrorKind::Interrupted` within the original timeout budget.
#[derive(Debug)]
pub(crate) struct Resuming(Box<dyn Transport>);

impl Transport for Resuming {
    fn buffers(&mut self) -> &mut dyn Buffers {
        self.0.buffers()
    }

    fn transmit_output(&mut self, amount: usize, timeout: NextTimeout) -> Result<(), Error> {
        self.0.transmit_output(amount, timeout)
    }

    fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, Error> {
        let started = Instant::now();
        let mut next = timeout;
        loop {
            match self.0.await_input(next) {
                Err(Error::Io(ref e)) if e.kind() == io::ErrorKind::Interrupted => {}
                done => return done,
            }
            if let Duration::Exact(total) = timeout.after {
                match total.checked_sub(started.elapsed()) {
                    Some(left) if !left.is_zero() => next.after = Duration::Exact(left),
                    _ => return Err(Error::Timeout(timeout.reason)),
                }
            }
        }
    }

    fn is_open(&mut self) -> bool {
        self.0.is_open()
    }

    fn is_tls(&self) -> bool {
        self.0.is_tls()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
        thread,
        time::Duration as StdDuration,
    };

    use ureq::{Timeout, unversioned::transport::LazyBuffers};

    use super::*;

    #[derive(Debug)]
    struct MockScriptTransport {
        buffers: LazyBuffers,
        errors: VecDeque<io::ErrorKind>,
        pause: StdDuration,
        waits: Arc<Mutex<Vec<NextTimeout>>>,
    }

    impl Transport for MockScriptTransport {
        fn buffers(&mut self) -> &mut dyn Buffers {
            &mut self.buffers
        }

        fn transmit_output(&mut self, _: usize, _: NextTimeout) -> Result<(), Error> {
            Ok(())
        }

        fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, Error> {
            self.waits.lock().expect("lock").push(timeout);
            if !self.pause.is_zero() {
                thread::sleep(self.pause);
            }
            match self.errors.pop_front() {
                Some(kind) => Err(Error::Io(kind.into())),
                None => Ok(true),
            }
        }

        fn is_open(&mut self) -> bool {
            true
        }
    }

    #[test]
    fn resumes_interrupted_reads_within_deadline() {
        let waits = Arc::new(Mutex::new(Vec::new()));
        let script = MockScriptTransport {
            buffers: LazyBuffers::new(64, 64),
            errors: VecDeque::from([io::ErrorKind::Interrupted, io::ErrorKind::Interrupted]),
            pause: StdDuration::from_millis(2),
            waits: waits.clone(),
        };
        let mut transport = Resuming(Box::new(script));
        let timeout = NextTimeout {
            after: Duration::from_secs(2),
            reason: Timeout::Global,
            per_read: None,
            per_write: None,
        };
        assert!(transport.await_input(timeout).expect("should succeed"));
        assert_eq!(waits.lock().expect("lock").len(), 3);
    }
}
