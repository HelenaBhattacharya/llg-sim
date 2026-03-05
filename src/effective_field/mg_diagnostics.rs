// src/effective_field/mg_diagnostics.rs
//
// Timing and diagnostic infrastructure for the multigrid demag solver.
//
// The 3D padded-box solver (demag_poisson_mg.rs) has its own built-in timing
// controlled by LLG_DEMAG_MG_TIMING. This module provides shared utilities
// that can be used by any MG-based solver.
//
// Environment variables:
//   LLG_DEMAG_MG_TIMING=1        Print per-step timing breakdown
//   LLG_DEMAG_MG_CONVERGENCE=1   Print per-V-cycle residual norms

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Feature flags (from env vars, read once)
// ---------------------------------------------------------------------------

#[inline]
pub fn timing_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("LLG_DEMAG_MG_TIMING")
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    })
}

#[inline]
pub fn convergence_log_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("LLG_DEMAG_MG_CONVERGENCE")
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    })
}

// ---------------------------------------------------------------------------
// Per-phase timing accumulator
// ---------------------------------------------------------------------------

/// Accumulated wall-clock times for the major phases of the demag pipeline,
/// stored as nanoseconds in atomics for thread safety.
pub struct DemagTimingAccum {
    pub total_ns: AtomicU64,
    pub call_count: AtomicUsize,
}

impl DemagTimingAccum {
    pub const fn new() -> Self {
        Self {
            total_ns: AtomicU64::new(0),
            call_count: AtomicUsize::new(0),
        }
    }

    pub fn print_summary(&self) {
        let n = self.call_count.load(Ordering::Relaxed);
        if n == 0 { return; }
        let to_ms = |ns: u64| ns as f64 / 1e6;
        let total = self.total_ns.load(Ordering::Relaxed);
        eprintln!(
            "[demag_mg timing] calls={}, total={:.1}ms, avg={:.2}ms/call",
            n, to_ms(total), to_ms(total) / n as f64,
        );
    }
}

pub static TIMING: DemagTimingAccum = DemagTimingAccum::new();

// ---------------------------------------------------------------------------
// Convenience timer
// ---------------------------------------------------------------------------

/// RAII timer that accumulates elapsed nanoseconds into an AtomicU64.
pub struct PhaseTimer<'a> {
    target: &'a AtomicU64,
    start: Instant,
}

impl<'a> PhaseTimer<'a> {
    #[inline]
    pub fn start(target: &'a AtomicU64) -> Self {
        Self { target, start: Instant::now() }
    }
}

impl<'a> Drop for PhaseTimer<'a> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_nanos() as u64;
        self.target.fetch_add(elapsed, Ordering::Relaxed);
    }
}