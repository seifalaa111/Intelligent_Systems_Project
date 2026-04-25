/**
 * MIDAN Engine — PerformanceSyncManager
 * Controls pacing: computation time does NOT determine perceived speed (invariant #11).
 * Implements floor/ceiling/heartbeat budget model.
 *
 * CASE 1 (fast — actual < floor): hold until floor. Visual state continues.
 * CASE 2 (on time — actual ≤ floor+80ms): reveal immediately on completion.
 * CASE 3 (late — actual > floor+80ms, ≤ ceiling): extend 80ms increments + HEARTBEAT events.
 * CASE 4 (exceeded — actual > ceiling): reveal immediately, log overage.
 *
 * Adaptive floor: sustained load (3 consecutive extended) → raise baseline.
 * Recovery: 5 consecutive CASE 1/2 → floor decreases 1u per recovery.
 */

import { PERFORMANCE_BUDGET, EVENT } from '../core/constants.js';

export class PerformanceSyncManager {
  #store;
  #eventBus;
  #activePhases;  // phase → { startedAt, timer, heartbeatCount, waitResolve }
  #stats;         // per-phase stats tracking

  constructor(store, eventBus) {
    this.#store        = store;
    this.#eventBus     = eventBus;
    this.#activePhases = new Map();
    this.#stats        = {
      interpreting: { consecutiveExtended: 0, consecutiveOnTime: 0 },
      analyzing:    { consecutiveExtended: 0, consecutiveOnTime: 0 },
    };
  }

  /**
   * Begin timing a phase. Returns a Promise that resolves when reveal is permitted.
   * The pipeline awaits this before revealing output.
   *
   * @param {'interpreting'|'analyzing'} phase
   * @returns {Function} resolve — call this when actual computation completes
   */
  startPhase(phase) {
    const startedAt    = Date.now();
    let resolveReveal;
    const waitPromise  = new Promise(resolve => { resolveReveal = resolve; });

    const entry = { startedAt, heartbeatCount: 0, waitResolve: resolveReveal };
    this.#activePhases.set(phase, entry);

    return {
      /**
       * Signal that computation has completed. The manager decides when to reveal.
       * @param {number} [actualMs] - override for testing; defaults to elapsed time
       */
      complete: (actualMs) => this._onComplete(phase, actualMs ?? Date.now() - startedAt),
      wait: waitPromise,
    };
  }

  /**
   * Handle computation completion and determine reveal timing.
   */
  _onComplete(phase, actualMs) {
    const entry = this.#activePhases.get(phase);
    if (!entry) return;

    const state      = this.#store.getState();
    const floors     = state.performance.adjusted_floors;
    const floor      = floors[phase] ?? PERFORMANCE_BUDGET[`${phase.toUpperCase()}_FLOOR`];
    const ceiling    = PERFORMANCE_BUDGET[`${phase.toUpperCase()}_CEILING`];

    if (actualMs < floor) {
      // CASE 1: fast — hold until floor
      this._trackCase(phase, 'ontime');
      setTimeout(() => {
        this.#activePhases.delete(phase);
        entry.waitResolve();
      }, floor - actualMs);

    } else if (actualMs <= floor + PERFORMANCE_BUDGET.HEARTBEAT_UNIT) {
      // CASE 2: on time — reveal immediately
      this._trackCase(phase, 'ontime');
      this.#activePhases.delete(phase);
      entry.waitResolve();

    } else if (actualMs <= ceiling) {
      // CASE 3: late — already elapsed time, fire missed heartbeats then reveal
      this._trackCase(phase, 'extended');
      const elapsed       = actualMs - floor;
      const unitsMissed   = Math.min(
        Math.ceil(elapsed / PERFORMANCE_BUDGET.HEARTBEAT_UNIT),
        PERFORMANCE_BUDGET.MAX_HEARTBEATS - 1,
      );

      // Fire heartbeats for missed units, staggered
      for (let i = 0; i < unitsMissed; i++) {
        const hb = i + 1;
        setTimeout(() => {
          this._fireHeartbeat(hb, phase);
        }, i * PERFORMANCE_BUDGET.HEARTBEAT_UNIT);
      }

      // After heartbeats, reveal
      setTimeout(() => {
        this.#activePhases.delete(phase);
        this._updateAdaptiveFloor(phase);
        entry.waitResolve();
      }, unitsMissed * PERFORMANCE_BUDGET.HEARTBEAT_UNIT);

    } else {
      // CASE 4: exceeded ceiling — reveal immediately, log overage
      this._trackCase(phase, 'extended');
      console.warn(`[PerformanceSync] ${phase} exceeded ceiling: ${actualMs}ms > ${ceiling}ms`);
      this.#activePhases.delete(phase);
      this._updateAdaptiveFloor(phase);
      entry.waitResolve();
    }

    // Track latency sample
    this._trackLatency(phase, actualMs);
  }

  /**
   * Fire a HEARTBEAT event (H1–H5).
   * The perception layer hooks into these to drive visual heartbeat behaviors.
   * No UI logic in this class — events only.
   */
  _fireHeartbeat(n, phase) {
    this.#eventBus.emit(EVENT.HEARTBEAT, {
      n,
      phase,
      // Heartbeat types per spec:
      // H1: input shadow expands 2px
      // H2: caret glow advances one tier
      // H3: background gradient speed +15%
      // H4: input shadow pulse (repeat H1)
      // H5: no heartbeat — commit to reveal on next completion
      type: n === 1 ? 'shadow_pulse'
          : n === 2 ? 'caret_advance'
          : n === 3 ? 'gradient_accelerate'
          : n === 4 ? 'shadow_pulse'
          : 'final',
    });
  }

  /**
   * Update adjusted_floors in SESSION_STATE based on sustained load or recovery.
   */
  _updateAdaptiveFloor(phase) {
    const phaseKey  = phase === 'interpreting' ? 'interpreting' : 'analyzing';
    const phaseStats = this.#stats[phaseKey];

    if (phaseStats.consecutiveExtended >= PERFORMANCE_BUDGET.SUSTAINED_LOAD_COUNT) {
      // Raise floor by 1u
      this.#store.update(
        (current) => ({
          performance: {
            ...current.performance,
            adjusted_floors: {
              ...current.performance.adjusted_floors,
              [phaseKey]: Math.min(
                current.performance.adjusted_floors[phaseKey] + PERFORMANCE_BUDGET.FLOOR_DELTA,
                PERFORMANCE_BUDGET[`${phaseKey.toUpperCase()}_CEILING`],
              ),
            },
            latency_tracker: {
              ...current.performance.latency_tracker,
              sustained_load:  true,
              sustained_count: phaseStats.consecutiveExtended,
            },
          },
        }),
        { sourceLayer: 'PERFORMANCE' },
      );
      phaseStats.consecutiveExtended = 0;

    } else if (phaseStats.consecutiveOnTime >= PERFORMANCE_BUDGET.RECOVERY_COUNT) {
      // Decrease floor by 1u
      const baseline = PERFORMANCE_BUDGET[`${phaseKey.toUpperCase()}_FLOOR`];
      this.#store.update(
        (current) => ({
          performance: {
            ...current.performance,
            adjusted_floors: {
              ...current.performance.adjusted_floors,
              [phaseKey]: Math.max(
                current.performance.adjusted_floors[phaseKey] - PERFORMANCE_BUDGET.FLOOR_DELTA,
                baseline,
              ),
            },
            latency_tracker: {
              ...current.performance.latency_tracker,
              sustained_load:  false,
            },
          },
        }),
        { sourceLayer: 'PERFORMANCE' },
      );
      phaseStats.consecutiveOnTime = 0;
    }
  }

  _trackCase(phase, type) {
    const phaseKey   = phase === 'interpreting' ? 'interpreting' : 'analyzing';
    const phaseStats = this.#stats[phaseKey];
    if (type === 'extended') {
      phaseStats.consecutiveExtended++;
      phaseStats.consecutiveOnTime = 0;
    } else {
      phaseStats.consecutiveOnTime++;
      phaseStats.consecutiveExtended = 0;
    }
  }

  _trackLatency(phase, actualMs) {
    const phaseKey = phase === 'interpreting' ? 'interpreting' : 'analyzing';
    this.#store.update(
      (current) => {
        const tracker  = current.performance.latency_tracker;
        const samples  = [...(tracker.samples ?? []), actualMs].slice(-PERFORMANCE_BUDGET.LATENCY_WINDOW);
        const avg      = samples.reduce((a, b) => a + b, 0) / samples.length;
        return {
          performance: {
            ...current.performance,
            latency_tracker: {
              ...tracker,
              samples,
              rolling_average: Math.round(avg),
            },
          },
        };
      },
      { sourceLayer: 'PERFORMANCE' },
    );
  }

  /** Cancel all active phase timers (on engine destroy). */
  destroy() {
    this.#activePhases.clear();
  }
}
