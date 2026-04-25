/**
 * MIDAN Engine — EventBus
 * Explicit version rejection policy (Fix #2):
 *
 *   event.version < state.version - MAX_DRIFT  → REJECT  (genuinely stale)
 *   event.version in [state.version - MAX_DRIFT, state.version] → APPLY
 *   event.version > state.version              → QUEUE   (future event — apply when state catches up)
 *
 * MAX_DRIFT is per-event-type: tight for critical transitions (0-1), loose for background layers (2-3).
 * A zero-drift event is rejected the instant state advances past its version.
 *
 * The queue is bounded (max 32 events per type). Overflow drops the oldest entry.
 */

// Dispatch priority for pending-queue ordering (lower number = dispatched first).
// Applied only when draining the pending queue — synchronous emits fire in emission order.
//
// P1 — System transitions   (state machine authority — must process first)
// P2 — Tier 1 signal updates (registry changes, completeness)
// P3 — Interpretation        (component text, conclusions, confidence)
// P4 — Tier 2 signal updates (behavioral inferences, async results)
// P5 — Raw input             (lowest — high-frequency, best-effort)
const EVENT_PRIORITY = {
  // P1: System transitions
  STATE_TRANSITION:               1,
  PIPELINE_INTERRUPTED:           1,
  CONCLUSION_RETRACTED:           1,
  ASSUMPTION_CONFIRMED:           1,
  ASSUMPTION_REJECTED:            1,
  THREAD_SWITCHED:                1,
  THREAD_ARCHIVED:                1,

  // P2: Tier 1 signal updates
  SIGNAL_DETECTED:                2,
  SIGNAL_REMOVED:                 2,
  SIGNAL_PROMOTED:                2,
  SIGNAL_DEMOTED:                 2,
  COMPLETENESS_UPDATED:           2,
  PIPELINE_STARTED:               2,
  PIPELINE_COMPLETED:             2,
  PIPELINE_CHECKPOINT:            2,

  // P3: Interpretation
  INTERPRETATION_UPDATED:         3,
  CONCLUSION_UPDATED:             3,
  CONFIDENCE_CHANGED:             3,
  OUTPUT_LAYER_REVEALED:          3,

  // P4: Tier 2 / background signals
  UNCERTAIN_ACTIVATED:            4,
  UNCERTAIN_DEACTIVATED:          4,
  INTERRUPTION_CLASSIFIED:        4,
  THREAD_CREATED:                 4,
  THREAD_SNAPSHOT_COMPACTED:      4,
  HEARTBEAT:                      4,

  // P5: Raw input + informational (highest frequency, lowest urgency)
  INPUT_RAW_UPDATED:              5,
  SESSION_STATE_UPDATED:          5,
  STALE_WRITE_REJECTED:           5,
  ANIMATION_REGISTERED:           5,
  ANIMATION_COMPLETED:            5,
  SYNC_POINT_CLEARED:             5,
};

// Max version drift before an event is considered stale (per event type)
// 0 = must match exactly, 99 = never stale
const MAX_DRIFT = {
  // Critical / ordering-sensitive — must be at current version
  STATE_TRANSITION:               0,
  PIPELINE_INTERRUPTED:           0,
  CONCLUSION_RETRACTED:           0,
  ASSUMPTION_CONFIRMED:           0,
  ASSUMPTION_REJECTED:            0,
  THREAD_SWITCHED:                0,
  THREAD_ARCHIVED:                0,

  // Near-current — tolerate 1 concurrent write from another layer
  SIGNAL_DETECTED:                1,
  SIGNAL_REMOVED:                 1,
  SIGNAL_PROMOTED:                1,
  SIGNAL_DEMOTED:                 1,
  PIPELINE_STARTED:               1,
  PIPELINE_COMPLETED:             1,
  PIPELINE_CHECKPOINT:            1,
  OUTPUT_LAYER_REVEALED:          1,

  // Background — tolerate up to 3 concurrent layer writes
  INPUT_RAW_UPDATED:              3,
  INTERPRETATION_UPDATED:         3,
  COMPLETENESS_UPDATED:           3,
  CONFIDENCE_CHANGED:             3,
  CONCLUSION_UPDATED:             3,

  // Informational — never stale
  SESSION_STATE_UPDATED:         99,
  STALE_WRITE_REJECTED:          99,
  ANIMATION_REGISTERED:          99,
  ANIMATION_COMPLETED:           99,
  SYNC_POINT_CLEARED:            99,
  HEARTBEAT:                     99,
  THREAD_CREATED:                99,
  UNCERTAIN_ACTIVATED:           99,
  UNCERTAIN_DEACTIVATED:         99,
  INTERRUPTION_CLASSIFIED:       99,
  THREAD_SNAPSHOT_COMPACTED:     99,
};

const PENDING_QUEUE_MAX      = 32;
// Any queued event older than this is promoted past lower-running high-priority events,
// preventing indefinite starvation of P3-P5 events under sustained P1-P2 load.
const STARVATION_MAX_WAIT_MS = 400;

export class EventBus {
  #handlers;
  #pendingQueue;    // eventType → Array<{payload, evtVersion, addedAt}>  (future events)
  #getVersion;      // () → current SESSION_STATE version
  #isDraining;      // re-entrancy guard for _drainQueue
  #deferredVersion; // highest version requested during an active drain

  /**
   * @param {Function} getVersion - returns current state version
   */
  constructor(getVersion) {
    this.#handlers        = new Map();
    this.#pendingQueue    = new Map();
    this.#getVersion      = getVersion ?? (() => 0);
    this.#isDraining      = false;
    this.#deferredVersion = null;
  }

  /**
   * Subscribe to an event type. Returns unsubscribe function.
   * @param {string}   eventType
   * @param {Function} handler    - (envelope: { eventType, payload, version, timestamp }) => void
   */
  on(eventType, handler) {
    if (!this.#handlers.has(eventType)) this.#handlers.set(eventType, new Set());
    this.#handlers.get(eventType).add(handler);
    return () => this.off(eventType, handler);
  }

  off(eventType, handler) {
    this.#handlers.get(eventType)?.delete(handler);
  }

  /**
   * Emit an event with explicit version policy enforcement.
   *
   * @param {string} eventType
   * @param {Object} payload
   * @param {number} [originVersion] - version at which this event was generated
   */
  emit(eventType, payload = {}, originVersion = null) {
    const stateVersion = this.#getVersion();
    const evtVersion   = originVersion ?? stateVersion;
    const maxDrift     = MAX_DRIFT[eventType] ?? 3;

    // ── Version gate ──────────────────────────────────────────────────────────
    if (evtVersion < stateVersion - maxDrift) {
      // STALE — reject
      this._debugLog('REJECTED (stale)', eventType, evtVersion, stateVersion, maxDrift);
      return;
    }

    if (evtVersion > stateVersion) {
      // FUTURE — queue until state catches up
      this._enqueue(eventType, payload, evtVersion);
      this._debugLog('QUEUED (future)', eventType, evtVersion, stateVersion, maxDrift);
      return;
    }

    // ── APPLY ─────────────────────────────────────────────────────────────────
    const envelope = { eventType, payload, version: evtVersion, timestamp: Date.now() };
    this._dispatch(eventType, envelope);

    // After dispatch, drain any queued events that are now current
    this._drainQueue(stateVersion);
  }

  /**
   * Called by StateReducer after each state write to drain queued future events.
   * @param {number} newVersion
   */
  onStateAdvanced(newVersion) {
    this._drainQueue(newVersion);
  }

  destroy() {
    this.#handlers.clear();
    this.#pendingQueue.clear();
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  _dispatch(eventType, envelope) {
    const handlers = this.#handlers.get(eventType);
    if (!handlers || handlers.size === 0) return;
    handlers.forEach(fn => {
      try { fn(envelope); }
      catch (err) { console.error(`[EventBus] handler error for "${eventType}"`, err); }
    });
  }

  _enqueue(eventType, payload, evtVersion) {
    if (!this.#pendingQueue.has(eventType)) this.#pendingQueue.set(eventType, []);
    const queue = this.#pendingQueue.get(eventType);
    queue.push({ payload, evtVersion, addedAt: Date.now() });
    // Bounded queue — drop oldest if overflow
    if (queue.length > PENDING_QUEUE_MAX) queue.shift();
  }

  _drainQueue(currentVersion) {
    // Re-entrancy guard: if a handler triggers another drain (e.g. via store.update →
    // onStateAdvanced), defer it until the current drain loop finishes.
    if (this.#isDraining) {
      this.#deferredVersion = this.#deferredVersion == null
        ? currentVersion
        : Math.max(this.#deferredVersion, currentVersion);
      return;
    }

    this.#isDraining = true;
    try {
      this._drainPass(currentVersion);
    } finally {
      this.#isDraining = false;
      // If a deferred drain was requested during this pass, run it now.
      if (this.#deferredVersion != null) {
        const v = this.#deferredVersion;
        this.#deferredVersion = null;
        this._drainQueue(v);
      }
    }
  }

  _drainPass(currentVersion) {
    // Collect all events that are now current (or stale) across all queued types.
    const ready = [];
    const now   = Date.now();

    for (const [eventType, queue] of this.#pendingQueue.entries()) {
      const maxDrift = MAX_DRIFT[eventType] ?? 3;
      let i = 0;
      while (i < queue.length) {
        const { payload, evtVersion, addedAt } = queue[i];

        if (evtVersion < currentVersion - maxDrift) {
          queue.splice(i, 1); // stale — drop silently
        } else if (evtVersion <= currentVersion) {
          ready.push({ eventType, payload, evtVersion, addedAt });
          queue.splice(i, 1); // remove before dispatch to prevent double-fire
        } else {
          i++; // still future — leave it
        }
      }
      if (queue.length === 0) this.#pendingQueue.delete(eventType);
    }

    if (ready.length === 0) return;

    // Sort by priority so P1 (system transitions) always dispatches before P5 (input).
    // Ties keep arrival order (stable sort in V8 / modern JS).
    ready.sort((a, b) =>
      (EVENT_PRIORITY[a.eventType] ?? 5) - (EVENT_PRIORITY[b.eventType] ?? 5),
    );

    // ── Starvation prevention ──────────────────────────────────────────────────
    // If a lower-priority event has been waiting longer than STARVATION_MAX_WAIT_MS,
    // promote it to dispatch just after all P1 events so it runs this pass.
    // This guarantees INPUT / HEARTBEAT events eventually run under sustained P1-P2 load.
    const p1Boundary = ready.findIndex(e => (EVENT_PRIORITY[e.eventType] ?? 5) > 1);
    const insertAt   = p1Boundary === -1 ? 0 : p1Boundary;

    for (let i = ready.length - 1; i >= insertAt; i--) {
      if ((now - ready[i].addedAt) >= STARVATION_MAX_WAIT_MS) {
        const [starving] = ready.splice(i, 1);
        ready.splice(insertAt, 0, starving); // promote to just after P1 block
        break; // one promotion per pass — prevents priority inversion
      }
    }

    for (const { eventType, payload, evtVersion } of ready) {
      const envelope = { eventType, payload, version: evtVersion, timestamp: Date.now() };
      this._dispatch(eventType, envelope);
    }
  }

  _debugLog(action, type, evtV, stateV, drift) {
    // Only log in dev; suppressed in production
    if (typeof process !== 'undefined' && process.env?.NODE_ENV === 'production') return;
    // Uncomment for verbose debugging:
    // console.debug(`[EventBus] ${action} "${type}" evtV=${evtV} stateV=${stateV} drift=${drift}`);
  }
}
