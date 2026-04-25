/**
 * MIDAN Engine — Layer 2: Interpreted Structure
 *
 * SCHEDULING: debounced macrotask — setTimeout(80ms normal, 160ms during PAUSED).
 * WRITES: zero direct store writes. Emits INTERPRETATION_UPDATED; StateReducer applies it.
 * PRIORITY: lowest of the three layers (macrotask, fires after L1 synchronous path and L3 timers).
 *
 * Granularity rule: only emit components that CHANGED vs. current interpretation.
 * user_override rule: the StateReducer (not this layer) enforces the user_override flag.
 * This layer computes and emits. The reducer decides what to apply.
 */

import { LAYER_TIMINGS, SYSTEM_STATE, EVENT } from '../core/constants.js';
import { extractInterpretation } from '../signals/detector.js';

export class InterpretedStructureLayer {
  #getState;
  #eventBus;
  #timer;

  /**
   * @param {Function} getState  - () => current SESSION_STATE (read-only)
   * @param {import('../core/event-bus.js').EventBus} eventBus
   */
  constructor(getState, eventBus) {
    this.#getState = getState;
    this.#eventBus = eventBus;
    this.#timer    = null;
  }

  /**
   * Schedule a re-extraction. Resets debounce timer on every call.
   * Called by the engine after every Layer 1 event.
   */
  schedule() {
    clearTimeout(this.#timer);
    const state = this.#getState();
    const delay = state.system_state === SYSTEM_STATE.PAUSED
      ? LAYER_TIMINGS.LAYER_2_PAUSED
      : LAYER_TIMINGS.LAYER_2_NORMAL;

    // Capture version at schedule time for staleness reference
    const scheduledVersion = state.version;

    this.#timer = setTimeout(() => this._run(scheduledVersion), delay);
  }

  cancel() {
    clearTimeout(this.#timer);
    this.#timer = null;
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  _run(scheduledVersion) {
    const state = this.#getState();
    const raw   = state.input.raw;

    if (!raw || raw.trim().length === 0) return;

    // Compute changed interpretation components
    const changes = extractInterpretation(raw, state.interpretation);
    if (Object.keys(changes).length === 0) return;

    // Emit with origin version = version when this pass fires (not when scheduled)
    this.#eventBus.emit(
      EVENT.INTERPRETATION_UPDATED,
      {
        components:       changes,
        scheduledVersion, // informational: version at schedule time
      },
      state.version, // origin version = current state version at emit time
    );
  }
}
