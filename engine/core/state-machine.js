/**
 * MIDAN Engine — StateMachine
 * 10 system states. Transitions are driven by signal completeness,
 * interpretation stability, and real input events — never arbitrary timers.
 */

import { SYSTEM_STATE, EVENT, LAYER_TIMINGS, UNCERTAINTY } from './constants.js';

// Valid transitions: from → Set<to>
const TRANSITIONS = {
  [SYSTEM_STATE.IDLE]:         new Set([SYSTEM_STATE.LISTENING]),
  [SYSTEM_STATE.LISTENING]:    new Set([SYSTEM_STATE.RECEIVING, SYSTEM_STATE.IDLE]),
  [SYSTEM_STATE.RECEIVING]:    new Set([SYSTEM_STATE.INTERPRETING, SYSTEM_STATE.UNCERTAIN, SYSTEM_STATE.PAUSED, SYSTEM_STATE.ANALYZING]),
  [SYSTEM_STATE.INTERPRETING]: new Set([SYSTEM_STATE.ANALYZING, SYSTEM_STATE.PAUSED, SYSTEM_STATE.UNCERTAIN]),
  [SYSTEM_STATE.ANALYZING]:    new Set([SYSTEM_STATE.REVEALING, SYSTEM_STATE.PAUSED]),
  [SYSTEM_STATE.REVEALING]:    new Set([SYSTEM_STATE.COMPLETED, SYSTEM_STATE.PAUSED]),
  [SYSTEM_STATE.WAITING]:      new Set([SYSTEM_STATE.RECEIVING, SYSTEM_STATE.PAUSED]),
  [SYSTEM_STATE.PAUSED]:       new Set([SYSTEM_STATE.RECEIVING, SYSTEM_STATE.REDIRECTED, SYSTEM_STATE.ANALYZING, SYSTEM_STATE.INTERPRETING]),
  [SYSTEM_STATE.REDIRECTED]:   new Set([SYSTEM_STATE.INTERPRETING]),
  [SYSTEM_STATE.COMPLETED]:    new Set([SYSTEM_STATE.LISTENING, SYSTEM_STATE.PAUSED, SYSTEM_STATE.REDIRECTED]),
  [SYSTEM_STATE.UNCERTAIN]:    new Set([SYSTEM_STATE.RECEIVING, SYSTEM_STATE.INTERPRETING]),
};

// UNCERTAIN is an overlay — these states can also be UNCERTAIN simultaneously
const UNCERTAIN_CAPABLE = new Set([
  SYSTEM_STATE.RECEIVING,
  SYSTEM_STATE.INTERPRETING,
]);

export class StateMachine {
  #current;
  #eventBus;
  #store;
  #analysisTimer;     // timer for ANALYSIS_TRIGGER delay
  #uncertaintyActive; // tracks UNCERTAIN overlay separately from main state

  /**
   * @param {import('./store.js').SessionStore} store
   * @param {import('./event-bus.js').EventBus} eventBus
   */
  constructor(store, eventBus) {
    this.#store          = store;
    this.#eventBus       = eventBus;
    this.#current        = SYSTEM_STATE.IDLE;
    this.#analysisTimer  = null;
    this.#uncertaintyActive = false;

    this._bindEvents();
  }

  get current() { return this.#current; }

  /**
   * Attempt a transition. Returns true if accepted.
   * @param {string} toState
   * @param {Object} [meta] - optional metadata to attach to transition event
   */
  transition(toState, meta = {}) {
    const fromState = this.#current;

    if (!TRANSITIONS[fromState]?.has(toState)) {
      console.warn(`[StateMachine] Invalid transition: ${fromState} → ${toState}`);
      return false;
    }

    this.#current = toState;

    this.#store.update(
      () => ({ system_state: toState }),
      { sourceLayer: 'STATE_MACHINE' },
    );

    this.#eventBus.emit(
      EVENT.STATE_TRANSITION,
      { from: fromState, to: toState, ...meta },
    );

    this._onEnter(toState, fromState, meta);
    return true;
  }

  /**
   * Evaluate whether UNCERTAIN overlay should activate/deactivate.
   * Called after every completeness or signal registry update.
   */
  evaluateUncertainty() {
    const state       = this.#store.getState();
    const completeness = state.signals.completeness;
    const raw          = state.input.raw;
    const registry     = state.signals.registry;

    const allTier2     = registry.length > 0 && registry.every(s => s.tier === 'TIER2');
    const longNoSignal = raw.length >= UNCERTAINTY.LONG_NO_SIGNAL_CHARS && registry.length === 0;
    const lowComplete  = completeness < UNCERTAINTY.LOW_COMPLETENESS;

    const shouldBeUncertain = lowComplete || allTier2 || longNoSignal;

    if (shouldBeUncertain && !this.#uncertaintyActive) {
      this.#uncertaintyActive = true;
      this.#eventBus.emit(EVENT.UNCERTAIN_ACTIVATED, { reason: { lowComplete, allTier2, longNoSignal } });
    } else if (!shouldBeUncertain && this.#uncertaintyActive) {
      this.#uncertaintyActive = false;
      this.#eventBus.emit(EVENT.UNCERTAIN_DEACTIVATED, {});
    }
  }

  get isUncertain() { return this.#uncertaintyActive; }

  // ── Internal: event-driven auto-transitions ─────────────────────────────

  _bindEvents() {
    // Raw input → RECEIVING
    this.#eventBus.on(EVENT.INPUT_RAW_UPDATED, ({ payload }) => {
      if (this.#current === SYSTEM_STATE.LISTENING) {
        this.transition(SYSTEM_STATE.RECEIVING, { trigger: 'keydown' });
      }
      // Keydown during analysis → PAUSED
      if (this.#current === SYSTEM_STATE.ANALYZING || this.#current === SYSTEM_STATE.REVEALING) {
        this._handleInterruption();
      }
    });

    // Completeness crosses 0.50 → INTERPRETING
    this.#eventBus.on(EVENT.COMPLETENESS_UPDATED, ({ payload }) => {
      if (this.#current === SYSTEM_STATE.RECEIVING || this.#current === SYSTEM_STATE.UNCERTAIN) {
        if (payload.completeness >= 0.50) {
          this.transition(SYSTEM_STATE.INTERPRETING, { trigger: 'completeness_threshold' });
          this._scheduleAnalysis();
        }
      }
      this.evaluateUncertainty();
    });

    // Signal registry changed → re-evaluate uncertainty
    this.#eventBus.on(EVENT.SIGNAL_DETECTED, () => this.evaluateUncertainty());
    this.#eventBus.on(EVENT.SIGNAL_REMOVED,  () => this.evaluateUncertainty());

    // Interpretation updated → reset analysis countdown
    this.#eventBus.on(EVENT.INTERPRETATION_UPDATED, () => {
      if (this.#current === SYSTEM_STATE.INTERPRETING) {
        this._scheduleAnalysis();
      }
    });

    // Pipeline completed → REVEALING
    this.#eventBus.on(EVENT.PIPELINE_COMPLETED, () => {
      if (this.#current === SYSTEM_STATE.ANALYZING) {
        this.transition(SYSTEM_STATE.REVEALING, { trigger: 'pipeline_complete' });
      }
    });

    // All layers revealed → COMPLETED
    this.#eventBus.on(EVENT.OUTPUT_LAYER_REVEALED, ({ payload }) => {
      if (payload.isFinal && this.#current === SYSTEM_STATE.REVEALING) {
        this.transition(SYSTEM_STATE.COMPLETED, { trigger: 'all_layers_revealed' });
      }
    });

    // COMPLETED + new input → LISTENING (then RECEIVING via INPUT_RAW_UPDATED)
    this.#eventBus.on(EVENT.INPUT_RAW_UPDATED, () => {
      if (this.#current === SYSTEM_STATE.COMPLETED) {
        this.transition(SYSTEM_STATE.LISTENING, { trigger: 'new_input_after_complete' });
      }
    });
  }

  _scheduleAnalysis() {
    clearTimeout(this.#analysisTimer);

    const _normalize    = s => s.trim().replace(/\s+/g, ' ');
    const rawAtSchedule = _normalize(this.#store.getState().input.raw);

    this.#analysisTimer = setTimeout(() => {
      const state = this.#store.getState();
      const now   = Date.now();

      // Semantic stability: ignore whitespace changes, minor edits (Fix 1)
      if (_normalize(state.input.raw) !== rawAtSchedule) return;

      // Rapid typing suppression: minimum 500ms silence window (Fix 2)
      // Uses state.input.last_change (set by Layer 1 on every keystroke)
      if ((now - (state.input.last_change ?? 0)) < 500) return;

      // State machine must still be in INTERPRETING
      if (this.#current !== SYSTEM_STATE.INTERPRETING) return;

      // Do not escalate uncertain inputs to analysis
      if (this.#uncertaintyActive) return;

      this.transition(SYSTEM_STATE.ANALYZING, { trigger: 'interpretation_stable' });
      this.#eventBus.emit(EVENT.PIPELINE_STARTED, {});
    }, LAYER_TIMINGS.ANALYSIS_TRIGGER);
  }

  _handleInterruption() {
    clearTimeout(this.#analysisTimer);
    this.transition(SYSTEM_STATE.PAUSED, { trigger: 'user_interrupted' });
  }

  _onEnter(state, from, meta) {
    if (state === SYSTEM_STATE.IDLE || state === SYSTEM_STATE.ANALYZING) {
      // Clear pending stability timer on IDLE (teardown) and on ANALYZING entry (Fix 4):
      // if manual submit raced with the timer, entering ANALYZING must cancel it to
      // prevent a second pipeline trigger after the user-initiated run is already live.
      clearTimeout(this.#analysisTimer);
    }
    if (state === SYSTEM_STATE.REDIRECTED) {
      // REDIRECTED → INTERPRETING immediately (new thread)
      setTimeout(() => this.transition(SYSTEM_STATE.INTERPRETING, { trigger: 'new_thread' }), 0);
    }
  }

  /** Tear down timers on engine shutdown. */
  destroy() {
    clearTimeout(this.#analysisTimer);
  }
}
