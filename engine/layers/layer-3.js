/**
 * MIDAN Engine — Layer 3: Signal Layer
 *
 * SCHEDULING:
 *   Tier 1 — word-boundary trigger (last char typed is boundary char).
 *             Fires synchronously within the keydown handler, same tick as Layer 1.
 *   Tier 2 — setTimeout 180ms after word boundary (260ms during PAUSED).
 *             Fires as a macrotask, BEFORE Layer 2's 80ms debounce clears in most cases.
 *
 * WRITES: zero direct store writes. Emits SIGNAL_DETECTED / SIGNAL_REMOVED;
 *         StateReducer applies them.
 *
 * AUTHORITY: Layer 3 owns signal classification (per conflict resolution rule).
 *            Layer 2 owns component text values.
 *            These namespaces are separate — no conflict possible.
 *
 * Layer 3 Tier 1 fires synchronously on word boundary in the same tick as Layer 1.
 * This means L3 Tier 1 state changes complete BEFORE L2's debounced timer fires,
 * ensuring completeness-triggered state transitions (e.g., INTERPRETING) happen
 * before L2 interpretation is applied.
 */

import { LAYER_TIMINGS, SYSTEM_STATE, EVENT } from '../core/constants.js';
import { detectTier1, detectTier2 } from '../signals/detector.js';

const WORD_BOUNDARY_CHARS = new Set([
  ' ', '.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '"', "'", '\n', '\t',
]);

export class SignalLayer {
  #getState;
  #eventBus;
  #tier2Timer;
  #lastKnownSignalTypes; // Set<string> — tracks what we've emitted as detected

  /**
   * @param {Function} getState  - () => current SESSION_STATE (read-only)
   * @param {import('../core/event-bus.js').EventBus} eventBus
   */
  constructor(getState, eventBus) {
    this.#getState            = getState;
    this.#eventBus            = eventBus;
    this.#tier2Timer          = null;
    this.#lastKnownSignalTypes = new Set();
  }

  /**
   * Process input after a keydown event.
   * Tier 1 runs synchronously if last character is a word boundary.
   * Tier 2 is always scheduled (independent timer chain).
   *
   * @param {string} text - current raw input value
   */
  process(text) {
    const state      = this.#getState();
    const lastChar   = text.length > 0 ? text[text.length - 1] : '';
    const isWordBoundary = WORD_BOUNDARY_CHARS.has(lastChar);

    if (isWordBoundary) {
      this._runTier1(text, state.version);
    }

    this._scheduleTier2(text, state.system_state, state.version);
  }

  /**
   * Apply a user-confirmed signal reclassification (Type B correction).
   * Emits SIGNAL_REMOVED (old) + SIGNAL_DETECTED (new type).
   *
   * @param {string} signalId
   * @param {string} oldType
   * @param {string} newType
   */
  reclassify(signalId, oldType, newType) {
    const state = this.#getState();
    this.#eventBus.emit(EVENT.SIGNAL_REMOVED, { type: oldType, reclassified: true }, state.version);
    this.#eventBus.emit(
      EVENT.SIGNAL_DETECTED,
      { signal: { id: signalId, type: newType, tier: 'TIER1', added_at: Date.now() }, reclassified: true },
      state.version,
    );
  }

  cancel() {
    clearTimeout(this.#tier2Timer);
    this.#tier2Timer = null;
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  _runTier1(text, originVersion) {
    const detected = detectTier1(text);
    const detectedTypes = new Set(detected.map(s => s.type));
    const tier1Types    = new Set(['problem', 'geographic', 'domain', 'audience', 'monetization_absent']);

    // Emit SIGNAL_DETECTED for newly found signals
    for (const sig of detected) {
      if (!this.#lastKnownSignalTypes.has(sig.type)) {
        this.#eventBus.emit(EVENT.SIGNAL_DETECTED, { signal: sig }, originVersion);
        this.#lastKnownSignalTypes.add(sig.type);
      }
    }

    // Emit SIGNAL_REMOVED for Tier 1 signals no longer in the text
    for (const knownType of [...this.#lastKnownSignalTypes]) {
      if (tier1Types.has(knownType) && !detectedTypes.has(knownType)) {
        this.#eventBus.emit(EVENT.SIGNAL_REMOVED, { type: knownType }, originVersion);
        this.#lastKnownSignalTypes.delete(knownType);
      }
    }
  }

  _scheduleTier2(text, systemState, originVersion) {
    clearTimeout(this.#tier2Timer);
    const delay = systemState === SYSTEM_STATE.PAUSED
      ? LAYER_TIMINGS.LAYER_3_TIER2_PAUSED
      : LAYER_TIMINGS.LAYER_3_TIER2_NORMAL;

    this.#tier2Timer = setTimeout(() => {
      this._runTier2(delay, originVersion);
    }, delay);
  }

  _runTier2(delay, scheduledOriginVersion) {
    const state     = this.#getState();
    const text      = state.input.raw;
    const detected  = detectTier2(text);
    const tier2Types = new Set(['friction', 'structural_complexity', 'value_ambiguity', 'behavioral']);
    const detectedTypes = new Set(detected.map(s => s.type));

    // Emit SIGNAL_DETECTED for new Tier 2 signals
    for (const sig of detected) {
      if (!this.#lastKnownSignalTypes.has(sig.type)) {
        this.#eventBus.emit(EVENT.SIGNAL_DETECTED, { signal: sig }, state.version);
        this.#lastKnownSignalTypes.add(sig.type);
      }
    }

    // Emit SIGNAL_REMOVED for Tier 2 signals no longer inferred
    for (const knownType of [...this.#lastKnownSignalTypes]) {
      if (tier2Types.has(knownType) && !detectedTypes.has(knownType)) {
        this.#eventBus.emit(EVENT.SIGNAL_REMOVED, { type: knownType }, state.version);
        this.#lastKnownSignalTypes.delete(knownType);
      }
    }
  }
}
