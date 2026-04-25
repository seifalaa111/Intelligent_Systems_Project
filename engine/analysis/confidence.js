/**
 * MIDAN Engine — ConfidenceEngine
 * Per-conclusion confidence tiers. Smoothing. Reversal guard.
 * Confidence is per-conclusion, never per-analysis (invariant #10).
 */

import {
  CONFIDENCE_TIER, TIMING, REVERSAL_GUARD_MS, EVENT,
  SIGNAL_TYPE,
} from '../core/constants.js';

// Stabilization window: 160ms (2u) — no signal registry changes for 160ms
const STABILIZATION_MS = 160;

// Transition durations (spec exact)
const TRANSITION = {
  ADJACENT:     320, // 4u — upward or downward adjacent tier
  SKIP_STEP1:   160, // 4u pass-through at intermediate tier (LOW→HIGH passes through MODERATE for 160ms)
  SKIP_STEP2:   320, // then advance to HIGH
  DOWNWARD:     320, // 4u direct, no intermediate
};

const TIER_ORDER = ['LOW', 'MODERATE', 'HIGH'];

export class ConfidenceEngine {
  #store;
  #eventBus;
  #stabilizationTimer;
  #pendingTransition;

  constructor(store, eventBus) {
    this.#store              = store;
    this.#eventBus           = eventBus;
    this.#stabilizationTimer = null;
    this.#pendingTransition  = null;
  }

  /**
   * Called after signal registry changes (from Layer 3 or assumption confirmation).
   * Schedules a re-evaluation after the 160ms stabilization window.
   */
  scheduleReEvaluation() {
    clearTimeout(this.#stabilizationTimer);
    this.#stabilizationTimer = setTimeout(() => {
      this._evaluate();
    }, STABILIZATION_MS);
  }

  /**
   * Update per-conclusion confidence after backend analysis response.
   * Maps backend outputs directly — no normalization, no synthetic values.
   *
   * @param {Object} conclusionMap - { [conclusionId]: 'HIGH' | 'MODERATE' | 'LOW' }
   * @param {number} capturedVersion
   */
  applyFromBackend(conclusionMap, capturedVersion) {
    const state = this.#store.getState();

    this.#store.update(
      (current) => ({
        confidence: {
          ...current.confidence,
          per_conclusion: { ...current.confidence.per_conclusion, ...conclusionMap },
          overall:        this._computeOverall(conclusionMap),
        },
      }),
      { sourceLayer: 'API', capturedVersion },
    );

    this.#eventBus.emit(EVENT.CONFIDENCE_CHANGED, { conclusionMap }, this.#store.getState().version);
  }

  /**
   * Called when an assumption is confirmed (promotes signal confidence, recalculates per-conclusion).
   * @param {string} assumptionId
   * @param {Array}  affectedConclusions - conclusion IDs that depended on this assumption
   * @param {number} capturedVersion
   */
  onAssumptionConfirmed(assumptionId, affectedConclusions, capturedVersion) {
    const state   = this.#store.getState();
    const updated = { ...state.confidence.per_conclusion };

    // Remove this assumption from load_bearing; upgrade affected conclusions
    const loadBearing = state.confidence.load_bearing.filter(id => id !== assumptionId);
    for (const cid of affectedConclusions) {
      if (updated[cid] === CONFIDENCE_TIER.LOW) {
        updated[cid] = CONFIDENCE_TIER.MODERATE;
      }
    }

    this.#store.update(
      (current) => ({
        confidence: {
          ...current.confidence,
          per_conclusion: updated,
          load_bearing:   loadBearing,
          overall:        this._computeOverall(updated),
        },
      }),
      { sourceLayer: 'USER', capturedVersion },
    );

    this.#eventBus.emit(
      EVENT.CONFIDENCE_CHANGED,
      { trigger: 'assumption_confirmed', assumptionId, affectedConclusions },
      this.#store.getState().version,
    );
  }

  /**
   * Called when an assumption is rejected (signal returns to ABSENT, retraction protocol triggered).
   * @param {string} assumptionId
   * @param {Array}  affectedConclusions
   * @param {number} capturedVersion
   */
  onAssumptionRejected(assumptionId, affectedConclusions, capturedVersion) {
    const state   = this.#store.getState();
    const updated = { ...state.confidence.per_conclusion };

    for (const cid of affectedConclusions) {
      updated[cid] = CONFIDENCE_TIER.LOW;
    }

    this.#store.update(
      (current) => ({
        confidence: {
          ...current.confidence,
          per_conclusion: updated,
          overall:        this._computeOverall(updated),
        },
      }),
      { sourceLayer: 'USER', capturedVersion },
    );

    this.#eventBus.emit(
      EVENT.CONFIDENCE_CHANGED,
      { trigger: 'assumption_rejected', assumptionId, affectedConclusions },
      this.#store.getState().version,
    );
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  /**
   * Re-evaluate confidence based on current signal registry.
   * Only runs after stabilization window — no rapid oscillation.
   */
  _evaluate() {
    const state    = this.#store.getState();
    const registry = state.signals.registry;

    // Apply compound rules from arch spec
    const hasProblem  = registry.some(s => s.type === SIGNAL_TYPE.PROBLEM);
    const hasMarket   = registry.some(s => s.type === SIGNAL_TYPE.GEOGRAPHIC || s.type === SIGNAL_TYPE.AUDIENCE);
    const hasMechanism = registry.some(s => s.type === SIGNAL_TYPE.DOMAIN || s.type === 'mechanism');
    const hasLoadBearing = state.confidence.load_bearing.length > 0;

    let tier;
    if (!hasProblem) {
      tier = CONFIDENCE_TIER.LOW;        // Tier 1 absent → cap LOW
    } else if (!hasMarket) {
      tier = CONFIDENCE_TIER.MODERATE;   // Tier 2 absent → cap MODERATE
    } else if (hasProblem && hasMarket && hasMechanism && !hasLoadBearing) {
      tier = CONFIDENCE_TIER.HIGH;       // all strong signals, no load-bearing assumptions
    } else if (hasProblem && hasMarket) {
      tier = hasLoadBearing ? CONFIDENCE_TIER.MODERATE : CONFIDENCE_TIER.MODERATE;
    } else {
      tier = CONFIDENCE_TIER.LOW;
    }

    const current = state.confidence.overall;
    if (tier === current) return;

    this._transition(current, tier, state.version);
  }

  /**
   * Execute a confidence transition with correct timing per spec.
   * Upward adjacent: 320ms
   * Upward skip (LOW→HIGH): pass through MODERATE 160ms, then advance 320ms
   * Downward: 320ms direct
   */
  _transition(from, to, originVersion) {
    const fromIdx = TIER_ORDER.indexOf(from);
    const toIdx   = TIER_ORDER.indexOf(to);
    const isUp    = toIdx > fromIdx;
    const isSkip  = Math.abs(toIdx - fromIdx) > 1;

    // Reversal guard: check if the trigger signal was just added and removed
    const state = this.#store.getState();
    const triggerId = state.confidence._transition_trigger_id;
    if (triggerId) {
      const triggerSignal = state.signals.registry.find(s => s.id === triggerId);
      if (!triggerSignal) {
        // Trigger signal was removed — suppress upward transition if recently started
        if (isUp && state.confidence._transition_start) {
          const elapsed = Date.now() - state.confidence._transition_start;
          if (elapsed < REVERSAL_GUARD_MS) return; // TRANSIENT — suppress
        }
      }
    }

    if (isUp && isSkip) {
      // LOW → HIGH: pass through MODERATE for 160ms first
      this._writeTier(CONFIDENCE_TIER.MODERATE, 'in_progress', originVersion);
      setTimeout(() => {
        const current = this.#store.getState();
        if (current.version > originVersion + 5) return; // too stale
        this._writeTier(to, 'complete', originVersion);
      }, TRANSITION.SKIP_STEP1 + TRANSITION.SKIP_STEP2);
    } else {
      this._writeTier(to, 'complete', originVersion);
    }
  }

  _writeTier(tier, status, originVersion) {
    this.#store.update(
      (current) => ({
        confidence: {
          ...current.confidence,
          overall:            tier,
          _transition_start:  status === 'in_progress' ? Date.now() : null,
          _transition_from_tier: status === 'in_progress' ? current.confidence.overall : null,
        },
      }),
      { sourceLayer: 'CONFIDENCE', capturedVersion: originVersion },
    );

    this.#eventBus.emit(
      EVENT.CONFIDENCE_CHANGED,
      { tier, status },
      this.#store.getState().version,
    );
  }

  /** overall confidence = min(per_conclusion values) — per spec */
  _computeOverall(perConclusion) {
    const values = Object.values(perConclusion);
    if (values.length === 0) return CONFIDENCE_TIER.LOW;
    if (values.includes(CONFIDENCE_TIER.LOW))      return CONFIDENCE_TIER.LOW;
    if (values.includes(CONFIDENCE_TIER.MODERATE)) return CONFIDENCE_TIER.MODERATE;
    return CONFIDENCE_TIER.HIGH;
  }

  destroy() {
    clearTimeout(this.#stabilizationTimer);
  }
}
