/**
 * MIDAN Engine — Completeness Calculator
 * Computes signals.completeness (0.0–1.0) from the signal registry.
 * Also provides isReady() for the CTA threshold.
 *
 * INTERPRETING threshold:   completeness >= 0.50
 * READY/CTA threshold:      isReady() per entry spec (≥2 distinct categories OR 1 strong Tier1 + 1 Tier2)
 * UNCERTAINTY threshold:    completeness < 0.30 (evaluated by StateMachine)
 */

import { COMPLETENESS_WEIGHT, SIGNAL_TIER, SIGNAL_TYPE, EVENT } from '../core/constants.js';

/**
 * Calculate completeness from signal registry.
 * Uses category weights — only one signal per category contributes.
 *
 * @param {Array} registry - signals.registry from SESSION_STATE
 * @returns {number} 0.0–1.0
 */
export function calculateCompleteness(registry) {
  if (!registry || registry.length === 0) return 0.0;

  // Collect unique signal types present
  const presentTypes = new Set(registry.map(s => s.type));

  let score = 0;
  for (const type of presentTypes) {
    const weight = COMPLETENESS_WEIGHT[type] ?? 0;
    score += weight;
  }

  return Math.max(0, Math.min(1.0, score));
}

/**
 * Evaluate CTA READY condition from entry spec:
 *   ≥2 distinct signal categories confirmed
 *   OR 1 Tier 1 strong (problem) + 1 Tier 2 inferred
 *
 * NOT READY: domain-only, single Tier 2, same-type chips only
 *
 * @param {Array} registry
 * @returns {boolean}
 */
export function isReady(registry) {
  if (!registry || registry.length === 0) return false;

  const types  = [...new Set(registry.map(s => s.type))];
  const tier1s = registry.filter(s => s.tier === SIGNAL_TIER.TIER1);
  const tier2s = registry.filter(s => s.tier === SIGNAL_TIER.TIER2);

  // Rule 1: ≥2 distinct categories
  if (types.length >= 2) return true;

  // Rule 2: 1 strong Tier1 (specifically problem) + 1 Tier2
  const hasProblem = tier1s.some(s => s.type === SIGNAL_TYPE.PROBLEM);
  if (hasProblem && tier2s.length >= 1) return true;

  return false;
}

/**
 * Returns LOW_SIGNAL_CONFIDENCE indicator state.
 * Fires when: ≥40 chars typed, 0 signals OR all Tier 2 only.
 *
 * @param {string} rawText
 * @param {Array}  registry
 * @returns {boolean}
 */
export function isLowSignalConfidence(rawText, registry) {
  if (!rawText || rawText.length < 40) return false;

  if (registry.length === 0) return true;
  if (registry.every(s => s.tier === SIGNAL_TIER.TIER2)) return true;

  return false;
}

/**
 * Writes updated completeness to SESSION_STATE via store.
 * Only writes if value changed.
 *
 * @param {import('../core/store.js').SessionStore} store
 * @param {import('../core/event-bus.js').EventBus} eventBus
 * @param {number} capturedVersion
 */
export function syncCompleteness(store, eventBus, capturedVersion) {
  const state    = store.getState();
  const registry = state.signals.registry;
  const newVal   = calculateCompleteness(registry);

  if (Math.abs(newVal - state.signals.completeness) < 0.001) return; // no meaningful change

  store.update(
    () => ({ signals: { ...state.signals, completeness: newVal } }),
    { sourceLayer: 'LAYER_3', capturedVersion },
  );

  eventBus.emit(
    EVENT.COMPLETENESS_UPDATED,
    {
      completeness: newVal,
      isReady:      isReady(registry),
      isLowSignal:  isLowSignalConfidence(state.input.raw, registry),
    },
    state.version,
  );
}
