/**
 * MIDAN Engine — RecomputationEngine
 *
 * DEPENDENCY MODEL:
 *   FORWARD_GRAPH  — signal type or interpretation component → which conclusions recompute
 *   BACKWARD_GRAPH — conclusion → which signal types support it
 *
 * RETRACTION RULE:
 *   When signal S is removed, walk FORWARD_GRAPH[S] for every affected conclusion C.
 *   Then check BACKWARD_GRAPH[C]: if NO remaining registry signal satisfies any
 *   backward edge for C, retract C explicitly (never silently remove).
 *
 * RECOMPUTE RULE:
 *   Affected conclusions that are NOT retracted are scheduled for granular recompute
 *   via the API adapter — never re-derived in the frontend.
 */

import { EVENT } from '../core/constants.js';

// ── Forward dependency graph ──────────────────────────────────────────────────
// Key: signal type (string) or 'component:<name>' for L2 interpretation updates
// Value: conclusion IDs that must recompute when this source changes
const FORWARD_GRAPH = Object.freeze({
  // ── Tier 1 signals ──────────────────────────────────────────────────────────
  problem: [
    'opportunity_assessment',
    'feasibility_assessment',
    'risk_map',
    'summary',
  ],
  geographic: [
    'opportunity_assessment',
    'demand_signal',
    'timing_assessment',
  ],
  domain: [
    'feasibility_assessment',
    'defensibility_assessment',
    'execution_risk',
  ],
  audience: [
    'opportunity_assessment',
    'demand_signal',
  ],
  monetization_absent: [
    'opportunity_assessment',
    'risk_map',
  ],

  // ── Tier 2 signals ──────────────────────────────────────────────────────────
  friction: [
    'execution_risk',
    'risk_map',
    'timing_assessment',
  ],
  structural_complexity: [
    'feasibility_assessment',
    'execution_risk',
    'risk_map',
  ],
  value_ambiguity: [
    'feasibility_assessment',
    'defensibility_assessment',
    'risk_map',
  ],
  behavioral: [
    'demand_signal',
    'timing_assessment',
  ],

  // ── Interpretation components (L2 update → conclusions recompute) ────────────
  'component:problem': [
    'opportunity_assessment',
    'feasibility_assessment',
    'risk_map',
    'summary',
  ],
  'component:market': [
    'opportunity_assessment',
    'demand_signal',
    'timing_assessment',
  ],
  'component:mechanism': [
    'feasibility_assessment',
    'defensibility_assessment',
    'execution_risk',
    'risk_map',
  ],
  'component:context': [
    'execution_risk',
    'risk_map',
    'timing_assessment',
  ],
  'component:traction': [
    'opportunity_assessment',
    'demand_signal',
  ],
});

// ── Backward dependency graph ─────────────────────────────────────────────────
// Key: conclusion ID
// Value: signal types that support this conclusion.
// A conclusion with ZERO remaining support signals is retracted.
const BACKWARD_GRAPH = Object.freeze({
  opportunity_assessment:   ['problem', 'geographic', 'audience', 'monetization_absent'],
  demand_signal:            ['geographic', 'audience', 'behavioral'],
  feasibility_assessment:   ['problem', 'domain', 'structural_complexity', 'value_ambiguity'],
  defensibility_assessment: ['domain', 'value_ambiguity'],
  execution_risk:           ['domain', 'friction', 'structural_complexity'],
  risk_map:                 ['problem', 'monetization_absent', 'friction', 'structural_complexity', 'value_ambiguity'],
  timing_assessment:        ['geographic', 'friction', 'behavioral'],
  summary:                  ['problem'],
});

// ── Phase → conclusion ID mapping (for stale-layer recompute) ─────────────────
// Mirrors the pipeline's three-phase partition.
const PHASE_CONCLUSIONS = Object.freeze({
  1: ['opportunity_assessment', 'demand_signal'],
  2: ['feasibility_assessment', 'defensibility_assessment', 'execution_risk'],
  3: ['risk_map', 'timing_assessment', 'summary'],
});

// ── RecomputationEngine ───────────────────────────────────────────────────────

export class RecomputationEngine {
  #store;
  #eventBus;
  #apiAdapter;
  #recomputeTimers;

  constructor(store, eventBus, apiAdapter) {
    this.#store           = store;
    this.#eventBus        = eventBus;
    this.#apiAdapter      = apiAdapter;
    this.#recomputeTimers = {};
  }

  /**
   * Conclusions affected when a source changes (forward traversal).
   *
   * @param {string} changedSource - signal type or 'component:<name>'
   * @returns {string[]}
   */
  computeAffected(changedSource) {
    return FORWARD_GRAPH[changedSource] ?? [];
  }

  /**
   * Conclusions that lose ALL support when a signal is removed (backward traversal).
   * Checks the CURRENT registry — called after the signal has already been removed.
   *
   * @param {string} removedSignalType
   * @returns {string[]} conclusionIds to retract
   */
  computeRetracted(removedSignalType) {
    const state         = this.#store.getState();
    const registry      = state.signals.registry;
    const remainingTypes = new Set(registry.map(s => s.type));
    const retracted     = [];

    for (const [conclusionId, requiredSignals] of Object.entries(BACKWARD_GRAPH)) {
      if (!requiredSignals.includes(removedSignalType)) continue;
      if (!state.output.conclusions[conclusionId])      continue;

      const stillSupported = requiredSignals.some(sig => remainingTypes.has(sig));
      if (!stillSupported) retracted.push(conclusionId);
    }

    return retracted;
  }

  /**
   * Full signal-removal handler:
   * 1. Retract conclusions with no remaining support.
   * 2. Schedule recompute for affected conclusions that are NOT retracted.
   *
   * @param {string} removedSignalType
   */
  onSignalRemoved(removedSignalType) {
    const toRetract   = this.computeRetracted(removedSignalType);
    const retractSet  = new Set(toRetract);
    const toRecompute = this.computeAffected(removedSignalType)
      .filter(id => !retractSet.has(id));

    for (const conclusionId of toRetract) {
      this.retract(conclusionId, `"${removedSignalType}" signal removed`);
    }

    if (toRecompute.length > 0) {
      this._scheduleRecompute(toRecompute, { changedSource: removedSignalType });
    }
  }

  /**
   * Full signal-addition handler: schedule recompute for affected conclusions.
   *
   * @param {string} addedSignalType
   */
  onSignalAdded(addedSignalType) {
    const affected = this.computeAffected(addedSignalType);
    if (affected.length > 0) {
      this._scheduleRecompute(affected, { changedSource: addedSignalType });
    }
  }

  /**
   * Interpretation component update handler: schedule recompute for affected conclusions.
   *
   * @param {string} componentKey - e.g. 'problem', 'market'
   */
  onComponentUpdated(componentKey) {
    const affected = this.computeAffected(`component:${componentKey}`);
    if (affected.length > 0) {
      this._scheduleRecompute(affected, { changedSource: `component:${componentKey}` });
    }
  }

  /**
   * Recompute a specific set of conclusions via the API adapter.
   * Never re-derives intelligence in the frontend — backend is authoritative.
   *
   * @param {string[]} conclusionIds
   * @param {Object}   context - { changedSource, signal, component }
   */
  async recompute(conclusionIds, context = {}) {
    if (!conclusionIds || conclusionIds.length === 0) return;

    const state = this.#store.getState();
    this._markStale(conclusionIds, state.version);

    try {
      const result = await this.#apiAdapter.recomputeConclusions(
        conclusionIds,
        state,
        context,
      );

      if (!result) return;

      for (const [conclusionId, newText] of Object.entries(result)) {
        this._updateConclusion(conclusionId, newText, state.version);
      }
    } catch (err) {
      console.error('[Recomputation] error during granular recompute', err);
      // Leave stale-flagged state visible rather than retract on transient error
    }
  }

  /**
   * Retraction protocol (spec-exact):
   * Conclusion fades out (visual layer handles opacity→0, 300ms),
   * replaced by an italic retraction statement explaining what was removed.
   * Never silently removed.
   *
   * @param {string} conclusionId
   * @param {string} reason
   */
  retract(conclusionId, reason) {
    const state = this.#store.getState();
    if (!state.output.conclusions[conclusionId]) return;

    const retractedText = `Removed: ${reason}`;

    this.#store.update(
      (current) => ({
        output: {
          ...current.output,
          conclusions: {
            ...current.output.conclusions,
            [conclusionId]: {
              ...current.output.conclusions[conclusionId],
              retracted:       true,
              retracted_at:    Date.now(),
              retraction_text: retractedText,
            },
          },
        },
      }),
      { sourceLayer: 'RECOMPUTE' },
    );

    this.#eventBus.emit(
      EVENT.CONCLUSION_RETRACTED,
      { conclusionId, reason: retractedText },
      this.#store.getState().version,
    );
  }

  /**
   * Check all revealed phases for staleness and schedule granular recompute.
   * Called by state machine on version increment.
   */
  checkStaleLayers() {
    const state = this.#store.getState();
    const { layers_revealed, layer_versions } = state.output;

    for (const phaseId of layers_revealed) {
      const renderedAtVersion = layer_versions[phaseId];
      if (renderedAtVersion == null) continue;
      if (state.version <= renderedAtVersion) continue;

      const phaseConclusionIds = PHASE_CONCLUSIONS[phaseId] ?? [];
      const revealed = phaseConclusionIds.filter(id => state.output.conclusions[id]);
      if (revealed.length > 0) {
        this._scheduleRecompute(revealed, { phaseId, trigger: 'stale_layer' });
      }
    }
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  _markStale(conclusionIds, originVersion) {
    this.#eventBus.emit(
      EVENT.CONCLUSION_UPDATED,
      { conclusionIds, status: 'stale', originVersion },
      this.#store.getState().version,
    );
  }

  _updateConclusion(conclusionId, newText, originVersion) {
    this.#store.update(
      (current) => ({
        output: {
          ...current.output,
          conclusions: {
            ...current.output.conclusions,
            [conclusionId]: {
              ...current.output.conclusions[conclusionId],
              text:       newText,
              retracted:  false,
              updated_at: Date.now(),
              version:    current.version,
            },
          },
        },
      }),
      { sourceLayer: 'RECOMPUTE', capturedVersion: originVersion },
    );

    this.#eventBus.emit(
      EVENT.CONCLUSION_UPDATED,
      { conclusionId, status: 'updated', text: newText },
      this.#store.getState().version,
    );
  }

  // Debounced recompute scheduler — deduplicated by sorted conclusion list key
  _scheduleRecompute(conclusionIds, context) {
    const key = [...conclusionIds].sort().join(',');
    clearTimeout(this.#recomputeTimers[key]);

    this.#recomputeTimers[key] = setTimeout(async () => {
      delete this.#recomputeTimers[key];
      await this.recompute(conclusionIds, context);
    }, 160); // 2u debounce — prevent recompute thrashing on rapid signal changes
  }
}
