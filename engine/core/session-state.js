/**
 * MIDAN Engine — SESSION_STATE
 * Single source of truth. Never mutated directly.
 * All updates return a new frozen state via applyUpdate().
 */

import { SYSTEM_STATE, PIPELINE_STATUS, CONFIDENCE_TIER } from './constants.js';

/**
 * Returns a fresh, frozen initial SESSION_STATE.
 * All fields match the spec exactly.
 * @returns {Readonly<Object>}
 */
export function createInitialState() {
  return deepFreeze({
    version: 0,

    input: {
      raw:         '',
      length:      0,
      last_change: null,   // timestamp (ms)
      change_type: null,   // 'add' | 'delete' | 'paste' | 'replace'
    },

    interpretation: {
      problem:   _emptyComponent(),
      market:    _emptyComponent(),
      mechanism: _emptyComponent(),
      context:   _emptyComponent(),
      traction:  _emptyComponent(),
    },

    signals: {
      registry:     [],    // [{ id, type, tier, rank, confidence, trigger, added_at, version }]
      primary_id:   null,
      completeness: 0.0,
      last_removed: null,  // { type, isTransient, removedAt } — set by registry.remove()
    },

    confidence: {
      overall:               CONFIDENCE_TIER.LOW,
      per_conclusion:        {},  // { [conclusion_id]: tier }
      load_bearing:          [],  // [assumption_id]
      _transition_trigger_id: null,
      _transition_from_tier:  null,
      _transition_start:      null,
    },

    assumptions: {
      active: [],
      // [{ id, fills_signal, supports_conclusion, value, confirmed, confirmed_type }]
    },

    output: {
      layers_revealed:  [],   // [int] — layer numbers revealed in order
      current_layer:    null,
      interrupted_at:   null, // layer_id | null
      layer_versions:   {},   // { [layer_id]: version_when_rendered }
      conclusions:      {},   // { [conclusion_id]: { text, signals, assumptions, retracted } }
    },

    pipeline_state: {
      status:       PIPELINE_STATUS.IDLE,
      current_phase: null,   // 1 | 2 | 3 | null
      checkpoints:  [],
      // [{ phase_id, completed_at, completed_conclusions, state_version }]
      interrupted_at: null,  // { phase_id, timestamp } | null
    },

    threads: {
      active_id: 'thread_0',
      registry:  [{
        id:           'thread_0',
        created_at:   Date.now(),
        state:        'active',
        summary:      null,
        base_version: 0,
        base:         null,   // COMPACT_BASE — set on first snapshot
        diff:         [],     // PATCH_SET (RFC 6902 array)
      }],
    },

    system_state: SYSTEM_STATE.IDLE,

    performance: {
      latency_tracker: {
        window:          10,
        samples:         [],   // last N actual latencies per phase
        rolling_average: null,
        sustained_load:  false,
        sustained_count: 0,
      },
      expected_latency: {
        interpreting: 100,
        analyzing:    200,
      },
      baseline_floors: {
        interpreting: 400,
        analyzing:    800,
      },
      adjusted_floors: {
        interpreting: 400,
        analyzing:    800,
      },
    },
  });
}

function _emptyComponent() {
  return {
    value:         null,
    confidence:    null, // tier: LOW | MODERATE | HIGH
    version:       0,
    user_override: false,
  };
}

/**
 * Applies a partial update to state, incrementing version.
 * Returns the new frozen state.
 *
 * @param {Object} currentState  - current frozen SESSION_STATE
 * @param {Function} updater     - receives currentState, returns partial update object
 * @param {Object} [options]
 * @param {string} [options.sourceLayer]   - which layer is writing ('LAYER_1'|'LAYER_2'|'LAYER_3'|'PIPELINE'|'API'|'USER')
 * @param {number} [options.capturedVersion] - version when this layer captured input; null = always valid
 * @returns {{ state: Object, accepted: boolean }}
 */
export function applyUpdate(currentState, updater, { sourceLayer = 'UNKNOWN', capturedVersion = null } = {}) {
  // Stale write check: if the raw input changed since this layer captured it,
  // reject writes from interpretation/signal layers (not input or API layers)
  if (capturedVersion !== null && sourceLayer !== 'LAYER_1' && sourceLayer !== 'API' && sourceLayer !== 'USER') {
    const drift = currentState.version - capturedVersion;
    // Allow up to 3 version drift (other layers may have written)
    // but reject if version has moved more than 10 (input changed significantly)
    if (drift > 10) {
      return { state: currentState, accepted: false, reason: 'stale' };
    }
  }

  const partial = updater(currentState);
  if (!partial || Object.keys(partial).length === 0) {
    return { state: currentState, accepted: true };
  }

  const nextState = deepFreeze(mergeDeep(currentState, partial, { version: currentState.version + 1 }));
  return { state: nextState, accepted: true };
}

/**
 * Deep merges objects (non-mutating). Arrays are replaced, not merged.
 */
export function mergeDeep(target, ...sources) {
  const result = Object.assign({}, target);
  for (const source of sources) {
    if (!source || typeof source !== 'object') continue;
    for (const key of Object.keys(source)) {
      const sv = source[key];
      const tv = result[key];
      if (sv && typeof sv === 'object' && !Array.isArray(sv) && tv && typeof tv === 'object' && !Array.isArray(tv)) {
        result[key] = mergeDeep(tv, sv);
      } else {
        result[key] = sv;
      }
    }
  }
  return result;
}

/**
 * Recursively freezes an object and all nested objects/arrays.
 * Returns the frozen object.
 */
export function deepFreeze(obj) {
  Object.freeze(obj);
  Object.keys(obj).forEach(key => {
    const val = obj[key];
    if (val && typeof val === 'object' && !Object.isFrozen(val)) {
      deepFreeze(val);
    }
  });
  return obj;
}

/**
 * Returns a mutable deep clone of a frozen state (for thread snapshots etc.)
 */
export function cloneState(state) {
  return JSON.parse(JSON.stringify(state));
}

/**
 * Checks whether a revealed layer's data is stale (its inputs changed since render).
 * @param {Object} state
 * @param {number} layerId
 * @returns {boolean}
 */
export function isLayerStale(state, layerId) {
  const renderedAt = state.output.layer_versions[layerId];
  if (renderedAt == null) return false;
  return state.version > renderedAt;
}
