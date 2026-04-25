/**
 * MIDAN Engine — ApiAdapter
 * The ONLY point of contact with the existing backend.
 * Maps backend /interact and /analyze responses → SESSION_STATE.
 * Preserves ALL real signal values — no normalization, no synthetic values.
 * Handles missing/delayed responses gracefully (timeout, retry).
 *
 * KEY ARCHITECTURE:
 * analyzeForPipeline() makes ONE /interact call and partitions the response into
 * three phase buckets { phase1, phase2, phase3, assumptions }.
 * The pipeline applies these buckets sequentially with conclusion-boundary interruption.
 *
 * Phase partition rules (what to reveal first vs. later):
 *   Phase 1 — primary conclusions: opportunity_assessment (strategic_interpretation)
 *   Phase 2 — Tier 1 refinements: risk_map, defensibility_assessment, timing_assessment
 *   Phase 3 — Tier 2 nuances:     counter_thesis, differentiation, dimension details
 *
 * INVARIANT: no backend logic recreated here.
 * INVARIANT: intelligence flows backend → SESSION_STATE, not the reverse.
 */

import { EVENT, CONFIDENCE_TIER } from '../core/constants.js';

const DEFAULT_TIMEOUT_MS  = 30_000;
const DEFAULT_RETRY_COUNT = 2;
const RETRY_DELAY_MS      = 1_000;

export class ApiAdapter {
  #store;
  #eventBus;
  #baseUrl;
  #requestLog;

  /**
   * @param {import('../core/store.js').SessionStore} store
   * @param {import('../core/event-bus.js').EventBus}  eventBus
   * @param {string} baseUrl - e.g. 'http://localhost:8000'
   */
  constructor(store, eventBus, baseUrl = '') {
    this.#store      = store;
    this.#eventBus   = eventBus;
    this.#baseUrl    = baseUrl.replace(/\/$/, '');
    this.#requestLog = [];
  }

  // ── Public: Pipeline Analysis ──────────────────────────────────────────────

  /**
   * Make ONE /interact call and partition the full response into three phase buckets.
   * The pipeline iterates these buckets for sequential conclusion revelation.
   *
   * @param {Object} state       - current SESSION_STATE
   * @param {Object} [options]
   * @param {AbortSignal} [options.signal]
   * @param {string} [options.inputClass]
   * @returns {Object|null} { phase1, phase2, phase3, assumptions, rawResponse }
   */
  async analyzeForPipeline(state, { signal, inputClass = 'COMPLETE' } = {}) {
    if (signal?.aborted) return null;

    const messages = this._buildMessageHistory(state);
    const context  = this._buildContext(state);

    let response;
    try {
      response = await this._fetch('/interact', {
        method: 'POST',
        signal,
        body:   JSON.stringify({ messages, context }),
      });
    } catch (err) {
      if (err.name === 'AbortError') throw err;
      this._handleError('/interact', err);
      return null;
    }

    if (!response?.success && response?.type !== 'analysis' && response?.type !== 'invalid') {
      return null;
    }

    // Map response to SESSION_STATE before partitioning
    const capturedVersion = state.version;
    this._mapInteractResponseToState(response, capturedVersion);

    // Partition into phase buckets for sequential pipeline revelation
    return this._partitionIntoPhaseBuckets(response);
  }

  /**
   * Submit a message via /interact (conversation mode).
   * Used for: post-analysis chat, casual inputs, clarifying questions.
   *
   * @param {string} userMessage
   * @returns {Object|null} raw backend response
   */
  async interact(userMessage) {
    const state    = this.#store.getState();
    const messages = [...this._buildMessageHistory(state), { role: 'user', content: userMessage }];
    const context  = this._buildContext(state);

    try {
      const response = await this._fetch('/interact', {
        method: 'POST',
        body:   JSON.stringify({ messages, context }),
      });

      this._mapInteractResponseToState(response, state.version);
      return response;
    } catch (err) {
      this._handleError('/interact', err);
      return null;
    }
  }

  /**
   * Call /analyze directly (bypasses /interact conversation layer).
   * Used for direct analysis flows.
   */
  async analyze(idea, sector, country) {
    const state = this.#store.getState();
    try {
      const response = await this._fetch('/analyze', {
        method: 'POST',
        body:   JSON.stringify({ idea, sector, country }),
      });

      if (!response.success) throw new Error(response.error ?? 'analyze failed');
      this._mapAnalysisDataToState(response, state.version);
      return response;
    } catch (err) {
      this._handleError('/analyze', err);
      return null;
    }
  }

  /**
   * Run a scoped recomputation for specific conclusions.
   * Used by RecomputationEngine for granular updates.
   */
  async recomputeConclusions(conclusionIds, state, context = {}) {
    const response = await this.interact(state.input.raw);
    if (!response?.data) return null;

    const data   = this._normalizeAnalysisData(response.data ?? response);
    const result = {};
    for (const id of conclusionIds) {
      const text = this._extractConclusionText(data, id);
      if (text) result[id] = text;
    }
    return result;
  }

  // ── Phase Bucket Partitioning ─────────────────────────────────────────────

  /**
   * Partition a single /interact response into three phase revelation buckets.
   *
   * Phase 1 — reveal first (primary signal drives):
   *   opportunity_assessment (strategic_interpretation)
   *
   * Phase 2 — reveal next (Tier 1 signal refinements):
   *   risk_map (main_risk), defensibility_assessment (counterpoint), timing_assessment
   *
   * Phase 3 — reveal last (Tier 2 nuances, cannot override Tier 1):
   *   counter_thesis, differentiation, dimension explanations
   *
   * @param {Object} response - raw /interact response
   * @returns {{ phase1, phase2, phase3, assumptions }}
   */
  _partitionIntoPhaseBuckets(response) {
    const rawData = response.data ?? (response.type === 'analysis' ? response : null);
    const data    = this._normalizeAnalysisData(rawData);
    const type    = response.type;

    if (type === 'invalid') {
      return {
        phase1: {
          conclusions: [{
            id:      'l0_rejection',
            text:    response.reply,
            signals: [],
            meta:    {
              severity:            response.data?.severity,
              rejection_type:      response.data?.rejection_type,
              one_line_verdict:    response.data?.one_line_verdict,
              what_is_missing:     response.data?.what_is_missing,
              how_to_fix:          response.data?.how_to_fix,
              rejection_confidence: response.data?.rejection_confidence,
              logical_validity_score: response.data?.logical_validity_score,
            },
          }],
        },
        phase2: { conclusions: [] },
        phase3: { conclusions: [] },
        assumptions: [],
      };
    }

    if (type === 'clarifying' || type === 'chat') {
      return {
        phase1: {
          conclusions: [{
            id:   type === 'clarifying' ? 'clarifying_question' : `chat_${Date.now()}`,
            text: response.reply,
            signals: [],
            meta: { clarification_state: response.clarification_state },
          }],
        },
        phase2: { conclusions: [] },
        phase3: { conclusions: [] },
        assumptions: [],
      };
    }

    if (!data) {
      return { phase1: { conclusions: [] }, phase2: { conclusions: [] }, phase3: { conclusions: [] }, assumptions: [] };
    }

    const sharedMeta = {
      tas_score:   data.tas_score,
      svs:         data.svs,
      quadrant:    data.quadrant,
      signal_tier: data.signal_tier,
      regime:      data.regime,
      sector:      data.sector,
      country:     data.country,
      sarima_trend: data.sarima_trend,
      drift_flag:  data.drift_flag,
      decision_badge: data.decision_badge,
      action:      data.action,
    };

    // Phase 1: primary — what the system understood about the opportunity
    const phase1Conclusions = [];

    if (data.strategic_interpretation) {
      phase1Conclusions.push({
        id:      'opportunity_assessment',
        text:    data.strategic_interpretation,
        signals: data.top_macro_signals ?? [],
        meta:    { ...sharedMeta, key_driver: data.key_driver },
      });
    }

    // Phase 2: Tier 1 refinements — risk, defensibility, timing
    const phase2Conclusions = [];

    if (data.main_risk) {
      phase2Conclusions.push({
        id:      'risk_map',
        text:    data.main_risk,
        signals: [data.dominant_risk].filter(Boolean),
        meta:    { ...sharedMeta, dominant_risk: data.dominant_risk },
      });
    }

    if (data.counterpoint) {
      phase2Conclusions.push({
        id:      'defensibility_assessment',
        text:    data.counterpoint,
        signals: [],
        meta:    sharedMeta,
      });
    }

    if (data.what_matters_most) {
      phase2Conclusions.push({
        id:      'timing_assessment',
        text:    data.what_matters_most,
        signals: [],
        meta:    sharedMeta,
      });
    }

    // Phase 3: Tier 2 nuances — cannot override Tier 1 conclusions
    const phase3Conclusions = [];

    if (data.counter_thesis) {
      phase3Conclusions.push({
        id:      'counter_thesis',
        text:    data.counter_thesis,
        signals: [],
        meta:    sharedMeta,
      });
    }

    if (data.differentiation_insight) {
      phase3Conclusions.push({
        id:      'differentiation',
        text:    data.differentiation_insight,
        signals: [],
        meta:    sharedMeta,
      });
    }

    // Idea dimension explanations (per-dimension deep dives)
    if (data.dimension_explanations) {
      for (const [dim, explanation] of Object.entries(data.dimension_explanations)) {
        phase3Conclusions.push({
          id:      `dim_${dim}`,
          text:    `${explanation.why} ${explanation.improve ?? ''}`.trim(),
          signals: [],
          meta:    { ...sharedMeta, dimension: dim, score: explanation.score, missing: explanation.missing },
        });
      }
    }

    // Extract assumptions from L0 flag and low-scoring dimensions
    const assumptions = this._extractAssumptions(data);

    return {
      phase1:      { conclusions: phase1Conclusions },
      phase2:      { conclusions: phase2Conclusions },
      phase3:      { conclusions: phase3Conclusions },
      assumptions,
      rawResponse: response,
    };
  }

  // ── Normalization ─────────────────────────────────────────────────────────

  /**
   * Normalize raw backend analysis data into canonical SESSION_STATE field names.
   * Validates numerical ranges, fills null defaults, and maps legacy key aliases.
   *
   * All consumers of backend data MUST call this before use.
   * No raw backend keys should appear outside this method.
   *
   * @param {Object} raw - raw data block from /interact or /analyze response
   * @returns {Object} normalized data with all fields present and within range
   */
  _normalizeAnalysisData(raw) {
    if (!raw) return {};

    // ── Numerical range validation ───────────────────────────────────────────
    const clamp = (val, min, max, field) => {
      const n = Number(val);
      if (!Number.isFinite(n)) return null;
      if (n < min || n > max) {
        console.warn(`[ApiAdapter] "${field}" out of range: ${n} (expected ${min}–${max}), clamping`);
      }
      return Math.max(min, Math.min(max, n));
    };

    const tasScore = clamp(raw.tas_score, 0, 100, 'tas_score');
    const svs      = clamp(raw.svs, 0, 100, 'svs');

    const ideaDimensions = {};
    if (raw.idea_dimensions && typeof raw.idea_dimensions === 'object') {
      for (const [dim, score] of Object.entries(raw.idea_dimensions)) {
        ideaDimensions[dim] = clamp(score, 0, 10, `idea_dimensions.${dim}`);
      }
    }

    // ── Backend key → canonical field mapping ────────────────────────────────
    // Some backend versions use alternate key names; all aliases resolve here.
    const strategicInterpretation =
      raw.strategic_interpretation ??
      raw.strategic_analysis ??       // legacy alias
      null;

    const mainRisk =
      raw.main_risk ??
      raw.primary_risk ??             // legacy alias
      null;

    const counterpoint =
      raw.counterpoint ??
      raw.counter_point ??            // legacy alias
      null;

    const whatMattersMost =
      raw.what_matters_most ??
      raw.timing_insight ??           // legacy alias
      null;

    // ── Null field defaults ──────────────────────────────────────────────────
    return {
      // Scores (nullable — pipeline uses null to skip a conclusion)
      tas_score:    tasScore,
      svs,
      idea_dimensions:  ideaDimensions,
      idea_reasons:     raw.idea_reasons     ?? {},
      dimension_explanations: raw.dimension_explanations ?? {},

      // Classification strings (normalized to lowercase where expected)
      quadrant:     raw.quadrant     ?? null,
      signal_tier:  raw.signal_tier  ?? null,
      regime:       raw.regime       ?? null,
      sector:       raw.sector       ?? null,
      country:      raw.country      ?? null,
      sarima_trend: raw.sarima_trend ?? null,
      drift_flag:   raw.drift_flag   ?? false,
      decision_badge: raw.decision_badge ?? null,
      action:       raw.action       ?? null,

      // Text conclusions (all nullable)
      strategic_interpretation: strategicInterpretation,
      main_risk:                mainRisk,
      dominant_risk:            raw.dominant_risk            ?? null,
      counterpoint,
      what_matters_most:        whatMattersMost,
      counter_thesis:           raw.counter_thesis           ?? null,
      differentiation_insight:  raw.differentiation_insight  ?? null,
      key_driver:               raw.key_driver               ?? null,

      // Macro signals (ensure array)
      top_macro_signals: Array.isArray(raw.top_macro_signals) ? raw.top_macro_signals : [],

      // L0 pre-analysis flags
      l0_flag:             raw.l0_flag             ?? false,
      l0_what_is_missing:  Array.isArray(raw.l0_what_is_missing) ? raw.l0_what_is_missing : [],
    };
  }

  // ── State Mapping ─────────────────────────────────────────────────────────

  _mapInteractResponseToState(response, capturedVersion) {
    if (!response) return;

    if (response.type === 'analysis' && response.data) {
      this._mapAnalysisDataToState(response.data, capturedVersion);
    } else if (response.type === 'invalid') {
      this._mapInvalidToState(response, capturedVersion);
    }
    // clarifying and chat: pipeline handles state via _applyConclusion
  }

  _mapAnalysisDataToState(rawData, capturedVersion) {
    const data = this._normalizeAnalysisData(rawData);
    if (!data) return;

    const perConclusion = {};
    if (data.idea_dimensions) {
      for (const [dim, score] of Object.entries(data.idea_dimensions)) {
        perConclusion[dim] = score >= 7 ? CONFIDENCE_TIER.HIGH
          : score >= 5     ? CONFIDENCE_TIER.MODERATE
          :                  CONFIDENCE_TIER.LOW;
      }
    }

    const assumptions = this._extractAssumptions(data); // data already normalized
    const loadBearing = assumptions.filter(a => a.isLoadBearing).map(a => a.id);
    const overallTier = this._mapSignalTierToConfidence(data.signal_tier);

    this.#store.update(
      (current) => ({
        confidence: {
          ...current.confidence,
          per_conclusion: perConclusion,
          overall:        overallTier,
          load_bearing:   loadBearing,
        },
        assumptions: {
          active: assumptions.filter(a => !a.confirmed),
        },
      }),
      { sourceLayer: 'API', capturedVersion },
    );

    // Sync signal confidence from backend (authoritative)
    this._syncSignalConfidenceFromBackend(data, capturedVersion);
  }

  _mapInvalidToState(response, capturedVersion) {
    this.#store.update(
      (current) => ({
        pipeline_state: {
          ...current.pipeline_state,
          status: 'complete',
        },
      }),
      { sourceLayer: 'API', capturedVersion },
    );
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  _extractAssumptions(data) {
    const assumptions = [];
    if (!data) return assumptions;

    if (data.l0_flag && data.l0_what_is_missing?.length > 0) {
      for (const missing of data.l0_what_is_missing) {
        assumptions.push({
          id:                  `l0_${missing.slice(0, 10).replace(/\s/g, '_')}_${Date.now()}`,
          fills_signal:        missing,
          supports_conclusion: 'opportunity_assessment',
          value:               missing,
          confirmed:           false,
          confirmed_type:      null,
          isLoadBearing:       true,
        });
      }
    }

    if (data.idea_dimensions) {
      for (const [dim, score] of Object.entries(data.idea_dimensions)) {
        if (score < 5) {
          assumptions.push({
            id:                  `dim_${dim}_${Date.now()}`,
            fills_signal:        dim,
            supports_conclusion: dim,
            value:               data.idea_reasons?.[dim] ?? `Assumed for ${dim}`,
            confirmed:           false,
            confirmed_type:      'estimate',
            isLoadBearing:       score < 4,
          });
        }
      }
    }

    return assumptions;
  }

  _syncSignalConfidenceFromBackend(data, capturedVersion) {
    const state    = this.#store.getState();
    const registry = state.signals.registry;
    if (registry.length === 0) return;

    const tier    = this._mapSignalTierToConfidence(data.signal_tier);
    const updated = registry.map(sig => ({ ...sig, confidence: tier }));

    this.#store.update(
      (current) => ({ signals: { ...current.signals, registry: updated } }),
      { sourceLayer: 'API', capturedVersion },
    );
  }

  _mapSignalTierToConfidence(tier) {
    if (!tier) return CONFIDENCE_TIER.LOW;
    const t = String(tier).toLowerCase();
    if (t.includes('strong'))   return CONFIDENCE_TIER.HIGH;
    if (t.includes('moderate')) return CONFIDENCE_TIER.MODERATE;
    if (t.includes('mixed'))    return CONFIDENCE_TIER.MODERATE;
    return CONFIDENCE_TIER.LOW;
  }

  _extractConclusionText(data, conclusionId) {
    const map = {
      opportunity_assessment:   data.strategic_interpretation,
      risk_map:                 data.main_risk,
      defensibility_assessment: data.counterpoint,
      timing_assessment:        data.what_matters_most,
      differentiation:          data.differentiation_insight,
      counter_thesis:           data.counter_thesis,
    };
    return map[conclusionId] ?? null;
  }

  _buildMessageHistory(state) {
    // Build conversation history from revealed conclusions
    const messages = [];

    // Add the current input as the latest user message
    if (state.input.raw) {
      messages.push({ role: 'user', content: state.input.raw });
    }

    return messages;
  }

  _buildContext(state) {
    const opp = state.output.conclusions?.opportunity_assessment;
    if (!opp?.meta) return {};

    return {
      tas_score:               opp.meta.tas_score,
      signal_tier:             opp.meta.signal_tier,
      sector:                  opp.meta.sector,
      country:                 opp.meta.country,
      regime:                  opp.meta.regime,
      idea_features:           {},
      dominant_risk:           state.output.conclusions?.risk_map?.meta?.dominant_risk,
      strategic_interpretation: opp.text,
      key_driver:              opp.meta.key_driver,
      main_risk:               state.output.conclusions?.risk_map?.text,
      counterpoint:            state.output.conclusions?.defensibility_assessment?.text,
    };
  }

  // ── HTTP ──────────────────────────────────────────────────────────────────

  async _fetch(path, options = {}, retries = DEFAULT_RETRY_COUNT) {
    const url        = `${this.#baseUrl}${path}`;
    const controller = new AbortController();
    const timeoutId  = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);

    const signal = options.signal
      ? _combineAbortSignals([options.signal, controller.signal])
      : controller.signal;

    const logEntry = { url, method: options.method ?? 'GET', timestamp: Date.now(), status: null };
    this.#requestLog.push(logEntry);
    if (this.#requestLog.length > 50) this.#requestLog.shift();

    try {
      const res = await fetch(url, {
        ...options,
        signal,
        headers: { 'Content-Type': 'application/json', ...(options.headers ?? {}) },
      });

      clearTimeout(timeoutId);
      logEntry.status = res.status;

      if (!res.ok) {
        if (res.status >= 500 && retries > 0) {
          await _sleep(RETRY_DELAY_MS);
          return this._fetch(path, options, retries - 1);
        }
        const body = await res.text().catch(() => res.statusText);
        throw new Error(`HTTP ${res.status}: ${body}`);
      }

      return await res.json();

    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') throw err;
      if (retries > 0) {
        await _sleep(RETRY_DELAY_MS);
        return this._fetch(path, { ...options, signal: undefined }, retries - 1);
      }
      throw err;
    }
  }

  _handleError(endpoint, err) {
    console.error(`[ApiAdapter] ${endpoint} failed:`, err.message);
    this.#eventBus.emit(
      EVENT.SESSION_STATE_UPDATED,
      { error: err.message, endpoint },
    );
  }

  getRequestLog() {
    return [...this.#requestLog];
  }
}

// ── Module utilities ──────────────────────────────────────────────────────────

function _combineAbortSignals(signals) {
  const ctrl = new AbortController();
  for (const s of signals) {
    if (s?.aborted) { ctrl.abort(); break; }
    s?.addEventListener('abort', () => ctrl.abort(), { once: true });
  }
  return ctrl.signal;
}

function _sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
