/**
 * MIDAN Engine — OutputLayerController
 *
 * THIS IS NOT A RENDERER.
 *
 * It is a stability gating system that sits between SESSION_STATE and any consumer.
 * It knows what exists in state, evaluates whether conditions are right to reveal it,
 * and only then allows a layer to become visible — step by step, in dependency order.
 *
 * Four layers, revealed in strict sequence:
 *   Interpretation → Summary → Signal Expansion → Action
 *
 * Status lifecycle:
 *   HIDDEN → PENDING → VISIBLE ↔ UPDATING → STALE → PENDING → VISIBLE
 *
 * HIDDEN    — conditions not met, not approaching. THE LAYER DOES NOT RENDER.
 * PENDING   — conditions partially met, approaching reveal. THE LAYER DOES NOT RENDER.
 *             PENDING is NOT a soft-visible state. Nothing shows until VISIBLE.
 * VISIBLE   — all conditions satisfied, content is shown (renders)
 * UPDATING  — visible but content is being revised (stays shown — not reset)
 * STALE     — upstream fully broke; must re-qualify through PENDING before VISIBLE
 *
 * STABILITY (decoupled per layer type):
 *   Interpretation layer  → interpretation stability ONLY (interpStable)
 *   Summary/SignalExp     → signal stability ONLY (signalStable)
 *   Action                → signal stability + confidence stability (confStable)
 *   Timers are per-layer — exactly one pending re-evaluation per layer.
 *
 * HYSTERESIS:
 *   STALE → PENDING → VISIBLE (no shortcut STALE → VISIBLE).
 *   UPDATING → VISIBLE requires conditions to hold for HYSTERESIS_MS (200ms)
 *   before transitioning — prevents oscillation from rapid confidence tier changes.
 *
 * CONFIDENCE DELTA:
 *   When confidence tier changes, Action immediately goes UPDATING.
 *   Action can only become VISIBLE when confidence has been stable ≥ CONFIDENCE_STABILITY_MS.
 *
 * HARD RESET:
 *   If interpretation components shift so dramatically that similarity to the
 *   previous snapshot drops below MAJOR_SHIFT_THRESHOLD (0.25), all layers are
 *   forced HIDDEN and the system re-qualifies from scratch. This is the only
 *   scenario where VISIBLE layers disappear without a pipeline-driven degradation.
 *
 * TIMER DRIFT GUARD:
 *   If a PENDING or UPDATING layer's timer hasn't fired within MAX_WAIT_MS (1200ms),
 *   the next scheduled check fires in 16ms regardless of computed delay — prevents
 *   indefinite stall under rapid high-frequency writes.
 */

import { EVENT, PIPELINE_STATUS, CONFIDENCE_TIER } from '../core/constants.js';

// ── Layer identifiers (ordered) ─────────────────────────────────────────────

export const LAYER_ID = Object.freeze({
  INTERPRETATION:   'interpretation',
  SUMMARY:          'summary',
  SIGNAL_EXPANSION: 'signal_expansion',
  ACTION:           'action',
});

export const LAYER_STATUS = Object.freeze({
  HIDDEN:   'hidden',
  PENDING:  'pending',
  VISIBLE:  'visible',
  UPDATING: 'updating',
  STALE:    'stale',
});

// Ordered reveal sequence — evaluation always walks this order
const LAYER_ORDER = [
  LAYER_ID.INTERPRETATION,
  LAYER_ID.SUMMARY,
  LAYER_ID.SIGNAL_EXPANSION,
  LAYER_ID.ACTION,
];

// SESSION_STATE output.conclusions keys that must be present (not retracted) for each layer
const REQUIRED_CONCLUSIONS = Object.freeze({
  [LAYER_ID.INTERPRETATION]:   [],
  [LAYER_ID.SUMMARY]:          ['opportunity_assessment', 'demand_signal'],
  [LAYER_ID.SIGNAL_EXPANSION]: ['feasibility_assessment', 'defensibility_assessment', 'execution_risk'],
  [LAYER_ID.ACTION]:           ['risk_map', 'timing_assessment', 'summary'],
});

// Layers that must be VISIBLE or UPDATING before this layer can reveal
const UPSTREAM_DEPS = Object.freeze({
  [LAYER_ID.INTERPRETATION]:   [],
  [LAYER_ID.SUMMARY]:          [LAYER_ID.INTERPRETATION],
  [LAYER_ID.SIGNAL_EXPANSION]: [LAYER_ID.SUMMARY],
  [LAYER_ID.ACTION]:           [LAYER_ID.SIGNAL_EXPANSION],
});

// Layers that go UPDATING when this layer degrades (downstream propagation)
const DOWNSTREAM_DEPS = Object.freeze({
  [LAYER_ID.INTERPRETATION]:   [LAYER_ID.SUMMARY, LAYER_ID.SIGNAL_EXPANSION, LAYER_ID.ACTION],
  [LAYER_ID.SUMMARY]:          [LAYER_ID.SIGNAL_EXPANSION, LAYER_ID.ACTION],
  [LAYER_ID.SIGNAL_EXPANSION]: [LAYER_ID.ACTION],
  [LAYER_ID.ACTION]:           [],
});

// Layers whose content is produced by the pipeline (go UPDATING on pipeline start/interrupt)
const PIPELINE_DEP_LAYERS = [LAYER_ID.SUMMARY, LAYER_ID.SIGNAL_EXPANSION, LAYER_ID.ACTION];

// Interpretation component keys — tracked for stability fingerprint and similarity
const INTERP_KEYS = ['problem', 'market', 'mechanism', 'context', 'traction'];

// ── Timing constants ──────────────────────────────────────────────────────────

// Interpretation/signals must have been stable this long before their layer reveals
const STABILITY_WINDOW_MS = 400;

// Confidence tier must have been stable this long before Action reveals
const CONFIDENCE_STABILITY_MS = 600;

// UPDATING → VISIBLE requires conditions to hold for this long (oscillation guard)
const HYSTERESIS_MS = 200;

// Once VISIBLE, a layer stays visible at least this long before going STALE
const MIN_HOLD_MS = Object.freeze({
  [LAYER_ID.INTERPRETATION]:   0,
  [LAYER_ID.SUMMARY]:          800,
  [LAYER_ID.SIGNAL_EXPANSION]: 1200,
  [LAYER_ID.ACTION]:           2000,
});

// If a PENDING/UPDATING layer hasn't resolved within this window, force a 16ms re-check
// to prevent indefinite stall under rapid writes (timer drift guard)
const MAX_WAIT_MS = 1200;

// ── Hard reset constants ──────────────────────────────────────────────────────

// Similarity below this triggers a hard reset (all layers HIDDEN)
const MAJOR_SHIFT_THRESHOLD = 0.25;

// Minimum populated interpretation components before a snapshot is considered stable enough
// to compare (avoids resetting on early sparse state)
const MIN_SNAPSHOT_COMPONENTS = 2;

// ── Controller ───────────────────────────────────────────────────────────────

export class OutputLayerController {
  #store;
  #eventBus;
  #layers;                         // Map<layerId, LayerRecord>
  #subscribers;                    // Set<Function>
  #unsubscribers;                  // Array<Function>

  // Stability clocks — updated when relevant state changes (never coalesced)
  #lastSignalChangeAt;             // ms timestamp of last signals.registry change
  #lastInterpretationChangeAt;     // ms timestamp of last interpretation component change
  #lastConfidenceChangeAt;         // ms timestamp of last confidence.overall tier change

  // Per-layer stability re-evaluation timers
  // At most one pending timer per layer — cancelled before rescheduling
  #stabilityTimers;                // Map<layerId, timeoutId>

  // Hysteresis tracking: when an UPDATING layer first satisfied all conditions
  // Cleared on successful → VISIBLE transition; reset when conditions break
  #layerFirstReadyAt;              // Map<layerId, number | null>

  // Fingerprints for change detection
  #prevRegistryKey;                // last-seen signal type fingerprint
  #prevInterpKey;                  // last-seen interpretation fingerprint
  #prevConfidenceTier;             // last-seen confidence.overall tier

  // Hard reset: snapshot of interpretation values when ≥ MIN_SNAPSHOT_COMPONENTS present
  #prevInterpSnapshot;             // Object<key, string|null> | null

  constructor(store, eventBus) {
    this.#store        = store;
    this.#eventBus     = eventBus;
    this.#subscribers  = new Set();
    this.#unsubscribers = [];

    this.#lastSignalChangeAt        = 0;
    this.#lastInterpretationChangeAt = 0;
    this.#lastConfidenceChangeAt    = 0;

    this.#stabilityTimers    = new Map();
    this.#layerFirstReadyAt  = new Map();
    this.#prevRegistryKey    = '';
    this.#prevInterpKey      = '';
    this.#prevConfidenceTier = null; // null = first observation (no delta yet)
    this.#prevInterpSnapshot = null;

    // All layers start HIDDEN
    this.#layers = new Map(
      LAYER_ORDER.map(id => [id, _initialLayerRecord()]),
    );

    this._bind();
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Returns a frozen snapshot of the current output layer state.
   * Safe to call from any context.
   *
   * Shape:
   *   {
   *     layers: {
   *       [LAYER_ID]: {
   *         status:     LAYER_STATUS,
   *         data:       Object | null,   // snapshot of relevant SESSION_STATE slice (null when HIDDEN)
   *         revealedAt: number | null,   // ms timestamp of most recent VISIBLE entry
   *         updatedAt:  number,          // ms timestamp of last status or data change
   *       }
   *     }
   *   }
   *
   * NOTE: Only VISIBLE and UPDATING layers have meaningful data. HIDDEN and PENDING
   * layers always have data: null. Consumers MUST NOT render based on PENDING status.
   */
  getOutputState() {
    const layers = {};
    for (const [id, layer] of this.#layers) {
      layers[id] = { ...layer };
    }
    return Object.freeze({ layers: Object.freeze(layers) });
  }

  /**
   * Subscribe to output state changes.
   * Handler called whenever any layer's status or data changes.
   * Returns unsubscribe function.
   *
   * @param {Function} fn - (outputState) => void
   */
  subscribe(fn) {
    this.#subscribers.add(fn);
    return () => this.#subscribers.delete(fn);
  }

  destroy() {
    this.#stabilityTimers.forEach(t => clearTimeout(t));
    this.#stabilityTimers.clear();
    this.#unsubscribers.forEach(fn => fn());
    this.#unsubscribers = [];
    this.#subscribers.clear();
  }

  // ── Event binding ─────────────────────────────────────────────────────────

  _bind() {
    // Primary driver — re-evaluate on every state change
    this.#unsubscribers.push(
      this.#store.subscribe((state) => this._onStateChange(state)),
    );

    // Pipeline started or interrupted → pipeline-dep visible layers go UPDATING
    this.#unsubscribers.push(
      this.#eventBus.on(EVENT.PIPELINE_STARTED, () => this._onPipelineEvent()),
    );
    this.#unsubscribers.push(
      this.#eventBus.on(EVENT.PIPELINE_INTERRUPTED, () => this._onPipelineEvent()),
    );

    // Conclusion retracted → owning layer + downstream go UPDATING
    this.#unsubscribers.push(
      this.#eventBus.on(EVENT.CONCLUSION_RETRACTED, ({ payload }) => {
        if (payload?.conclusionId) this._onConclusionRetracted(payload.conclusionId);
      }),
    );
  }

  // ── Core evaluation loop ──────────────────────────────────────────────────

  _onStateChange(state) {
    const now = Date.now();

    // ── Update signal stability clock ──────────────────────────────────────
    const registryKey = state.signals.registry.map(s => s.type).sort().join('\0');
    if (registryKey !== this.#prevRegistryKey) {
      this.#prevRegistryKey    = registryKey;
      this.#lastSignalChangeAt = now;
    }

    // ── Update interpretation stability clock ──────────────────────────────
    const interpKey = INTERP_KEYS
      .map(k => `${state.interpretation[k]?.value ?? ''}:${state.interpretation[k]?.version ?? 0}`)
      .join('\0');
    if (interpKey !== this.#prevInterpKey) {
      this.#prevInterpKey               = interpKey;
      this.#lastInterpretationChangeAt  = now;
    }

    // ── Confidence delta detection ─────────────────────────────────────────
    const newConfTier = state.confidence?.overall ?? CONFIDENCE_TIER.LOW;
    if (this.#prevConfidenceTier !== null && newConfTier !== this.#prevConfidenceTier) {
      // Tier changed — reset confidence stability clock
      this.#lastConfidenceChangeAt = now;
      // Mark Action UPDATING immediately so the user sees "revising" rather than stale data
      const actionLayer = this.#layers.get(LAYER_ID.ACTION);
      if (actionLayer.status === LAYER_STATUS.VISIBLE) {
        this.#layers.set(LAYER_ID.ACTION, {
          ...actionLayer, status: LAYER_STATUS.UPDATING, updatedAt: now,
        });
      }
    }
    this.#prevConfidenceTier = newConfTier;

    // ── Hard reset: detect major interpretation shift (Fix 5) ──────────────
    const currentInterp    = state.interpretation;
    const populatedCount   = INTERP_KEYS.filter(k => currentInterp[k]?.value != null).length;

    if (this.#prevInterpSnapshot !== null && populatedCount >= MIN_SNAPSHOT_COMPONENTS) {
      const similarity = _componentPresenceSimilarity(
        this.#prevInterpSnapshot, currentInterp,
      );
      if (similarity < MAJOR_SHIFT_THRESHOLD) {
        // Idea shifted entirely — collapse everything and re-qualify from scratch
        this._forceAllHidden(now);
        this.#prevInterpSnapshot = null; // will re-snapshot below after re-init
      }
    }

    // Update snapshot whenever we have enough components to compare against later
    if (populatedCount >= MIN_SNAPSHOT_COMPONENTS) {
      this.#prevInterpSnapshot = Object.fromEntries(
        INTERP_KEYS.map(k => [k, currentInterp[k]?.value ?? null]),
      );
    }

    // ── Compute decoupled stability booleans ───────────────────────────────
    const interpStable = (now - this.#lastInterpretationChangeAt) >= STABILITY_WINDOW_MS;
    const signalStable = (now - this.#lastSignalChangeAt)         >= STABILITY_WINDOW_MS;
    const confStable   = (now - this.#lastConfidenceChangeAt)     >= CONFIDENCE_STABILITY_MS;

    // ── Evaluate each layer in dependency order ────────────────────────────
    let anyChanged        = false;
    const newlyVisible    = []; // layers that transitioned → VISIBLE this pass

    for (const layerId of LAYER_ORDER) {
      const prevStatus = this.#layers.get(layerId).status;
      if (this._evaluateLayer(layerId, state, now, interpStable, signalStable, confStable)) {
        anyChanged = true;
        const currStatus = this.#layers.get(layerId).status;
        if (prevStatus !== LAYER_STATUS.VISIBLE && currStatus === LAYER_STATUS.VISIBLE) {
          newlyVisible.push(layerId);
        }
      }
    }

    if (anyChanged) this._notifySubscribers();

    // Schedule per-layer re-evaluation timers (with drift guard)
    this._scheduleStabilityCheck(now);

    // After scheduling, cascade newly-visible parents to downstream layers (Fix 3).
    // This fires AFTER _scheduleStabilityCheck so it overrides computed delays with 16ms.
    for (const layerId of newlyVisible) {
      this._cascadeDownstreamTimers(layerId);
    }
  }

  /**
   * Evaluate a single layer against current state.
   * Returns true if status or data changed.
   */
  _evaluateLayer(layerId, state, now, interpStable, signalStable, confStable) {
    const layer      = this.#layers.get(layerId);
    const prevStatus = layer.status;
    const elig       = this._eligibility(layerId, state, layer, interpStable, signalStable, confStable);
    const data       = this._extractData(layerId, state);

    let nextStatus = prevStatus;

    switch (prevStatus) {
      case LAYER_STATUS.HIDDEN:
        // HIDDEN → PENDING:  conditions approaching (upstream near + some own conditions)
        // HIDDEN → VISIBLE:  fully satisfied (fast path — no pending stage needed)
        if (elig.ready)     nextStatus = LAYER_STATUS.VISIBLE;
        else if (elig.near) nextStatus = LAYER_STATUS.PENDING;
        break;

      case LAYER_STATUS.PENDING:
        // PENDING does NOT render. It is an internal eligibility hold.
        // Transition to VISIBLE only when all conditions are fully met.
        if (elig.ready)      nextStatus = LAYER_STATUS.VISIBLE;
        else if (!elig.near) nextStatus = LAYER_STATUS.HIDDEN;
        break;

      case LAYER_STATUS.VISIBLE:
        // UPDATING is set externally by _onPipelineEvent() and _onConclusionRetracted().
        // This path only handles: conditions fully broke → STALE (after hold expires).
        if (elig.degraded && _holdExpired(layer, now, layerId)) {
          nextStatus = LAYER_STATUS.STALE;
        }
        break;

      case LAYER_STATUS.UPDATING: {
        // UPDATING → VISIBLE: all conditions satisfied AND hysteresis window passed.
        // Hysteresis prevents instant re-reveal after rapid confidence tier oscillation.
        if (elig.ready) {
          const firstReady = this.#layerFirstReadyAt.get(layerId);
          if (firstReady == null) {
            // First evaluation where all conditions are met — start tracking
            this.#layerFirstReadyAt.set(layerId, now);
            // Stay UPDATING this pass; timer will fire after HYSTERESIS_MS
          } else if ((now - firstReady) >= HYSTERESIS_MS) {
            // Hysteresis window passed — conditions held long enough
            nextStatus = LAYER_STATUS.VISIBLE;
            this.#layerFirstReadyAt.delete(layerId);
          }
          // else: still within hysteresis window — stay UPDATING
        } else {
          // Conditions broke — reset hysteresis, possibly degrade
          this.#layerFirstReadyAt.delete(layerId);
          if (elig.degraded && _holdExpired(layer, now, layerId)) {
            nextStatus = LAYER_STATUS.STALE;
          }
        }
        break;
      }

      case LAYER_STATUS.STALE:
        // Hysteresis: STALE cannot jump directly to VISIBLE.
        // Must re-qualify through PENDING → VISIBLE so the layer earns trust again.
        if (elig.near) nextStatus = LAYER_STATUS.PENDING;
        else           nextStatus = LAYER_STATUS.HIDDEN;
        break;
    }

    const statusChanged = nextStatus !== prevStatus;
    // Skip data comparison for hidden layers — data is irrelevant and null
    const dataChanged   = prevStatus !== LAYER_STATUS.HIDDEN && !_dataEqual(data, layer.data);

    if (!statusChanged && !dataChanged) return false;

    // revealedAt: set when first entering VISIBLE, cleared on HIDDEN/STALE
    let revealedAt = layer.revealedAt;
    if (nextStatus === LAYER_STATUS.VISIBLE && prevStatus !== LAYER_STATUS.VISIBLE) {
      revealedAt = now;
    } else if (nextStatus === LAYER_STATUS.HIDDEN || nextStatus === LAYER_STATUS.STALE) {
      revealedAt = null;
    }

    this.#layers.set(layerId, { status: nextStatus, data, revealedAt, updatedAt: now });

    if (statusChanged) {
      if (nextStatus === LAYER_STATUS.VISIBLE) {
        // Announce reveal — isFinal: true triggers COMPLETED transition in state machine
        this.#eventBus.emit(
          EVENT.OUTPUT_LAYER_REVEALED,
          { layerId, isFinal: layerId === LAYER_ID.ACTION },
          state.version,
        );
      }
      if (prevStatus === LAYER_STATUS.VISIBLE &&
          (nextStatus === LAYER_STATUS.STALE || nextStatus === LAYER_STATUS.UPDATING)) {
        // Propagate degradation to all downstream visible layers
        this._propagateDownstream(layerId, now);
      }
    }

    return true;
  }

  /**
   * Compute eligibility for a single layer.
   *
   * Returns:
   *   ready    — all conditions satisfied; layer should reveal
   *   near     — partially satisfied; layer should show PENDING
   *   degraded — layer is active but conditions broke (goes STALE after hold)
   *
   * Upstream-gone degraded fires even during a pipeline run — the pipeline cannot
   * fix a layer that no longer has a valid parent.
   * Own-conditions degraded is suppressed while the pipeline is running — it may fix it.
   */
  _eligibility(layerId, state, layer, interpStable, signalStable, confStable) {
    const pipelineRun = state.pipeline_state.status === PIPELINE_STATUS.RUNNING;
    const isActive    = layer.status !== LAYER_STATUS.HIDDEN;

    // ── Upstream dependency check ─────────────────────────────────────────────
    const deps       = UPSTREAM_DEPS[layerId];
    const depsActive = deps.every(d => {
      const s = this.#layers.get(d)?.status;
      return s === LAYER_STATUS.VISIBLE || s === LAYER_STATUS.UPDATING;
    });
    const depsNear   = deps.length === 0 || deps.some(d => {
      const s = this.#layers.get(d)?.status;
      return s === LAYER_STATUS.VISIBLE || s === LAYER_STATUS.UPDATING || s === LAYER_STATUS.PENDING;
    });

    // Upstream gone → degrade unconditionally (pipeline cannot restore a missing parent)
    if (!depsActive) {
      return { ready: false, near: depsNear, degraded: isActive };
    }

    // ── Required conclusions ──────────────────────────────────────────────────
    const required   = REQUIRED_CONCLUSIONS[layerId];
    const allPresent = required.every(id => {
      const c = state.output.conclusions[id];
      return c && !c.retracted;
    });
    const somePresent = required.length === 0 || required.some(id => {
      const c = state.output.conclusions[id];
      return c && !c.retracted;
    });

    // ── Layer-specific conditions (decoupled stability per layer type) ─────────
    let conditionsMet  = false;
    let conditionsNear = false;

    switch (layerId) {
      case LAYER_ID.INTERPRETATION: {
        const hasAnyValue  = INTERP_KEYS.some(k => state.interpretation[k]?.value != null);
        const hasAnySignal = state.signals.registry.length > 0;
        conditionsNear = hasAnyValue || hasAnySignal;
        // Interpretation uses interpretation stability only — signal rhythm is irrelevant here
        conditionsMet  = hasAnyValue && interpStable;
        break;
      }
      case LAYER_ID.SUMMARY:
      case LAYER_ID.SIGNAL_EXPANSION: {
        conditionsMet  = allPresent && signalStable; // signal stability only
        conditionsNear = somePresent;
        break;
      }
      case LAYER_ID.ACTION: {
        // LOW confidence = reasoning too uncertain to recommend action
        const confOk = state.confidence?.overall !== CONFIDENCE_TIER.LOW;
        // Action requires: conclusions + signal stability + confidence OK + confidence stable
        conditionsMet  = allPresent && signalStable && confOk && confStable;
        conditionsNear = allPresent && (!signalStable || !confOk || !confStable);
        break;
      }
    }

    // Never "ready" during pipeline run — conclusions may be mid-write
    const ready    = conditionsMet && !pipelineRun;
    const near     = conditionsNear && !ready;
    // Degraded: active layer with broken conditions, but only when pipeline won't fix it
    const degraded = isActive && !conditionsMet && !pipelineRun;

    return { ready, near, degraded };
  }

  // ── Data extraction ───────────────────────────────────────────────────────

  /**
   * Snapshot the SESSION_STATE slice relevant to this layer.
   * Consumers read ONLY this — never raw SESSION_STATE directly.
   * Returns null for HIDDEN layers (no data snapshot needed or exposed).
   */
  _extractData(layerId, state) {
    switch (layerId) {
      case LAYER_ID.INTERPRETATION:
        return {
          components:   { ...state.interpretation },
          completeness: state.signals.completeness,
          primary_id:   state.signals.primary_id,
          signals:      state.signals.registry.slice(),
        };
      case LAYER_ID.SUMMARY:
        return {
          opportunity_assessment: state.output.conclusions['opportunity_assessment'] ?? null,
          demand_signal:          state.output.conclusions['demand_signal']          ?? null,
        };
      case LAYER_ID.SIGNAL_EXPANSION:
        return {
          feasibility_assessment:   state.output.conclusions['feasibility_assessment']   ?? null,
          defensibility_assessment: state.output.conclusions['defensibility_assessment'] ?? null,
          execution_risk:           state.output.conclusions['execution_risk']           ?? null,
          confidence:               { ...state.confidence },
          signals:                  state.signals.registry.slice(),
        };
      case LAYER_ID.ACTION:
        return {
          risk_map:          state.output.conclusions['risk_map']          ?? null,
          timing_assessment: state.output.conclusions['timing_assessment'] ?? null,
          summary:           state.output.conclusions['summary']           ?? null,
          assumptions:       state.assumptions.active.slice(),
          confidence:        { ...state.confidence },
        };
      default:
        return null;
    }
  }

  // ── Pipeline event handlers ───────────────────────────────────────────────

  // Pipeline started or interrupted → pipeline-dep visible layers go UPDATING.
  // Keeps content visible while revision is in progress.
  _onPipelineEvent() {
    const now = Date.now();
    let changed = false;
    for (const layerId of PIPELINE_DEP_LAYERS) {
      const layer = this.#layers.get(layerId);
      if (layer.status === LAYER_STATUS.VISIBLE) {
        this.#layers.set(layerId, { ...layer, status: LAYER_STATUS.UPDATING, updatedAt: now });
        changed = true;
      }
    }
    if (changed) this._notifySubscribers();
  }

  // Conclusion retracted → owning layer + all downstream go UPDATING
  _onConclusionRetracted(conclusionId) {
    const now          = Date.now();
    const ownerLayerId = LAYER_ORDER.find(id => REQUIRED_CONCLUSIONS[id].includes(conclusionId));
    if (!ownerLayerId) return;

    const owner = this.#layers.get(ownerLayerId);
    if (owner.status === LAYER_STATUS.HIDDEN || owner.status === LAYER_STATUS.PENDING) return;

    let changed = false;
    if (owner.status === LAYER_STATUS.VISIBLE) {
      this.#layers.set(ownerLayerId, { ...owner, status: LAYER_STATUS.UPDATING, updatedAt: now });
      changed = true;
    }
    this._propagateDownstream(ownerLayerId, now);
    changed = true;

    if (changed) this._notifySubscribers();
  }

  // Mark all downstream layers of `fromLayerId` as UPDATING (if currently VISIBLE)
  _propagateDownstream(fromLayerId, now) {
    for (const downstreamId of DOWNSTREAM_DEPS[fromLayerId] ?? []) {
      const downstream = this.#layers.get(downstreamId);
      if (downstream.status === LAYER_STATUS.VISIBLE) {
        this.#layers.set(downstreamId, {
          ...downstream, status: LAYER_STATUS.UPDATING, updatedAt: now,
        });
      }
    }
  }

  // ── Per-layer stability re-check scheduling ───────────────────────────────

  /**
   * For each PENDING or UPDATING layer, schedule exactly one re-evaluation timer —
   * firing when its specific stability window would expire.
   * Cancels any existing timer for that layer before scheduling a new one.
   * Clears timers for layers that no longer need re-evaluation.
   *
   * Timer drift guard (MAX_WAIT_MS): if the layer has been stuck in PENDING/UPDATING
   * for more than MAX_WAIT_MS, override computed delay with 16ms to force a re-check.
   * This prevents indefinite stall when rapid high-frequency writes keep pushing
   * stability windows forward without ever reaching the re-evaluation point.
   */
  _scheduleStabilityCheck(now) {
    const signalAge = now - this.#lastSignalChangeAt;
    const interpAge = now - this.#lastInterpretationChangeAt;
    const confAge   = now - this.#lastConfidenceChangeAt;

    for (const layerId of LAYER_ORDER) {
      const layer = this.#layers.get(layerId);
      const needsCheck =
        layer.status === LAYER_STATUS.PENDING ||
        layer.status === LAYER_STATUS.UPDATING;

      if (!needsCheck) {
        // Layer resolved — cancel any pending timer
        const t = this.#stabilityTimers.get(layerId);
        if (t != null) { clearTimeout(t); this.#stabilityTimers.delete(layerId); }
        continue;
      }

      // Compute when this layer's stability window expires (per layer type)
      let delayMs;
      switch (layerId) {
        case LAYER_ID.INTERPRETATION:
          delayMs = Math.max(16, STABILITY_WINDOW_MS - interpAge + 16);
          break;
        case LAYER_ID.SUMMARY:
        case LAYER_ID.SIGNAL_EXPANSION:
          delayMs = Math.max(16, STABILITY_WINDOW_MS - signalAge + 16);
          break;
        case LAYER_ID.ACTION: {
          const sigWait  = Math.max(0, STABILITY_WINDOW_MS     - signalAge);
          const confWait = Math.max(0, CONFIDENCE_STABILITY_MS - confAge);
          delayMs = Math.max(16, Math.max(sigWait, confWait) + 16);
          break;
        }
      }

      // Hysteresis timer: if UPDATING and in hysteresis window, also bound by HYSTERESIS_MS
      if (layer.status === LAYER_STATUS.UPDATING) {
        const firstReady = this.#layerFirstReadyAt.get(layerId);
        if (firstReady != null) {
          const hysteresisWait = Math.max(16, HYSTERESIS_MS - (now - firstReady) + 16);
          delayMs = Math.min(delayMs, hysteresisWait);
        }
      }

      // Timer drift guard: if layer has been waiting too long, force immediate re-check
      const waitedMs = now - layer.updatedAt;
      if (waitedMs >= MAX_WAIT_MS) {
        delayMs = 16;
      }

      // Cancel previous, schedule fresh — guarantees exactly one pending timer per layer
      const prev = this.#stabilityTimers.get(layerId);
      if (prev != null) clearTimeout(prev);

      this.#stabilityTimers.set(layerId, setTimeout(() => {
        this.#stabilityTimers.delete(layerId);
        this._onStateChange(this.#store.getState());
      }, delayMs));
    }
  }

  /**
   * When a layer becomes VISIBLE, its downstream layers should re-evaluate promptly —
   * their upstream dependency was just satisfied. Reschedule any existing timers to 16ms.
   * Called AFTER _scheduleStabilityCheck so we override computed delays, not the reverse.
   *
   * @param {string} fromLayerId - the layer that just became VISIBLE
   */
  _cascadeDownstreamTimers(fromLayerId) {
    for (const downstreamId of DOWNSTREAM_DEPS[fromLayerId] ?? []) {
      const downstream = this.#layers.get(downstreamId);
      // Only cascade to layers actively waiting — HIDDEN with no timer still gets
      // evaluated in the same pass via LAYER_ORDER; no timer needed there.
      if (downstream.status === LAYER_STATUS.PENDING ||
          downstream.status === LAYER_STATUS.UPDATING) {
        const prev = this.#stabilityTimers.get(downstreamId);
        if (prev != null) clearTimeout(prev);
        this.#stabilityTimers.set(downstreamId, setTimeout(() => {
          this.#stabilityTimers.delete(downstreamId);
          this._onStateChange(this.#store.getState());
        }, 16));
      }
    }
  }

  // ── Hard reset ────────────────────────────────────────────────────────────

  /**
   * Force all layers to HIDDEN and clear all pending timers + tracking state.
   * Called when interpretation similarity drops below MAJOR_SHIFT_THRESHOLD —
   * the idea has changed so fundamentally that prior conclusions are invalid.
   *
   * All stability clocks are reset to `now` so layers must re-stabilize from scratch.
   */
  _forceAllHidden(now) {
    // Cancel all pending stability timers
    this.#stabilityTimers.forEach(t => clearTimeout(t));
    this.#stabilityTimers.clear();

    // Reset hysteresis tracking — old first-ready timestamps are no longer valid
    this.#layerFirstReadyAt.clear();

    // Reset stability clocks — new idea, new stability window
    this.#lastSignalChangeAt        = now;
    this.#lastInterpretationChangeAt = now;
    this.#lastConfidenceChangeAt    = now;

    // Collapse all non-HIDDEN layers
    let changed = false;
    for (const layerId of LAYER_ORDER) {
      const layer = this.#layers.get(layerId);
      if (layer.status !== LAYER_STATUS.HIDDEN) {
        this.#layers.set(layerId, {
          ...layer,
          status:     LAYER_STATUS.HIDDEN,
          revealedAt: null,
          updatedAt:  now,
        });
        changed = true;
      }
    }

    if (changed) this._notifySubscribers();
  }

  // ── Notification ──────────────────────────────────────────────────────────

  _notifySubscribers() {
    const out = this.getOutputState();
    this.#subscribers.forEach(fn => {
      try { fn(out); }
      catch (e) { console.error('[OutputLayerController] subscriber error', e); }
    });
  }
}

// ── Pure helpers ──────────────────────────────────────────────────────────────

function _initialLayerRecord() {
  return { status: LAYER_STATUS.HIDDEN, data: null, revealedAt: null, updatedAt: Date.now() };
}

function _holdExpired(layer, now, layerId) {
  return (now - (layer.revealedAt ?? 0)) >= MIN_HOLD_MS[layerId];
}

/**
 * Compute interpretation similarity between two snapshots.
 *
 * Snapshots store `{ [key]: string|null }` for each INTERP_KEY.
 * Similarity = (keys still present) / (keys that were present in prevSnap).
 * Returns 1.0 when prevSnap is null or too sparse to compare meaningfully.
 *
 * @param {Object} prevSnap     - Object<key, string|null>
 * @param {Object} currentInterp - SESSION_STATE.interpretation slice
 * @returns {number} [0, 1]
 */
function _componentPresenceSimilarity(prevSnap, currentInterp) {
  if (!prevSnap) return 1;
  const prevPresent = INTERP_KEYS.filter(k => prevSnap[k] != null);
  if (prevPresent.length < MIN_SNAPSHOT_COMPONENTS) return 1; // too sparse to judge
  const stillPresent = prevPresent.filter(k => currentInterp[k]?.value != null).length;
  return stillPresent / prevPresent.length;
}

/**
 * Structural equality check — suppresses no-op subscriber notifications.
 * Only called for non-hidden layers; JSON.stringify is sufficient for our
 * data shapes (no circular refs, no non-serializable values).
 */
function _dataEqual(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  try { return JSON.stringify(a) === JSON.stringify(b); }
  catch { return false; }
}
