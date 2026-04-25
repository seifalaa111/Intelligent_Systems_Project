/**
 * MIDAN Engine — PerceptionSyncManager
 *
 * THE PERCEPTION LAYER.
 *
 * This is not a renderer. It does not touch the DOM. It does not manage CSS.
 * It does not know what anything looks like.
 *
 * It does one thing: translate the raw, instantaneous stream of system events
 * into a coherent sequence of timed perception signals — deciding WHEN each
 * change should be felt, HOW LONG it takes to settle, and HOW changes that
 * belong together are visually connected.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * WHY THIS EXISTS
 *
 * Without this layer, every system event maps directly to an immediate visual
 * change. That produces a system that is:
 *   - Visually reactive (fast, mechanical, robotic)
 *   - Not perceptually meaningful (changes feel disconnected)
 *   - Unable to convey intent or weight
 *
 * With this layer, the system:
 *   - Pauses briefly before important conclusions (the weight of thought)
 *   - Connects related changes so they feel caused by each other
 *   - Adapts smoothly when interrupted (redirects instead of snaps)
 *   - Reveals things only when it "trusts" them (stability → presentation)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * RHYTHM
 *
 * Base unit: 80ms. Every delay and duration is a multiple of 80ms.
 * This is not an animation constraint. It is a perceptual one.
 * Humans subconsciously detect rhythm. When timing is consistent, the system
 * feels intentional. When it is not, it feels chaotic.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * MOTION HIERARCHY
 *
 * PRIMARY   — Structural changes (layer reveals, state transitions, retractions)
 *             Duration: 3u (240ms). Stagger between items: 1u (80ms).
 *             Preceded by PERCEPTION_ANTICIPATE when predictable.
 *
 * SECONDARY — Refinements (interpretation updates, conclusions arriving, tier shifts)
 *             Duration: 2u (160ms). Items batch simultaneously.
 *             These feel responsive but never disruptive.
 *
 * TERTIARY  — Micro-adjustments (signal detection, completeness ticks, pipeline pulses)
 *             Duration: 1u (80ms). Items batch simultaneously.
 *             Nearly invisible — they support the narrative without interrupting it.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * COALESCING
 *
 * Events that arrive within one rhythm unit (80ms) of each other are grouped
 * into a single sequence. This creates coherence: related changes that happen
 * near-simultaneously feel connected, not like independent flickers.
 *
 * Layer reveals bypass coalescing — they are time-sensitive and always PRIMARY.
 * Heartbeats bypass coalescing — they are rhythm-critical and always TERTIARY.
 * Interruptions bypass coalescing — they clear the batch and redirect immediately.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * SEQUENCING
 *
 * Within a coalesced batch:
 *   1. PRIMARY items first, then SECONDARY, then TERTIARY
 *   2. PRIMARY items stagger by 1u — causality is perceptible
 *   3. SECONDARY and TERTIARY items within the same tier are simultaneous —
 *      refinements feel like a single coherent update, not a cascade
 *   4. A gap separates tiers — so PRIMARY fully enters before SECONDARY begins
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * ANTICIPATION
 *
 * Before structural reveals, PERCEPTION_ANTICIPATE fires 2u (160ms) early.
 * The UI uses this window to prepare — not with obvious indicators, but with
 * subtle layout readiness. If the reveal does not follow, anticipation state
 * is silently dropped.
 *
 * Anticipation fires on:
 *   - System entering ANALYZING (output is coming)
 *   - System entering REVEALING (layers are about to appear)
 *   - Each non-final layer reveal (the next layer is coming)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * INTERRUPTION
 *
 * When a sequence is interrupted (PAUSED, IDLE, pipeline cancelled):
 *   1. PERCEPTION_INTERRUPT fires immediately with `progress` [0,1]
 *      — the UI knows where in the animation it is and can redirect
 *   2. The pending batch is cleared (no stale items emit)
 *   3. The next sequence starts with a 1u redirect buffer — giving the UI
 *      time to complete the redirect before new content arrives
 *
 * The motion never snaps. It redirects.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * UNCERTAINTY SCALING
 *
 * When the uncertainty overlay is active (UNCERTAIN_ACTIVATED), all sequence
 * durations are scaled by UNCERTAINTY.MOTION_DURATION_SCALE (1.4×), snapped
 * to the nearest rhythm unit. This slows the perceived pace — the system
 * signals that it is working with incomplete information.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * OUTPUT EVENTS
 *
 * PERCEPTION_SEQUENCE   — A timed sequence of items ready to begin.
 *   { id, items, totalDuration, isInterrupt, redirectsFrom }
 *   items[]: { targetId, change, weight, delay, duration, meta }
 *
 * PERCEPTION_ANTICIPATE — The system is about to make a meaningful reveal.
 *   { for, window }
 *
 * PERCEPTION_INTERRUPT  — An active sequence should redirect, not snap.
 *   { sequenceId, progress, reason }
 *
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * CONSTRAINTS
 *
 * This class does not write to SESSION_STATE.
 * This class does not call store.update().
 * This class emits only PERCEPTION_* events.
 * All timing is on the 80ms rhythm grid.
 */

import {
  EVENT,
  SYSTEM_STATE,
  UNCERTAINTY,
  PERCEPTION_WEIGHT,
} from '../core/constants.js';

// ── Rhythm ─────────────────────────────────────────────────────────────────

const RHYTHM = 80; // base unit, ms

/** Snap a duration to the nearest rhythm unit (minimum: 1u). */
function _snap(ms) {
  return Math.max(RHYTHM, Math.round(ms / RHYTHM) * RHYTHM);
}

/** Scale duration for uncertainty state and snap to grid. */
function _scaleUncertain(ms) {
  return _snap(ms * UNCERTAINTY.MOTION_DURATION_SCALE);
}

// ── Timing constants ────────────────────────────────────────────────────────

// Duration (ms) for each weight tier — how long a change takes to settle
const WEIGHT_DURATION = Object.freeze({
  [PERCEPTION_WEIGHT.PRIMARY]:   3 * RHYTHM,   // 240ms — structural, deliberate
  [PERCEPTION_WEIGHT.SECONDARY]: 2 * RHYTHM,   // 160ms — refinement, responsive
  [PERCEPTION_WEIGHT.TERTIARY]:  1 * RHYTHM,   //  80ms — micro, near-invisible
});

// Stagger between same-tier items within a batch
// PRIMARY items stagger so their causality is perceptible.
// SECONDARY / TERTIARY items batch simultaneously — they feel like one coherent update.
const WEIGHT_STAGGER = Object.freeze({
  [PERCEPTION_WEIGHT.PRIMARY]:   1 * RHYTHM,   // 80ms — causality visible
  [PERCEPTION_WEIGHT.SECONDARY]: 0,            //  0ms — simultaneous
  [PERCEPTION_WEIGHT.TERTIARY]:  0,            //  0ms — simultaneous
});

// Gap between weight tiers (after the previous tier's last item ends)
const TIER_GAP = Object.freeze({
  afterPrimary:   1 * RHYTHM,   // 80ms — secondary starts after primary has entered
  afterSecondary: 0,            //  0ms — tertiary may overlap secondary's tail
});

// Events collected within this window are grouped into a single sequence.
// This 1u delay is also the "slight pause" before the system responds —
// intentional, not latency.
const COALESCE_WINDOW = 1 * RHYTHM; // 80ms

// Soft boundary tolerance: if a new event arrives within this many ms after
// the previous batch flushed, it uses a shortened window (SPILL_MS) instead
// of a full COALESCE_WINDOW. This prevents perceptual discontinuity when
// related events straddle the 80ms boundary by a few milliseconds.
const SPILL_MS = 20;

// When a sequence is interrupted and a new one begins, first items are delayed
// by this amount — giving the UI time to redirect in-progress motion.
const REDIRECT_BUFFER = 1 * RHYTHM; // 80ms

// A sequence must be at least this long to be perceivable.
const PERCEPTION_FLOOR = 1 * RHYTHM; // 80ms

// Fire PERCEPTION_ANTICIPATE this many ms before a major reveal.
// 2u gives the UI time to prepare without being obvious.
const ANTICIPATION_WINDOW = 2 * RHYTHM; // 160ms

// If the anticipated event does not arrive within this window after PERCEPTION_ANTICIPATE
// fired, emit PERCEPTION_ANTICIPATE_CANCELLED so the UI can exit the "prepared" state.
// 10u (800ms) is generous — covers full analysis pipeline latency.
const ANTICIPATION_ABANDON_MS = 10 * RHYTHM; // 800ms

// How long each layer's reveal takes to settle — later layers carry more weight
// because they are more conclusive (action layer = full recommendation).
const LAYER_REVEAL_DURATION = Object.freeze({
  interpretation:   3 * RHYTHM,   // 240ms
  summary:          4 * RHYTHM,   // 320ms
  signal_expansion: 4 * RHYTHM,   // 320ms
  action:           5 * RHYTHM,   // 400ms
});

// Ordered layer sequence — used to predict the next layer for anticipation
const LAYER_ORDER = ['interpretation', 'summary', 'signal_expansion', 'action'];

// Sort weights for batch ordering (lower = emitted earlier)
const WEIGHT_SORT_ORDER = Object.freeze({
  [PERCEPTION_WEIGHT.PRIMARY]:   0,
  [PERCEPTION_WEIGHT.SECONDARY]: 1,
  [PERCEPTION_WEIGHT.TERTIARY]:  2,
});

// Within the PRIMARY tier, items are further sorted by semantic importance.
// State transitions lead — they define the system's mode and are the causal
// anchor everything else is seen relative to. Layer reveals follow (they
// confirm the state). Structural mutations (retractions, rejections) come last
// since they refine an already-visible conclusion.
// Returns a sort key — lower = leads the stagger.
function _primaryOrderOf(targetId) {
  if (targetId.startsWith('state:'))        return  0;  // system mode (highest)
  if (targetId.startsWith('layer:'))        return 10;  // structural reveal
  if (targetId.startsWith('assumption:'))   return 20;  // load-bearing assumption change
  if (targetId.startsWith('conclusion:'))   return 21;  // conclusion retraction
  if (targetId.startsWith('pipeline:'))     return 30;  // pipeline machinery
  return 99;                                            // unknown — trail everything
}

// ── PerceptionSyncManager ──────────────────────────────────────────────────

export class PerceptionSyncManager {
  #eventBus;
  #store;

  // Coalescing state
  #coalesceTimer;         // setTimeout handle for the 80ms collection window
  #pendingBatch;          // RawItem[] accumulating within the current window
  #lastFlushAt;           // number | null — ms timestamp of last _flushBatch() call (Fix 1)

  // Active sequence tracking (for interruption progress and overlap protection)
  #activeSequence;        // { id, startedAt, totalDuration } | null
  #activeSequenceTimer;   // setTimeout to clear #activeSequence after it completes

  // Redirect state — next sequence carries a 1u buffer for motion redirection
  #redirectPending;       // boolean
  #redirectPendingFromId; // string | null — the interrupted sequence's ID

  // Anticipation
  #anticipationTimer;       // setTimeout handle for scheduled PERCEPTION_ANTICIPATE
  #anticipationAbandonTimer; // setTimeout that fires CANCELLED if reveal never arrives (Fix 3)
  #anticipationFired;       // boolean — true if PERCEPTION_ANTICIPATE was emitted (Fix 3)
  #anticipationForTarget;   // string | null — what was anticipated (for CANCELLED payload)

  // Uncertainty
  #isUncertain;           // boolean — when true, durations scale × 1.4

  // Lifecycle
  #unsubscribers;         // Array<Function> — cleanup handles from eventBus.on()
  #sequenceCounter;       // monotonic counter for sequence IDs

  /**
   * @param {import('../core/store.js').SessionStore} store
   * @param {import('../core/event-bus.js').EventBus} eventBus
   */
  constructor(store, eventBus) {
    this.#eventBus                = eventBus;
    this.#store                   = store;
    this.#coalesceTimer           = null;
    this.#pendingBatch            = [];
    this.#lastFlushAt             = null;
    this.#activeSequence          = null;
    this.#activeSequenceTimer     = null;
    this.#redirectPending         = false;
    this.#redirectPendingFromId   = null;
    this.#anticipationTimer       = null;
    this.#anticipationAbandonTimer = null;
    this.#anticipationFired       = false;
    this.#anticipationForTarget   = null;
    this.#isUncertain             = false;
    this.#unsubscribers           = [];
    this.#sequenceCounter         = 0;

    this._bind();
  }

  destroy() {
    clearTimeout(this.#coalesceTimer);
    clearTimeout(this.#activeSequenceTimer);
    clearTimeout(this.#anticipationTimer);
    clearTimeout(this.#anticipationAbandonTimer);
    this.#unsubscribers.forEach(fn => fn());
    this.#unsubscribers           = [];
    this.#pendingBatch            = [];
    this.#activeSequence          = null;
    this.#anticipationFired       = false;
    this.#anticipationForTarget   = null;
  }

  // ── Event binding ────────────────────────────────────────────────────────

  _bind() {
    const sub = (type, fn) => {
      this.#unsubscribers.push(this.#eventBus.on(type, fn));
    };

    // ── Uncertainty tracking ──────────────────────────────────────────────
    // Must be synchronous with no batching — uncertainty affects all subsequent sequences.
    sub(EVENT.UNCERTAIN_ACTIVATED,   () => { this.#isUncertain = true; });
    sub(EVENT.UNCERTAIN_DEACTIVATED, () => { this.#isUncertain = false; });

    // ── System state transitions ──────────────────────────────────────────
    // Handled separately because PAUSED/IDLE require immediate interruption,
    // not coalesced batching.
    sub(EVENT.STATE_TRANSITION, ({ payload }) => {
      this._onStateTransition(payload ?? {});
    });

    // ── Layer reveals ─────────────────────────────────────────────────────
    // Bypass coalescing — reveals are time-sensitive, always PRIMARY, and
    // already gated by the OutputLayerController's stability window.
    sub(EVENT.OUTPUT_LAYER_REVEALED, ({ payload }) => {
      this._onLayerRevealed(payload ?? {});
    });

    // ── Pipeline events ───────────────────────────────────────────────────
    sub(EVENT.PIPELINE_STARTED, ({ payload }) => {
      this._enqueue({
        targetId: 'pipeline:active',
        change:   'enter',
        weight:   PERCEPTION_WEIGHT.SECONDARY,
        meta:     payload ?? {},
      });
    });

    sub(EVENT.PIPELINE_COMPLETED, ({ payload }) => {
      this._enqueue({
        targetId: 'pipeline:active',
        change:   'exit',
        weight:   PERCEPTION_WEIGHT.SECONDARY,
        meta:     payload ?? {},
      });
    });

    sub(EVENT.PIPELINE_INTERRUPTED, ({ payload }) => {
      // Pipeline interrupted → interrupt current sequence, then mark pipeline exit
      this._emitInterrupt('pipeline_interrupted');
      this._enqueue({
        targetId: 'pipeline:active',
        change:   'interrupt',
        weight:   PERCEPTION_WEIGHT.PRIMARY,
        meta:     payload ?? {},
      });
    });

    sub(EVENT.PIPELINE_CHECKPOINT, ({ payload }) => {
      // Checkpoint pulses are TERTIARY — they signal progress without demanding attention
      this._enqueue({
        targetId: `pipeline:checkpoint:${payload?.checkpoint?.phase_id ?? 'unknown'}`,
        change:   'pulse',
        weight:   PERCEPTION_WEIGHT.TERTIARY,
        meta:     payload ?? {},
      });
    });

    // ── Interpretation ────────────────────────────────────────────────────
    sub(EVENT.INTERPRETATION_UPDATED, ({ payload }) => {
      const components = payload?.components ?? [];
      // Each updated component is a separate SECONDARY item — they coalesce into
      // one batch and reveal simultaneously, feeling like a single coherent update.
      for (const comp of components) {
        this._enqueue({
          targetId: `interpretation:${comp}`,
          change:   'update',
          weight:   PERCEPTION_WEIGHT.SECONDARY,
          meta:     { component: comp },
        });
      }
    });

    sub(EVENT.CONCLUSION_UPDATED, ({ payload }) => {
      // Conclusions arriving during pipeline run — each is SECONDARY.
      // They collectively produce the "system is building its thought" feeling.
      this._enqueue({
        targetId: `conclusion:${payload?.conclusionId ?? 'unknown'}`,
        change:   payload?.status === 'applied' ? 'reveal' : 'update',
        weight:   PERCEPTION_WEIGHT.SECONDARY,
        meta:     payload ?? {},
      });
    });

    sub(EVENT.CONCLUSION_RETRACTED, ({ payload }) => {
      // Retractions are PRIMARY — they restructure the output, not merely refine it.
      // The user must notice that something they saw is no longer supported.
      this._enqueue({
        targetId: `conclusion:${payload?.conclusionId ?? 'unknown'}`,
        change:   'retract',
        weight:   PERCEPTION_WEIGHT.PRIMARY,
        meta:     payload ?? {},
      });
    });

    sub(EVENT.CONFIDENCE_CHANGED, ({ payload }) => {
      // Tier change = SECONDARY (the system's trust level shifted meaningfully)
      // Sub-tier fluctuation = TERTIARY (background recalibration)
      const isTierChange = payload?.previousTier !== payload?.tier;
      this._enqueue({
        targetId: 'confidence',
        change:   isTierChange ? 'tier_change' : 'update',
        weight:   isTierChange
          ? PERCEPTION_WEIGHT.SECONDARY
          : PERCEPTION_WEIGHT.TERTIARY,
        meta:     payload ?? {},
      });
    });

    sub(EVENT.ASSUMPTION_CONFIRMED, ({ payload }) => {
      // Confirmation is SECONDARY — it refines the reasoning without restructuring it
      this._enqueue({
        targetId: `assumption:${payload?.assumptionId ?? 'unknown'}`,
        change:   'confirm',
        weight:   PERCEPTION_WEIGHT.SECONDARY,
        meta:     payload ?? {},
      });
    });

    sub(EVENT.ASSUMPTION_REJECTED, ({ payload }) => {
      // Rejection is PRIMARY — it removes a load-bearing element, restructuring output
      this._enqueue({
        targetId: `assumption:${payload?.assumptionId ?? 'unknown'}`,
        change:   'reject',
        weight:   PERCEPTION_WEIGHT.PRIMARY,
        meta:     payload ?? {},
      });
    });

    // ── Signals ───────────────────────────────────────────────────────────
    sub(EVENT.SIGNAL_DETECTED, ({ payload }) => {
      // Tier 1 signals are structurally significant (problem, geography, friction).
      // Tier 2 signals are supporting context — TERTIARY.
      const tier   = payload?.signal?.tier;
      const weight = tier === 'TIER1'
        ? PERCEPTION_WEIGHT.SECONDARY
        : PERCEPTION_WEIGHT.TERTIARY;
      this._enqueue({
        targetId: `signal:${payload?.signal?.type ?? 'unknown'}`,
        change:   'detect',
        weight,
        meta:     payload ?? {},
      });
    });

    sub(EVENT.SIGNAL_REMOVED, ({ payload }) => {
      this._enqueue({
        targetId: `signal:${payload?.type ?? 'unknown'}`,
        change:   'remove',
        weight:   PERCEPTION_WEIGHT.TERTIARY,
        meta:     payload ?? {},
      });
    });

    sub(EVENT.SIGNAL_PROMOTED, ({ payload }) => {
      // Promotion changes the signal's role — SECONDARY (it matters more now)
      this._enqueue({
        targetId: `signal:${payload?.type ?? 'unknown'}`,
        change:   'promote',
        weight:   PERCEPTION_WEIGHT.SECONDARY,
        meta:     payload ?? {},
      });
    });

    sub(EVENT.SIGNAL_DEMOTED, ({ payload }) => {
      this._enqueue({
        targetId: `signal:${payload?.type ?? 'unknown'}`,
        change:   'demote',
        weight:   PERCEPTION_WEIGHT.SECONDARY,
        meta:     payload ?? {},
      });
    });

    sub(EVENT.COMPLETENESS_UPDATED, ({ payload }) => {
      // Completeness ticks are TERTIARY — the meter grows imperceptibly behind the scene
      this._enqueue({
        targetId: 'completeness',
        change:   'update',
        weight:   PERCEPTION_WEIGHT.TERTIARY,
        meta:     { completeness: payload?.completeness },
      });
    });

    // ── Heartbeat ─────────────────────────────────────────────────────────
    // Heartbeats bypass coalescing — they are rhythm-critical pulses that must
    // fire on-grid during slow analysis. Each fires independently.
    sub(EVENT.HEARTBEAT, ({ payload }) => {
      this._onHeartbeat(payload ?? {});
    });
  }

  // ── Coalescing ────────────────────────────────────────────────────────────

  /**
   * Add a raw item to the pending batch.
   * Starts the coalesce window if none is active.
   *
   * The 80ms window is simultaneously:
   *   (a) a grouping mechanism — related events batch together
   *   (b) the "slight pause" the spec describes — the system notices, then responds
   *
   * Soft boundary (Fix 1): if a new event arrives within SPILL_MS (20ms) after the
   * previous batch flushed, it uses a shortened window instead of a full COALESCE_WINDOW.
   * This prevents perceptual discontinuity when related events straddle the 80ms boundary
   * by a few milliseconds — e.g., event A at 79ms and event B at 81ms would otherwise
   * produce two separate sequences when one coherent sequence is correct.
   *
   * The spill window is still bounded: maximum additional delay is SPILL_MS (20ms),
   * which keeps total grouping latency well under 2u (160ms).
   *
   * @param {{ targetId, change, weight, meta, duration? }} item
   */
  _enqueue(item) {
    this.#pendingBatch.push(item);

    if (this.#coalesceTimer === null) {
      // Soft boundary: use a shortened window if we're within SPILL_MS of the last flush
      const msSinceFlush = this.#lastFlushAt === null
        ? Infinity
        : Date.now() - this.#lastFlushAt;

      const delay = msSinceFlush <= SPILL_MS
        ? Math.max(4, SPILL_MS - msSinceFlush) // micro-window: let more late arrivals in
        : COALESCE_WINDOW;                      // normal: full 80ms grouping pause

      this.#coalesceTimer = setTimeout(() => {
        this.#coalesceTimer = null;
        this._flushBatch();
      }, delay);
    }
  }

  /**
   * Flush the accumulated batch: build a timed sequence and emit it.
   * Called when the coalesce window expires.
   *
   * Overlap protection (Fix 2): if the previous sequence is still in motion,
   * the new sequence's base delay is shifted forward so it begins no earlier
   * than when the active sequence ends. This prevents overlapping motion chaos
   * without requiring explicit interruption — the new content simply queues
   * behind the current content's natural finish.
   *
   * Exception: if the new batch contains an explicit interrupt item (from a
   * PIPELINE_INTERRUPTED event), the overlap delay is zeroed — the interrupt
   * event itself handles the redirect story.
   */
  _flushBatch() {
    this.#lastFlushAt = Date.now(); // Fix 1: record flush time for soft boundary

    const batch = this.#pendingBatch;
    this.#pendingBatch = [];

    if (batch.length === 0) return;

    // Fix 2: compute overlap delay — how long until the active sequence finishes
    const overlapDelay = this._computeOverlapDelay(batch);

    const sequence = this._buildSequence(batch, overlapDelay);
    if (sequence) this._emitSequence(sequence);
  }

  /**
   * Compute the ms delay needed so the new sequence doesn't overlap the active one.
   * Returns 0 if there is no active sequence or if the batch signals an interrupt.
   *
   * @param {Array} batch - items about to be sequenced
   * @returns {number} delay in ms (may be 0)
   */
  _computeOverlapDelay(batch) {
    if (!this.#activeSequence) return 0;

    // Interrupt items explicitly replace the active sequence — no delay needed.
    // They carry their own redirect logic (REDIRECT_BUFFER is already in _buildSequence).
    const hasInterruptItem = batch.some(item => item.change === 'interrupt');
    if (hasInterruptItem) return 0;

    const activeEndAt = this.#activeSequence.startedAt + this.#activeSequence.totalDuration;
    return Math.max(0, activeEndAt - Date.now());
  }

  // ── State transition handling ─────────────────────────────────────────────

  _onStateTransition(payload) {
    const { to, from, trigger } = payload;

    // ── Immediate interruption paths (bypass coalescing) ──────────────────
    if (to === SYSTEM_STATE.PAUSED) {
      // User typed during active analysis — motion must redirect, not snap.
      this._emitInterrupt('user_paused');
      return; // no further item enqueued — INTERRUPT is the signal
    }

    if (to === SYSTEM_STATE.IDLE) {
      // Session reset — everything returns to neutral.
      this._emitInterrupt('idle_reset');
      return;
    }

    // ── Anticipation for upcoming structural events ────────────────────────
    // When entering ANALYZING, output is coming.
    // When entering REVEALING, layers are about to appear.
    if (to === SYSTEM_STATE.ANALYZING || to === SYSTEM_STATE.REVEALING) {
      this._scheduleAnticipation(to === SYSTEM_STATE.ANALYZING ? 'analysis' : 'reveal');
    }

    // ── Classify and enqueue ──────────────────────────────────────────────
    let weight = PERCEPTION_WEIGHT.SECONDARY;
    let change = 'enter';

    switch (to) {
      // PRIMARY — structural, the system is doing something definitive
      case SYSTEM_STATE.ANALYZING:
        weight = PERCEPTION_WEIGHT.PRIMARY;
        change = 'analyzing';
        break;
      case SYSTEM_STATE.REVEALING:
        weight = PERCEPTION_WEIGHT.PRIMARY;
        change = 'revealing';
        break;
      case SYSTEM_STATE.COMPLETED:
        weight = PERCEPTION_WEIGHT.PRIMARY;
        change = 'completed';
        break;
      case SYSTEM_STATE.REDIRECTED:
        weight = PERCEPTION_WEIGHT.PRIMARY;
        change = 'redirected';
        break;

      // SECONDARY — responsive but not disruptive
      case SYSTEM_STATE.INTERPRETING:
        weight = PERCEPTION_WEIGHT.SECONDARY;
        change = 'interpreting';
        break;
      case SYSTEM_STATE.UNCERTAIN:
        weight = PERCEPTION_WEIGHT.SECONDARY;
        change = 'uncertain';
        break;

      // TERTIARY — low signal, system is just active
      case SYSTEM_STATE.RECEIVING:
        weight = PERCEPTION_WEIGHT.TERTIARY;
        change = 'receiving';
        break;
      case SYSTEM_STATE.LISTENING:
        weight = PERCEPTION_WEIGHT.TERTIARY;
        change = 'listening';
        break;
    }

    this._enqueue({
      targetId: `state:${to.toLowerCase()}`,
      change,
      weight,
      meta: payload,
    });
  }

  // ── Layer reveal handling ─────────────────────────────────────────────────

  _onLayerRevealed(payload) {
    const { layerId, isFinal } = payload;
    if (!layerId) return;

    // Cancel pending anticipation — the reveal IS the major event.
    // Uses _cancelAnticipation so that if PERCEPTION_ANTICIPATE was already emitted,
    // the UI is NOT sent a CANCELLED signal (the reveal itself is the confirmation).
    // We clear the fired flag first so _cancelAnticipation skips the CANCELLED emit.
    clearTimeout(this.#anticipationAbandonTimer);
    this.#anticipationAbandonTimer = null;
    clearTimeout(this.#anticipationTimer);
    this.#anticipationTimer     = null;
    this.#anticipationFired     = false;  // suppress CANCELLED — reveal is the resolution
    this.#anticipationForTarget = null;

    // Layer reveals are always PRIMARY and always bypass coalescing.
    // Duration is layer-specific: later layers carry more weight.
    const duration = LAYER_REVEAL_DURATION[layerId] ?? WEIGHT_DURATION[PERCEPTION_WEIGHT.PRIMARY];

    const sequence = this._buildSequence([{
      targetId: `layer:${layerId}`,
      change:   'reveal',
      weight:   PERCEPTION_WEIGHT.PRIMARY,
      duration, // explicit override — layer-specific, not weight-default
      meta:     payload,
    }]);

    if (sequence) this._emitSequence(sequence);

    // If more layers are coming, anticipate the next one.
    // This is advisory — if the next layer does not reveal, the UI silently
    // drops the anticipation treatment.
    if (!isFinal) {
      const nextId = _nextLayerId(layerId);
      if (nextId) this._scheduleAnticipation(`reveal:${nextId}`);
    }
  }

  // ── Heartbeat handling ────────────────────────────────────────────────────

  _onHeartbeat(payload) {
    // Heartbeats are rhythm-critical — they must fire on-grid during slow analysis.
    // Each one is a direct emit, bypassing coalescing and sequence building.
    // They produce the "system is thinking" pulse the user perceives during latency.
    this.#eventBus.emit(EVENT.PERCEPTION_SEQUENCE, {
      id:            `hb-${++this.#sequenceCounter}`,
      items:         [{
        targetId: `heartbeat:${payload.type ?? 'pulse'}`,
        change:   'pulse',
        weight:   PERCEPTION_WEIGHT.TERTIARY,
        delay:    0,
        duration: RHYTHM,
        meta:     payload,
      }],
      totalDuration:  RHYTHM,
      isInterrupt:    false,
      redirectsFrom:  null,
    });
  }

  // ── Sequence construction ─────────────────────────────────────────────────

  /**
   * Build a timed sequence from a batch of classified items.
   *
   * Ordering rules:
   *   1. Sort by weight: PRIMARY → SECONDARY → TERTIARY
   *   2. Within PRIMARY: sub-sort by semantic importance (state > layer > pipeline)
   *      so causally leading items always precede dependent ones (Fix 5)
   *   3. PRIMARY items stagger by 1u (80ms) — causality is perceptible
   *   4. SECONDARY / TERTIARY batch simultaneously within tier — coherent update
   *   5. Tiers are separated by a gap (after PRIMARY: 1u, after SECONDARY: 0u)
   *   6. Durations scale × 1.4 during uncertainty, snapped to grid
   *   7. When redirecting: all items delayed by REDIRECT_BUFFER (1u = 80ms)
   *   8. extraBaseDelay: additional offset for overlap protection (Fix 2)
   *
   * @param {Array<{ targetId, change, weight, duration?, meta }>} batch
   * @param {number} [extraBaseDelay=0] - additional ms offset (overlap protection)
   * @returns {{ id, items, totalDuration, isInterrupt, redirectsFrom } | null}
   */
  _buildSequence(batch, extraBaseDelay = 0) {
    if (batch.length === 0) return null;

    // Sort by weight tier first, then by PRIMARY semantic order within that tier (Fix 5).
    // Stable sort (V8 guarantees) preserves arrival order for equal keys.
    const sorted = [...batch].sort((a, b) => {
      const wa = WEIGHT_SORT_ORDER[a.weight] ?? 2;
      const wb = WEIGHT_SORT_ORDER[b.weight] ?? 2;
      if (wa !== wb) return wa - wb;
      // Same tier: apply PRIMARY sub-sort (no-op for SECONDARY/TERTIARY — arrival order wins)
      if (wa === 0) return _primaryOrderOf(a.targetId) - _primaryOrderOf(b.targetId);
      return 0;
    });

    // Consume redirect state before building — each sequence only redirects once
    const isRedirecting = this.#redirectPending;
    const fromId        = this.#redirectPendingFromId;
    this.#redirectPending         = false;
    this.#redirectPendingFromId   = null;

    // Base delay: redirect buffer (if redirecting) + overlap protection offset (Fix 2)
    const baseDelay = (isRedirecting ? REDIRECT_BUFFER : 0) + extraBaseDelay;

    // Single-pass delay and duration assignment
    let prevTier    = null;   // weight tier of the previous item
    let prevTierEnd = baseDelay; // latest end-time seen for the previous tier
    let cursorDelay = baseDelay; // current delay cursor for within-tier placement

    const items = sorted.map(item => {
      const weight = item.weight ?? PERCEPTION_WEIGHT.TERTIARY;

      // ── Tier boundary: advance cursor past previous tier + gap ─────────
      if (prevTier !== null && prevTier !== weight) {
        const gap = prevTier === PERCEPTION_WEIGHT.PRIMARY   ? TIER_GAP.afterPrimary
                  : prevTier === PERCEPTION_WEIGHT.SECONDARY ? TIER_GAP.afterSecondary
                  : 0;
        cursorDelay = prevTierEnd + gap;
      }

      // ── Duration: use explicit override if present, else weight default ─
      const rawDuration = item.duration ?? WEIGHT_DURATION[weight];
      const duration = this.#isUncertain
        ? _scaleUncertain(rawDuration)
        : _snap(rawDuration);

      const delay = cursorDelay;
      const endAt = delay + duration;

      // ── Update tier tracking ──────────────────────────────────────────
      if (prevTier !== weight) {
        // Entering this tier for the first time
        prevTierEnd = endAt;
      } else {
        // Still within the same tier — track the latest end across all items
        prevTierEnd = Math.max(prevTierEnd, endAt);
      }

      // ── Advance cursor within tier ────────────────────────────────────
      // PRIMARY items stagger: next item starts 1u after this one.
      // SECONDARY/TERTIARY items: cursor stays put → next item is simultaneous.
      cursorDelay = delay + WEIGHT_STAGGER[weight];
      prevTier = weight;

      return {
        targetId: item.targetId,
        change:   item.change,
        weight,
        delay,
        duration,
        meta:     item.meta ?? {},
      };
    });

    // Total duration: end of the last item to settle, minimum PERCEPTION_FLOOR
    const totalDuration = Math.max(
      PERCEPTION_FLOOR,
      ...items.map(i => i.delay + i.duration),
    );

    return {
      id:            `${++this.#sequenceCounter}`,
      items,
      totalDuration,
      isInterrupt:   isRedirecting,
      redirectsFrom: isRedirecting ? fromId : null,
    };
  }

  // ── Sequence emission ─────────────────────────────────────────────────────

  /**
   * Emit a PERCEPTION_SEQUENCE event and track it as the active sequence.
   * Tracking enables interruption progress calculation.
   *
   * @param {{ id, items, totalDuration, isInterrupt, redirectsFrom }} sequence
   */
  _emitSequence(sequence) {
    // Track as active — needed for PERCEPTION_INTERRUPT progress calculation
    clearTimeout(this.#activeSequenceTimer);
    this.#activeSequence = {
      id:            sequence.id,
      startedAt:     Date.now(),
      totalDuration: sequence.totalDuration,
    };

    // Auto-clear tracking when sequence completes (it is no longer "active")
    this.#activeSequenceTimer = setTimeout(() => {
      if (this.#activeSequence?.id === sequence.id) {
        this.#activeSequence = null;
      }
    }, sequence.totalDuration);

    this.#eventBus.emit(EVENT.PERCEPTION_SEQUENCE, sequence);
  }

  // ── Anticipation ─────────────────────────────────────────────────────────

  /**
   * Schedule a PERCEPTION_ANTICIPATE event to fire ANTICIPATION_WINDOW (160ms) from now.
   * If another anticipation is already scheduled, the new one replaces it.
   *
   * Cancellation protocol (Fix 3):
   *   After PERCEPTION_ANTICIPATE fires, an abandon timer starts. If the anticipated
   *   event (layer reveal) does not arrive within ANTICIPATION_ABANDON_MS (800ms),
   *   PERCEPTION_ANTICIPATE_CANCELLED is emitted — the UI must exit its "prepared" state.
   *   This prevents the UI from being indefinitely stuck in a preparation posture
   *   when the anticipated reveal is blocked by unexpected conditions.
   *
   * @param {string} forTarget - semantic description of what is being anticipated
   */
  _scheduleAnticipation(forTarget) {
    // Cancel both the announce timer and any running abandon timer
    clearTimeout(this.#anticipationTimer);
    clearTimeout(this.#anticipationAbandonTimer);
    this.#anticipationFired     = false;
    this.#anticipationForTarget = forTarget;

    this.#anticipationTimer = setTimeout(() => {
      this.#anticipationTimer = null;
      this.#anticipationFired = true;

      this.#eventBus.emit(EVENT.PERCEPTION_ANTICIPATE, {
        for:    forTarget,
        window: ANTICIPATION_WINDOW,
      });

      // Start abandon countdown — if the reveal doesn't arrive within this window,
      // the UI must be told that the anticipated event is not coming.
      this.#anticipationAbandonTimer = setTimeout(() => {
        this.#anticipationAbandonTimer = null;
        if (this.#anticipationFired) {
          this.#anticipationFired     = false;
          this.#anticipationForTarget = null;
          this.#eventBus.emit(EVENT.PERCEPTION_ANTICIPATE_CANCELLED, {
            for:    forTarget,
            reason: 'timeout',
          });
        }
      }, ANTICIPATION_ABANDON_MS);
    }, ANTICIPATION_WINDOW);
  }

  /**
   * Cancel any pending or fired anticipation silently.
   * If PERCEPTION_ANTICIPATE was already emitted, fire PERCEPTION_ANTICIPATE_CANCELLED.
   *
   * @param {string} reason - why anticipation is being cancelled
   */
  _cancelAnticipation(reason) {
    clearTimeout(this.#anticipationTimer);
    clearTimeout(this.#anticipationAbandonTimer);
    this.#anticipationTimer       = null;
    this.#anticipationAbandonTimer = null;

    if (this.#anticipationFired) {
      const forTarget = this.#anticipationForTarget;
      this.#anticipationFired     = false;
      this.#anticipationForTarget = null;
      this.#eventBus.emit(EVENT.PERCEPTION_ANTICIPATE_CANCELLED, { for: forTarget, reason });
    } else {
      this.#anticipationFired     = false;
      this.#anticipationForTarget = null;
    }
  }

  // ── Interruption ──────────────────────────────────────────────────────────

  /**
   * Interrupt the current active sequence and signal graceful redirect.
   *
   * Clears pending batch (stale items must not emit after interruption).
   * Emits PERCEPTION_INTERRUPT with current progress.
   * Marks the next sequence as a redirect (1u buffer delay on all items).
   *
   * @param {string} reason
   */
  _emitInterrupt(reason) {
    // Clear pending coalesce — no stale events should fire after interruption
    clearTimeout(this.#coalesceTimer);
    this.#coalesceTimer  = null;
    this.#pendingBatch   = [];

    // Cancel anticipation — the anticipated reveal is not coming after an interruption.
    // If PERCEPTION_ANTICIPATE was already fired, emits PERCEPTION_ANTICIPATE_CANCELLED
    // so the UI can exit the "prepared" state it entered. (Fix 3)
    this._cancelAnticipation(reason);

    if (!this.#activeSequence) return; // nothing in motion — nothing to interrupt

    const interruptedAt  = Date.now();
    const elapsed        = interruptedAt - this.#activeSequence.startedAt;
    const totalDuration  = this.#activeSequence.totalDuration;

    // Time-based progress clamped to [0, 1].
    // The payload also includes raw timing fields (startedAt, totalDuration, interruptedAt)
    // so the UI layer can recalculate progress at actual render time — correcting for
    // any frame drops between when this interrupt fired and when the UI processes it. (Fix 4)
    const progress = Math.min(1, elapsed / Math.max(1, totalDuration));

    this.#eventBus.emit(EVENT.PERCEPTION_INTERRUPT, {
      sequenceId:    this.#activeSequence.id,
      progress,
      reason,
      // Raw timing — UI can compute frame-accurate progress at render time:
      //   actualProgress = (renderTime - startedAt) / totalDuration
      startedAt:     this.#activeSequence.startedAt,
      totalDuration,
      interruptedAt,
    });

    // Mark next sequence as a redirect — it will start with REDIRECT_BUFFER delay
    this.#redirectPending         = true;
    this.#redirectPendingFromId   = this.#activeSequence.id;

    clearTimeout(this.#activeSequenceTimer);
    this.#activeSequence = null;
  }
}

// ── Pure helpers ───────────────────────────────────────────────────────────

/**
 * Return the ID of the layer that follows `layerId` in reveal order, or null.
 */
function _nextLayerId(layerId) {
  const idx = LAYER_ORDER.indexOf(layerId);
  return (idx >= 0 && idx < LAYER_ORDER.length - 1) ? LAYER_ORDER[idx + 1] : null;
}
