/**
 * MIDAN Engine — AnalysisPipeline
 * ONE API call per analysis run. Three phases partition that single response
 * for sequential revelation with interruption-safe boundaries.
 *
 * The spec says "halt at current conclusion boundary (never mid-conclusion)."
 * Conclusions within each phase are applied one by one, with an abort check
 * between each — this is the "conclusion boundary" the spec describes.
 *
 * Phase checkpointing: each phase writes a checkpoint on completion.
 * On PAUSED→MERGING: resume from last valid checkpoint, skipping completed phases.
 * On PAUSED→DEEPENING: only targeted component re-runs.
 *
 * INVARIANT: ANALYZING never runs on uncorrected interpretation (#9).
 * INVARIANT: output always sequenced — never dumped (#5).
 */

import { PIPELINE_STATUS, EVENT } from '../core/constants.js';

export class AnalysisPipeline {
  #store;
  #eventBus;
  #apiAdapter;
  #performanceSync;
  #abortController;
  #isRunning;
  #cachedPhaseData;   // single API response, partitioned into phase buckets

  constructor(store, eventBus, apiAdapter, performanceSync) {
    this.#store          = store;
    this.#eventBus       = eventBus;
    this.#apiAdapter     = apiAdapter;
    this.#performanceSync = performanceSync;
    this.#abortController = null;
    this.#isRunning      = false;
    this.#cachedPhaseData = null;
  }

  /**
   * Start a fresh pipeline run.
   * ONE API call; three sequential phases partition the response.
   *
   * @param {Object} [options]
   * @param {number} [options.resumeFromPhase] - resume from checkpoint (MERGING path)
   * @param {string} [options.inputClass]      - COMPLETE | DIRECTIVE | PARTIAL
   * @param {AbortSignal} [options.signal]     - external cancellation signal from engine
   */
  async run({ resumeFromPhase = null, inputClass = 'COMPLETE', signal: externalSignal = null } = {}) {
    if (this.#isRunning) {
      console.warn('[Pipeline] already running — call interrupt() first');
      return;
    }

    // Bail immediately if the caller already aborted before we started
    if (externalSignal?.aborted) return;

    const state = this.#store.getState();

    // Invariant #9: ANALYZING never runs on uncorrected interpretation (DIRECTIVE skips)
    if (inputClass !== 'DIRECTIVE' && !this._isInterpretationResolved(state)) {
      console.warn('[Pipeline] interpretation not resolved — pipeline blocked');
      return;
    }

    this.#abortController = new AbortController();

    // Named handler so we can remove it in finally — anonymous listeners accumulate
    // across runs and keep the old AbortController alive, leaking memory.
    let _onExternalAbort = null;
    if (externalSignal) {
      _onExternalAbort = () => this.#abortController.abort();
      externalSignal.addEventListener('abort', _onExternalAbort);
    }

    this.#isRunning = true;

    this.#store.update(
      (current) => ({
        pipeline_state: {
          ...current.pipeline_state,
          status:         PIPELINE_STATUS.RUNNING,
          interrupted_at: null,
        },
      }),
      { sourceLayer: 'PIPELINE' },
    );

    this.#eventBus.emit(EVENT.PIPELINE_STARTED, { resumeFromPhase, inputClass });

    try {
      await this._execute(resumeFromPhase, inputClass);
    } catch (err) {
      if (err?.name === 'AbortError') {
        // _applyConclusion threw after macrotask yield; interrupt() already wrote
        // INTERRUPTED status and emitted the event — just clear current_phase.
        this._setCurrentPhase(null);
      } else {
        console.error('[Pipeline] unexpected error', err);
        this._setPipelineStatus(PIPELINE_STATUS.IDLE);
      }
    } finally {
      this.#isRunning       = false;
      this.#abortController = null;
      if (externalSignal && _onExternalAbort) {
        externalSignal.removeEventListener('abort', _onExternalAbort);
      }
    }
  }

  /**
   * Interrupt at the current conclusion boundary.
   * Preserves all completed phase checkpoints.
   */
  interrupt() {
    if (!this.#isRunning) return;

    this.#abortController?.abort();
    // Do NOT null abortController here — _execute() checks it after await

    const state = this.#store.getState();
    this.#store.update(
      (current) => ({
        pipeline_state: {
          ...current.pipeline_state,
          status:        PIPELINE_STATUS.INTERRUPTED,
          interrupted_at: {
            phase_id:  current.pipeline_state.current_phase,
            timestamp: Date.now(),
          },
        },
      }),
      { sourceLayer: 'PIPELINE' },
    );

    this.#eventBus.emit(EVENT.PIPELINE_INTERRUPTED, {
      phase: this.#store.getState().pipeline_state.current_phase,
    });
  }

  /**
   * Determine which phase to resume from after PAUSED → new input.
   * Per spec resume rules:
   *   New primary signal  → invalidate all checkpoints, restart Phase 1
   *   New Tier 1 signal   → invalidate Phase 2+3, resume Phase 2
   *   New Tier 2 signal   → invalidate Phase 3, resume Phase 3
   *   No new signals      → resume from interrupted_at.phase_id
   *
   * @param {Object} newSignals - { newPrimary, newTier1, newTier2 }
   * @returns {number} phase to start from
   */
  determineResumePhase({ newPrimary = false, newTier1 = false, newTier2 = false } = {}) {
    if (newPrimary) {
      this._invalidateCheckpointsFrom(1);
      this.#cachedPhaseData = null; // force fresh API call
      return 1;
    }
    if (newTier1) {
      this._invalidateCheckpointsFrom(2);
      this.#cachedPhaseData = null;
      return 2;
    }
    if (newTier2) {
      this._invalidateCheckpointsFrom(3);
      return 3;
    }
    const state = this.#store.getState();
    return state.pipeline_state.interrupted_at?.phase_id ?? 1;
  }

  get isRunning() { return this.#isRunning; }

  // ── Phase Execution ────────────────────────────────────────────────────────

  async _execute(resumeFromPhase, inputClass) {
    const signal     = this.#abortController?.signal;
    const startPhase = resumeFromPhase ?? 1;
    const state      = this.#store.getState();
    const checkpoints = state.pipeline_state.checkpoints;

    // ── Single API call (Phase 1 only, unless resuming with cached data) ──
    if (startPhase === 1 || !this.#cachedPhaseData) {
      this._setCurrentPhase(1);

      if (signal?.aborted) return this._onInterrupt(1);

      // Performance floor gate: start timer before API call, await floor before reveal.
      // Spec invariant #11: computation time does NOT determine perceived speed.
      const perfHandle = this.#performanceSync?.startPhase('analyzing');
      const apiStart   = Date.now();

      const phaseData = await this.#apiAdapter.analyzeForPipeline(
        state,
        { signal, inputClass },
      );

      const actualMs = Date.now() - apiStart;
      perfHandle?.complete(actualMs);

      if (signal?.aborted) return this._onInterrupt(1);
      if (!phaseData) {
        this._setPipelineStatus(PIPELINE_STATUS.IDLE);
        return;
      }

      this.#cachedPhaseData = phaseData;

      // Await the performance floor — CASE 1 holds here, CASE 2-4 resolve immediately or after heartbeats.
      // No conclusions are revealed until this resolves.
      await (perfHandle?.wait ?? Promise.resolve());

      if (signal?.aborted) return this._onInterrupt(1);
    }

    const { phase1, phase2, phase3, assumptions } = this.#cachedPhaseData;

    // Apply shared assumptions before phase conclusions
    if (assumptions?.length > 0) {
      this.#store.update(
        (current) => ({
          assumptions: {
            active: [...current.assumptions.active, ...assumptions],
          },
        }),
        { sourceLayer: 'PIPELINE' },
      );
    }

    // ── Phase 1 ────────────────────────────────────────────────────────────
    if (startPhase <= 1 && !this._isCheckpointValid(checkpoints, 1)) {
      this._setCurrentPhase(1);

      for (const conclusion of (phase1?.conclusions ?? [])) {
        if (signal?.aborted) return this._onInterrupt(1);
        await this._applyConclusion(conclusion, 1);
      }

      this._writeCheckpoint(1);
    }

    // ── Phase 2 ────────────────────────────────────────────────────────────
    if (startPhase <= 2 && !this._isCheckpointValid(checkpoints, 2)) {
      this._setCurrentPhase(2);

      for (const conclusion of (phase2?.conclusions ?? [])) {
        if (signal?.aborted) return this._onInterrupt(2);
        await this._applyConclusion(conclusion, 2);
      }

      this._writeCheckpoint(2);
    }

    // ── Phase 3 ────────────────────────────────────────────────────────────
    if (startPhase <= 3 && !this._isCheckpointValid(checkpoints, 3)) {
      this._setCurrentPhase(3);

      for (const conclusion of (phase3?.conclusions ?? [])) {
        if (signal?.aborted) return this._onInterrupt(3);
        await this._applyConclusion(conclusion, 3);
      }

      this._writeCheckpoint(3);
    }

    this._setPipelineStatus(PIPELINE_STATUS.COMPLETE);
    this.#eventBus.emit(EVENT.PIPELINE_COMPLETED, {
      checkpoints: this.#store.getState().pipeline_state.checkpoints,
    });
  }

  /**
   * Apply a single conclusion to SESSION_STATE.
   * Yields a full macrotask turn between conclusions — this is the "conclusion boundary"
   * the spec requires. A microtask (Promise.resolve) is NOT sufficient because
   * AbortController signals and other macrotask work won't have run yet.
   */
  async _applyConclusion(conclusion, phase) {
    // Macrotask yield — browser gets a full event loop turn between each conclusion
    await new Promise(r => setTimeout(r, 0));

    // Explicit abort check after yielding — abort signal only fires on the next tick
    if (this.#abortController?.signal?.aborted) {
      throw new DOMException('Pipeline interrupted at conclusion boundary', 'AbortError');
    }

    const { id, text, signals = [], assumptions = [], meta = {} } = conclusion;

    this.#store.update(
      (current) => ({
        output: {
          ...current.output,
          conclusions: {
            ...current.output.conclusions,
            [id]: {
              text,
              signals,
              assumptions,
              meta,
              retracted:  false,
              version:    current.version,
              phase,
              updated_at: Date.now(),
            },
          },
        },
      }),
      { sourceLayer: 'PIPELINE' },
    );

    this.#eventBus.emit(
      EVENT.CONCLUSION_UPDATED,
      { conclusionId: id, status: 'applied', phase },
      this.#store.getState().version,
    );
  }

  _setCurrentPhase(phase) {
    this.#store.update(
      (current) => ({ pipeline_state: { ...current.pipeline_state, current_phase: phase } }),
      { sourceLayer: 'PIPELINE' },
    );
  }

  _writeCheckpoint(phaseId) {
    const state = this.#store.getState();
    const checkpoint = {
      phase_id:              phaseId,
      completed_at:          Date.now(),
      completed_conclusions: Object.keys(state.output.conclusions),
      state_version:         state.version,
    };

    this.#store.update(
      (current) => ({
        pipeline_state: {
          ...current.pipeline_state,
          checkpoints: [
            ...current.pipeline_state.checkpoints.filter(c => c.phase_id !== phaseId),
            checkpoint,
          ],
        },
      }),
      { sourceLayer: 'PIPELINE' },
    );

    this.#eventBus.emit(EVENT.PIPELINE_CHECKPOINT, { checkpoint });
  }

  _isCheckpointValid(checkpoints, phaseId) {
    const cp = checkpoints?.find(c => c.phase_id === phaseId);
    if (!cp) return false;
    const state = this.#store.getState();
    // Valid if state hasn't advanced more than 3 versions since checkpoint
    return (state.version - cp.state_version) <= 3;
  }

  _invalidateCheckpointsFrom(fromPhase) {
    this.#store.update(
      (current) => ({
        pipeline_state: {
          ...current.pipeline_state,
          checkpoints: current.pipeline_state.checkpoints.filter(c => c.phase_id < fromPhase),
        },
      }),
      { sourceLayer: 'PIPELINE' },
    );
  }

  _setPipelineStatus(status) {
    this.#store.update(
      (current) => ({
        pipeline_state: { ...current.pipeline_state, status, current_phase: null },
      }),
      { sourceLayer: 'PIPELINE' },
    );
  }

  _onInterrupt(phase) {
    this._setCurrentPhase(null);
    this.#eventBus.emit(EVENT.PIPELINE_INTERRUPTED, { phase });
  }

  _isInterpretationResolved(state) {
    return state.interpretation.problem?.value != null;
  }
}
