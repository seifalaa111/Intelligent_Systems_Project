/**
 * MIDAN Engine — ThreadManager
 * One active thread always. Others: state→'inactive', snapshot saved.
 * Thread switch: current→snapshot, target→restored from snapshot.
 * Nothing lost. Everything deactivated/reactivated.
 * Summary: auto-generated from Layer 1 first sentence (or problem component).
 */

import { EVENT, ENGINE_SCHEMA_VERSION } from '../core/constants.js';
import { createCompactBase, generatePatch, restoreFromSnapshot } from './patch-set.js';
import { PatchCompactor } from './compaction.js';
import { createInitialState, cloneState } from '../core/session-state.js';

export class ThreadManager {
  #store;
  #eventBus;
  #compactor;
  #compactionBuffers; // threadId → Op[] (patches accumulated during compaction)

  constructor(store, eventBus) {
    this.#store             = store;
    this.#eventBus          = eventBus;
    this.#compactor         = new PatchCompactor();
    this.#compactionBuffers = new Map();
  }

  /**
   * Archive the currently active thread (snapshot current state).
   * Called before switching or on CLOSING.
   */
  archiveActive() {
    const state    = this.#store.getState();
    const threadId = state.threads.active_id;
    const thread   = state.threads.registry.find(t => t.id === threadId);

    if (!thread) return;

    const summary = this._generateSummary(state);
    const base    = thread.base ?? createCompactBase(state);
    const diff    = generatePatch(base, state);

    // Check compaction
    const needsCompaction = diff.length > 0;

    const updatedThread = {
      ...thread,
      state:          'inactive',
      summary,
      base,
      diff,
      schema_version: ENGINE_SCHEMA_VERSION,
    };

    this.#store.update(
      (current) => ({
        threads: {
          ...current.threads,
          registry: current.threads.registry.map(t =>
            t.id === threadId ? updatedThread : t,
          ),
        },
      }),
      { sourceLayer: 'SYSTEM' },
    );

    this.#eventBus.emit(EVENT.THREAD_ARCHIVED, { threadId, summary });

    // Trigger async compaction if needed
    if (needsCompaction) {
      this.#compactor.compactIfNeeded(
        updatedThread,
        ({ newBase, newDiff }) => this._applyCompaction(threadId, newBase, newDiff),
        () => this._drainBuffer(threadId),
      );
    }
  }

  /**
   * Switch to a target thread by ID.
   * 1. Archive current thread.
   * 2. Restore target thread from snapshot.
   * No recompute if versions match.
   *
   * @param {string} targetId
   */
  switchTo(targetId) {
    const state    = this.#store.getState();
    const currentId = state.threads.active_id;

    if (targetId === currentId) return;

    // Archive current
    this.archiveActive();

    // Restore target
    const target = this.#store.getState().threads.registry.find(t => t.id === targetId);
    if (!target || !target.base) {
      console.warn('[ThreadManager] target thread has no snapshot', targetId);
      return;
    }

    const defaults = cloneState(createInitialState());
    const restored = restoreFromSnapshot(target.base, target.diff ?? [], defaults);

    // Schema version check — if the snapshot was created by an older engine,
    // the conclusion shape may have changed. Flag all conclusions as stale.
    const schemaMismatch = (target.schema_version ?? 0) !== ENGINE_SCHEMA_VERSION;

    let restoredOutput = restored.output;
    if (schemaMismatch && restored.output?.conclusions) {
      const stalened = {};
      for (const [id, conclusion] of Object.entries(restored.output.conclusions)) {
        stalened[id] = { ...conclusion, stale: true };
      }
      restoredOutput = { ...restored.output, conclusions: stalened };
    }

    this.#store.update(
      () => ({
        ...restored,
        output: restoredOutput,
        threads: {
          active_id: targetId,
          registry: this.#store.getState().threads.registry.map(t =>
            t.id === targetId
              ? { ...t, state: 'active' }
              : t,
          ),
        },
      }),
      { sourceLayer: 'SYSTEM' },
    );

    const staleConclusions = schemaMismatch
      ? Object.keys(restoredOutput?.conclusions ?? {})
      : [];

    this.#eventBus.emit(EVENT.THREAD_SWITCHED, {
      from:             currentId,
      to:               targetId,
      needsRecompute:   schemaMismatch,
      staleConclusions,
    });
  }

  /**
   * Create a new thread (called on CLOSING/redirect).
   * Assigns a new ID, preserves registry, clears active session.
   *
   * @returns {string} new thread ID
   */
  createNew() {
    const state     = this.#store.getState();
    const newId     = `thread_${Date.now()}`;
    const newThread = {
      id:           newId,
      created_at:   Date.now(),
      state:        'active',
      summary:      null,
      base_version: state.version,
      base:         null,
      diff:         [],
    };

    this.#store.update(
      (current) => ({
        threads: {
          active_id: newId,
          registry:  [...current.threads.registry, newThread],
        },
      }),
      { sourceLayer: 'SYSTEM' },
    );

    this.#eventBus.emit(EVENT.THREAD_CREATED, { threadId: newId });
    return newId;
  }

  /**
   * Get all inactive threads (for context rail display).
   * Max 5 visible, most-recent-first.
   */
  getInactiveThreads() {
    const state = this.#store.getState();
    return state.threads.registry
      .filter(t => t.state === 'inactive' && t.summary)
      .sort((a, b) => b.created_at - a.created_at)
      .slice(0, 5);
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  _generateSummary(state) {
    // Use Layer 1 output first sentence from Layer 1 (summary), else problem component
    const layer1Text = Object.values(state.output.conclusions)[0]?.text;
    if (layer1Text) {
      const sentence = layer1Text.split(/[.!?]/)[0];
      return sentence.trim().slice(0, 120);
    }
    return state.interpretation.problem?.value?.slice(0, 120) ?? 'Untitled thread';
  }

  _applyCompaction(threadId, newBase, newDiff) {
    this.#store.update(
      (current) => ({
        threads: {
          ...current.threads,
          registry: current.threads.registry.map(t =>
            t.id === threadId ? { ...t, base: newBase, diff: newDiff } : t,
          ),
        },
      }),
      { sourceLayer: 'SYSTEM' },
    );

    this.#eventBus.emit(EVENT.THREAD_SNAPSHOT_COMPACTED, { threadId });
  }

  _drainBuffer(threadId) {
    const buf = this.#compactionBuffers.get(threadId) ?? [];
    this.#compactionBuffers.delete(threadId);
    return buf;
  }

  /** Buffer a patch during active compaction. */
  _bufferPatch(threadId, patch) {
    if (!this.#compactionBuffers.has(threadId)) {
      this.#compactionBuffers.set(threadId, []);
    }
    this.#compactionBuffers.get(threadId).push(...patch);
  }
}
