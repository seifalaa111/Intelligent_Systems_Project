/**
 * MIDAN Engine — PatchCompactor
 * When patches.length > 50: apply patches to base, generate new COMPACT_BASE, reset patches to [].
 * Runs async. New patches are buffered in COMPACTION_BUFFER during compaction.
 * Thread writes are not blocked during compaction.
 */

import { THREAD_COMPACTION_THRESHOLD } from '../core/constants.js';
import { createCompactBase, generatePatch, applyPatch } from './patch-set.js';

export class PatchCompactor {
  #activeCompactions; // threadId → Promise

  constructor() {
    this.#activeCompactions = new Map();
  }

  /**
   * Check if a thread needs compaction and trigger it if so.
   * Non-blocking: returns immediately. Updates the thread record via callback.
   *
   * @param {Object}   thread          - thread record from threads.registry
   * @param {Function} onComplete      - ({ newBase, newDiff }) => void — called when done
   * @param {Function} getBufferedPatch - () => Op[] — drain compaction buffer
   */
  compactIfNeeded(thread, onComplete, getBufferedPatch) {
    if (!thread || !thread.diff) return;
    if (thread.diff.length <= THREAD_COMPACTION_THRESHOLD) return;
    if (this.#activeCompactions.has(thread.id)) return; // already compacting

    const promise = this._compact(thread, onComplete, getBufferedPatch);
    this.#activeCompactions.set(thread.id, promise);

    promise.finally(() => {
      this.#activeCompactions.delete(thread.id);
    });
  }

  isCompacting(threadId) {
    return this.#activeCompactions.has(threadId);
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  async _compact(thread, onComplete, getBufferedPatch) {
    // Yield to allow buffered patches to accumulate
    await Promise.resolve();

    const base       = thread.base ?? {};
    const patches    = thread.diff ?? [];

    // Apply current patches to base → new compact base
    const newBaseRaw = applyPatch(base, patches);
    const newBase    = createCompactBase(newBaseRaw);

    // Drain the compaction buffer (patches added while we were compacting)
    const buffered = getBufferedPatch ? getBufferedPatch() : [];

    // New PATCH_SET = only buffered patches (compacted patches are now in the base)
    const newDiff = buffered;

    onComplete({ newBase, newDiff });
  }
}
