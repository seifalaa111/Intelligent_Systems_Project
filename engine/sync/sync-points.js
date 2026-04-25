/**
 * MIDAN Engine — SyncPointGate
 * Enforces SP1–SP4 gates.
 * No phase N+1 animation begins until all phase N animations are at rest.
 *
 * SP1 — SIGNAL_STABILIZATION:   registry stable 160ms → clears chip/caret/shadow animations
 * SP2 — INTERPRETATION_LOCK:    block assembled + stable 160ms → clears row/container fades
 * SP3 — ANALYSIS_START:         pipeline begins → clears anticipation animations
 * SP4 — REVEAL_START:           pipeline complete + floor elapsed → clears analysis-phase visuals
 *
 * Max delay: 240ms (3u). After: force-clear, cut remaining animations to final state.
 */

import { SYNC_POINT, SYNC_MAX_DELAY_MS, EVENT } from '../core/constants.js';

export class SyncPointGate {
  #registry;   // PhaseAnimationRegistry
  #eventBus;
  #gates;      // SYNC_POINT → Promise<void> | null

  /**
   * @param {import('./animation-registry.js').PhaseAnimationRegistry} registry
   * @param {import('../core/event-bus.js').EventBus} eventBus
   */
  constructor(registry, eventBus) {
    this.#registry = registry;
    this.#eventBus = eventBus;
    this.#gates    = new Map();
  }

  /**
   * Wait for a sync point to clear.
   * Returns a Promise that resolves when:
   *   a) The animation registry is empty, OR
   *   b) 240ms has elapsed (force-clear path)
   *
   * @param {string} syncPoint - SYNC_POINT constant
   * @returns {Promise<void>}
   */
  async waitForClear(syncPoint) {
    if (this.#registry.isEmpty) {
      this._emitCleared(syncPoint);
      return;
    }

    const latestCompletion = this.#registry.latestCompletion;
    const delay = Math.min(
      latestCompletion - Date.now() + 80, // buffer 1u
      SYNC_MAX_DELAY_MS,
    );

    return new Promise(resolve => {
      const timeoutId = setTimeout(() => {
        if (!this.#registry.isEmpty) {
          this.#registry.forceCompleteAll();
        }
        this._emitCleared(syncPoint);
        resolve();
      }, Math.max(0, delay));

      this.#registry.onEmpty(() => {
        clearTimeout(timeoutId);
        this._emitCleared(syncPoint);
        resolve();
      });
    });
  }

  /**
   * Clear all gates immediately (e.g., on interruption).
   * Fresh registry created for the new phase.
   */
  clearAll() {
    this.#registry.forceCompleteAll();
    this.#gates.clear();
  }

  /**
   * Reset for a new session or thread.
   */
  reset() {
    this.#registry.reset();
    this.#gates.clear();
  }

  _emitCleared(syncPoint) {
    this.#eventBus?.emit(EVENT.SYNC_POINT_CLEARED, { syncPoint });
  }
}
