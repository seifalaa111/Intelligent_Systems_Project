/**
 * MIDAN Engine — PhaseAnimationRegistry
 * Tracks all in-progress animations for sync point enforcement.
 * Sync points check this registry before advancing phases.
 * No phase N+1 animation begins until all phase N animations are at rest.
 */

import { EVENT } from '../core/constants.js';

export class PhaseAnimationRegistry {
  #active;     // animationId → { estimatedDuration, startedAt }
  #eventBus;
  #onEmpty;    // callback when registry becomes empty

  constructor(eventBus) {
    this.#active   = new Map();
    this.#eventBus = eventBus;
    this.#onEmpty  = null;
  }

  /**
   * Register an animation as in-progress.
   * @param {string} animationId
   * @param {number} estimatedDuration - ms
   */
  register(animationId, estimatedDuration) {
    this.#active.set(animationId, {
      estimatedDuration,
      startedAt: Date.now(),
    });

    this.#eventBus?.emit(EVENT.ANIMATION_REGISTERED, { animationId, estimatedDuration });
  }

  /**
   * Mark an animation as complete.
   * @param {string} animationId
   */
  complete(animationId) {
    const wasPresent = this.#active.has(animationId);
    this.#active.delete(animationId);

    this.#eventBus?.emit(EVENT.ANIMATION_COMPLETED, { animationId });

    if (wasPresent && this.isEmpty && this.#onEmpty) {
      this.#onEmpty();
      this.#onEmpty = null;
    }
  }

  /** Returns true when all registered animations have completed. */
  get isEmpty() {
    return this.#active.size === 0;
  }

  /**
   * Latest estimated completion time across all active animations.
   * @returns {number} ms timestamp (or Date.now() if none active)
   */
  get latestCompletion() {
    let latest = Date.now();
    for (const { estimatedDuration, startedAt } of this.#active.values()) {
      latest = Math.max(latest, startedAt + estimatedDuration);
    }
    return latest;
  }

  /**
   * Force-complete all active animations (cut to final state).
   * Called when sync point max delay (240ms) is exceeded.
   */
  forceCompleteAll() {
    const ids = [...this.#active.keys()];
    this.#active.clear();
    for (const id of ids) {
      this.#eventBus?.emit(EVENT.ANIMATION_COMPLETED, { animationId: id, forced: true });
    }
    this.#onEmpty?.();
    this.#onEmpty = null;
  }

  /**
   * Register a one-shot callback to fire when registry becomes empty.
   * Replaces any previous onEmpty callback.
   */
  onEmpty(cb) {
    if (this.isEmpty) {
      cb();
    } else {
      this.#onEmpty = cb;
    }
  }

  /** Clear all tracking (e.g., on interruption). */
  reset() {
    this.#active.clear();
    this.#onEmpty = null;
  }

  /** Snapshot of active animations for debugging. */
  getActive() {
    return Object.fromEntries(
      [...this.#active.entries()].map(([id, meta]) => [id, { ...meta }]),
    );
  }
}
