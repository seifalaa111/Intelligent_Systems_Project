/**
 * MIDAN Engine — SignalRegistry
 * Manages the signal registry within SESSION_STATE.
 * Enforces: PRIMARY is singular, demotion and promotion are simultaneous.
 * Writes to state via the store.
 */

import {
  SIGNAL_RANK, SIGNAL_TIER, SIGNAL_TYPE, SIGNAL_PRIORITY_ORDER,
  EVENT, REVERSAL_GUARD_MS,
} from '../core/constants.js';

export class SignalRegistry {
  #store;
  #eventBus;

  constructor(store, eventBus) {
    this.#store    = store;
    this.#eventBus = eventBus;
  }

  /**
   * Add a detected signal. Assigns rank by hierarchy rules.
   * If new signal outranks current PRIMARY: simultaneous demotion + promotion.
   * @param {Object} signalDescriptor - from detector.js
   * @param {number} capturedVersion
   */
  add(signalDescriptor, capturedVersion) {
    const state     = this.#store.getState();
    const registry  = [...state.signals.registry];

    // Deduplicate: if same type already exists, update instead of add
    const existingIdx = registry.findIndex(s => s.type === signalDescriptor.type);
    if (existingIdx !== -1) {
      // Update existing signal without rank change
      const updated = { ...registry[existingIdx], ...signalDescriptor, version: state.version };
      registry[existingIdx] = updated;
      this.#writeRegistry(registry, capturedVersion);
      return;
    }

    // Assign initial rank
    const { rank, demotedId } = this._assignRank(signalDescriptor, registry);
    const newSignal = {
      ...signalDescriptor,
      rank,
      version: state.version,
    };

    registry.push(newSignal);

    // Simultaneous demotion of current PRIMARY if new signal takes PRIMARY
    if (demotedId) {
      const demIdx = registry.findIndex(s => s.id === demotedId);
      if (demIdx !== -1) {
        registry[demIdx] = { ...registry[demIdx], rank: SIGNAL_RANK.SECONDARY };
        this.#eventBus.emit(EVENT.SIGNAL_DEMOTED, { id: demotedId, newRank: SIGNAL_RANK.SECONDARY });
      }
      this.#eventBus.emit(EVENT.SIGNAL_PROMOTED, { id: newSignal.id, newRank: SIGNAL_RANK.PRIMARY });
    }

    const newPrimaryId = registry.find(s => s.rank === SIGNAL_RANK.PRIMARY)?.id ?? null;

    this.#store.update(
      () => ({
        signals: {
          ...this.#store.getState().signals,
          registry,
          primary_id: newPrimaryId,
        },
      }),
      { sourceLayer: 'LAYER_3', capturedVersion },
    );
    // SIGNAL_DETECTED was already emitted by Layer 3 before reaching this reducer path.
    // Re-emitting here would loop back into the reducer's SIGNAL_DETECTED handler.
  }

  /**
   * Remove a signal by type.
   * Triggers retraction protocol for dependent conclusions if needed.
   * @param {string} type - signal type to remove
   * @param {number} capturedVersion
   * @returns {Object|null} removed signal or null
   */
  remove(type, capturedVersion) {
    const state    = this.#store.getState();
    const registry = [...state.signals.registry];
    const idx      = registry.findIndex(s => s.type === type);

    if (idx === -1) return null;

    const removed = registry[idx];

    // Reversal guard: if signal was added < REVERSAL_GUARD_MS ago, mark as TRANSIENT
    const age = Date.now() - removed.added_at;
    const isTransient = age < REVERSAL_GUARD_MS;

    registry.splice(idx, 1);

    // Reassign ranks after removal
    const reranked = this._rerank(registry);

    const newPrimaryId = reranked.find(s => s.rank === SIGNAL_RANK.PRIMARY)?.id ?? null;

    this.#store.update(
      () => ({
        signals: {
          ...this.#store.getState().signals,
          registry:   reranked,
          primary_id: newPrimaryId,
          last_removed: { type: removed.type, isTransient, removedAt: Date.now() },
        },
      }),
      { sourceLayer: 'LAYER_3', capturedVersion },
    );
    // SIGNAL_REMOVED was already emitted by Layer 3. Re-emitting loops the reducer.
    // isTransient is preserved in signals.last_removed for reversal-guard consumers.

    return { signal: removed, isTransient };
  }

  /**
   * Update an existing signal's confidence (from backend response).
   * @param {string} type
   * @param {string} confidence - 'HIGH' | 'MODERATE' | 'LOW'
   * @param {number} capturedVersion
   */
  updateConfidence(type, confidence, capturedVersion) {
    const state    = this.#store.getState();
    const registry = state.signals.registry.map(s =>
      s.type === type ? { ...s, confidence } : s,
    );

    this.#store.update(
      () => ({ signals: { ...state.signals, registry } }),
      { sourceLayer: 'API', capturedVersion },
    );
  }

  /** Returns current registry snapshot (read-only). */
  getAll() {
    return this.#store.getState().signals.registry;
  }

  /** Returns signal by type or null. */
  getByType(type) {
    return this.#store.getState().signals.registry.find(s => s.type === type) ?? null;
  }

  // ── Rank assignment ────────────────────────────────────────────────────────

  /**
   * Determines the rank for a new signal and which current PRIMARY to demote.
   * Rules from entry spec:
   *   Problem → PRIMARY
   *   Market → PRIMARY if no Problem
   *   Friction (Tier2) → PRIMARY if no Tier1
   *   Domain → weakest PRIMARY
   */
  _assignRank(newSig, existingRegistry) {
    const currentPrimary = existingRegistry.find(s => s.rank === SIGNAL_RANK.PRIMARY);
    const newPriority    = SIGNAL_PRIORITY_ORDER[newSig.type] ?? 99;

    let demotedId = null;
    let rank      = SIGNAL_RANK.TERTIARY;

    if (!currentPrimary) {
      // No primary yet — this becomes primary if Tier1 OR first signal
      if (newSig.tier === SIGNAL_TIER.TIER1 || existingRegistry.length === 0) {
        rank = SIGNAL_RANK.PRIMARY;
      } else {
        rank = SIGNAL_RANK.SECONDARY;
      }
    } else {
      const currentPriority = SIGNAL_PRIORITY_ORDER[currentPrimary.type] ?? 99;

      if (newPriority < currentPriority) {
        // New signal outranks current primary
        rank      = SIGNAL_RANK.PRIMARY;
        demotedId = currentPrimary.id;
      } else if (newSig.tier === SIGNAL_TIER.TIER1 && existingRegistry.filter(s => s.rank === SIGNAL_RANK.SECONDARY).length === 0) {
        rank = SIGNAL_RANK.SECONDARY;
      } else {
        rank = SIGNAL_RANK.TERTIARY;
      }
    }

    return { rank, demotedId };
  }

  /**
   * Re-rank remaining signals after a removal, maintaining hierarchy invariants.
   */
  _rerank(registry) {
    if (registry.length === 0) return [];

    // Sort by priority order
    const sorted = [...registry].sort((a, b) =>
      (SIGNAL_PRIORITY_ORDER[a.type] ?? 99) - (SIGNAL_PRIORITY_ORDER[b.type] ?? 99),
    );

    return sorted.map((sig, idx) => ({
      ...sig,
      rank: idx === 0
        ? SIGNAL_RANK.PRIMARY
        : idx <= 2
          ? SIGNAL_RANK.SECONDARY
          : SIGNAL_RANK.TERTIARY,
    }));
  }

  #writeRegistry(registry, capturedVersion) {
    const state = this.#store.getState();
    const primaryId = registry.find(s => s.rank === SIGNAL_RANK.PRIMARY)?.id ?? null;
    this.#store.update(
      () => ({ signals: { ...state.signals, registry, primary_id: primaryId } }),
      { sourceLayer: 'LAYER_3', capturedVersion },
    );
  }
}
