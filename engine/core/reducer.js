/**
 * MIDAN Engine — StateReducer
 * The ONLY entity that calls store.update().
 * Layers emit events; the reducer applies them to state.
 *
 * PURITY RULE (enforced):
 *   Reducer handlers are synchronous and side-effect-free.
 *   They ONLY call store.update(). No eventBus.emit(), no async, no timers.
 *   Side effects (completeness sync, COMPLETENESS_UPDATED emission, queue draining)
 *   are owned by the store and by engine-level event handlers in index.js.
 *
 * Scheduling guarantees (JavaScript event loop):
 *   Layer 1 — synchronous (immediate dispatch, macrotask: none)
 *   Layer 3 — word-boundary + setTimeout (macrotask, fires before Layer 2 at same keydown)
 *   Layer 2 — debounced setTimeout (macrotask, fires after Layer 1/3)
 *
 * Conflict resolution (per spec):
 *   Layer 3 wins on signal classification
 *   Layer 2 wins on component text values
 * Enforced by separate namespaces — signals.registry (L3) vs interpretation (L2).
 */

import { EVENT, INTERP_COMPONENT } from './constants.js';
import { SignalRegistry } from '../signals/registry.js';

export class StateReducer {
  #store;
  #eventBus;
  #signalRegistry;
  #unsubscribers;

  constructor(store, eventBus) {
    this.#store          = store;
    this.#eventBus       = eventBus;
    this.#signalRegistry = new SignalRegistry(store, eventBus);
    this.#unsubscribers  = [];

    this._bindAll();
  }

  get signalRegistry() { return this.#signalRegistry; }

  destroy() {
    this.#unsubscribers.forEach(fn => fn());
    this.#unsubscribers = [];
  }

  // ── Event → State mappings ────────────────────────────────────────────────
  // Each handler: receive event → write state → return. Nothing else.

  _bindAll() {
    // ── Layer 1: raw input ───────────────────────────────────────────────────
    this.#unsubscribers.push(
      this.#eventBus.on(EVENT.INPUT_RAW_UPDATED, ({ payload, version }) => {
        this.#store.update(
          () => ({
            input: {
              raw:         payload.raw,
              length:      payload.length,
              last_change: payload.timestamp,
              change_type: payload.change_type,
            },
          }),
          { sourceLayer: 'REDUCER', capturedVersion: version, eventType: EVENT.INPUT_RAW_UPDATED },
        );
        // No onStateAdvanced() — store.update() calls it internally.
      }),
    );

    // ── Layer 2: interpretation components (granular — only changed fields) ──
    this.#unsubscribers.push(
      this.#eventBus.on(EVENT.INTERPRETATION_UPDATED, ({ payload, version }) => {
        const { components } = payload;
        if (!components || Object.keys(components).length === 0) return;

        this.#store.update(
          (current) => {
            const patch = {};
            for (const [key, value] of Object.entries(components)) {
              // Respect user_override — never overwrite flagged components
              if (current.interpretation[key]?.user_override === true) continue;
              if (Object.values(INTERP_COMPONENT).includes(key)) {
                patch[key] = value;
              }
            }
            if (Object.keys(patch).length === 0) return null;
            return { interpretation: { ...current.interpretation, ...patch } };
          },
          { sourceLayer: 'REDUCER', capturedVersion: version, eventType: EVENT.INTERPRETATION_UPDATED },
        );
      }),
    );

    // ── Layer 3: signal add ──────────────────────────────────────────────────
    this.#unsubscribers.push(
      this.#eventBus.on(EVENT.SIGNAL_DETECTED, ({ payload, version }) => {
        if (!payload?.signal) return;
        // SignalRegistry writes to the store (rank assignment, deduplication).
        // It may emit SIGNAL_PROMOTED / SIGNAL_DEMOTED as informational events —
        // those are permitted because they do not write state and cannot loop back
        // into this handler.
        this.#signalRegistry.add(payload.signal, version);
        // Completeness is synced by the engine-level event handler (index.js).
      }),
    );

    // ── Layer 3: signal remove ───────────────────────────────────────────────
    this.#unsubscribers.push(
      this.#eventBus.on(EVENT.SIGNAL_REMOVED, ({ payload, version }) => {
        if (!payload?.type) return;
        this.#signalRegistry.remove(payload.type, version);
        // Completeness is synced by the engine-level event handler (index.js).
      }),
    );
  }
}
