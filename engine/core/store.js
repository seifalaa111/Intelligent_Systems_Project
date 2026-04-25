/**
 * MIDAN Engine — SessionStore
 * SINGLE mutable holder of SESSION_STATE.
 *
 * ENFORCEMENT RULES:
 *   1. applyUpdate() is the ONLY write path — no exceptions
 *   2. getState() returns a deep-frozen object — any mutation attempt throws in dev mode
 *   3. Dev mode wraps returned state in a Proxy that throws on any set() call
 *   4. After every commit, a structural hash is stored; before every commit it is verified
 *
 * Write gate: all writes must carry a { sourceLayer } tag.
 * Unknown source layers are rejected in strict mode.
 */

import { createInitialState, applyUpdate, deepFreeze } from './session-state.js';
import { EVENT } from './constants.js';

const KNOWN_LAYERS = new Set([
  'LAYER_1', 'LAYER_2', 'LAYER_3',
  'PIPELINE', 'API', 'USER', 'SYSTEM',
  'CONFIDENCE', 'RECOMPUTE', 'PERFORMANCE',
  'STATE_MACHINE', 'REDUCER', 'UNKNOWN',
]);

const IS_DEV = (() => {
  try { return (typeof process !== 'undefined' && process.env?.NODE_ENV !== 'production'); }
  catch { return true; } // browser without process — assume dev
})();

export class SessionStore {
  #state;
  #listeners;
  #violationLog;
  #eventBus;
  #devHash;           // structural hash after last commit (dev mode only)
  #writeCount;        // total committed writes (monotonic)
  #snapshotHook;      // optional (version, state, meta) => void for debug/replay/audit

  /**
   * @param {import('./event-bus.js').EventBus} eventBus
   */
  constructor(eventBus) {
    this.#state        = createInitialState();
    this.#listeners    = new Set();
    this.#violationLog = [];
    this.#eventBus     = eventBus;
    this.#writeCount   = 0;
    this.#devHash      = IS_DEV ? this.#hash(this.#state) : null;
    this.#snapshotHook = null;
  }

  /**
   * Register a snapshot hook for debugging, replay, and audit.
   * Called synchronously after every successful state commit.
   *
   * @param {Function|null} fn - (version: number, state: Readonly<Object>, meta: Object) => void
   *   meta = { sourceLayer, capturedVersion, eventType? }
   *   Pass null to clear the hook.
   */
  setSnapshotHook(fn) {
    this.#snapshotHook = typeof fn === 'function' ? fn : null;
  }

  /**
   * Returns current frozen SESSION_STATE.
   * In dev mode: wrapped in a mutation-detection Proxy.
   * NEVER mutate the returned value.
   */
  getState() {
    if (IS_DEV) {
      this.#assertNoMutation();
      return _mutationProxy(this.#state, 'SESSION_STATE');
    }
    return this.#state;
  }

  /**
   * The ONLY write path.
   * Validates source layer, checks for external mutation, applies update, re-freezes.
   *
   * @param {Function} updater          - (currentState) => partialUpdate | null
   * @param {Object}   options
   * @param {string}   options.sourceLayer     - required write tag
   * @param {number}   [options.capturedVersion] - version when layer captured its input
   * @returns {boolean} whether the write was accepted
   */
  update(updater, { sourceLayer = 'UNKNOWN', capturedVersion = null, eventType = null } = {}) {
    // ── Gate 1: known write source ───────────────────────────────────────────
    if (!KNOWN_LAYERS.has(sourceLayer)) {
      this.#logViolation(sourceLayer, 'unknown_layer', this.#state.version);
      console.error(`[Store] REJECTED: unknown sourceLayer "${sourceLayer}"`);
      return false;
    }

    // ── Gate 2: dev-mode external mutation detection ──────────────────────────
    if (IS_DEV) this.#assertNoMutation();

    // ── Gate 3: apply via controlled applyUpdate() ───────────────────────────
    const { state: nextState, accepted, reason } = applyUpdate(
      this.#state,
      updater,
      { sourceLayer, capturedVersion },
    );

    if (!accepted) {
      this.#logViolation(sourceLayer, reason ?? 'rejected', this.#state.version, capturedVersion);
      this.#eventBus?.emit(EVENT.STALE_WRITE_REJECTED, {
        sourceLayer,
        capturedVersion,
        currentVersion: this.#state.version,
        reason,
      });
      return false;
    }

    if (nextState === this.#state) return true; // no-op update

    // ── Commit ────────────────────────────────────────────────────────────────
    this.#state = nextState; // nextState is already deep-frozen by applyUpdate
    this.#writeCount++;

    // Update mutation hash
    if (IS_DEV) this.#devHash = this.#hash(this.#state);

    this.#notify(nextState);

    // Snapshot hook — deferred via queueMicrotask so heavy logging/processing by the
    // caller never blocks the synchronous commit+drain path.
    // State and meta are captured by value at commit time — correct for replay and audit.
    if (this.#snapshotHook) {
      const _hook    = this.#snapshotHook;
      const _version = this.#state.version;
      const _state   = this.#state;
      const _meta    = { sourceLayer, capturedVersion, eventType };
      queueMicrotask(() => {
        try { _hook(_version, _state, _meta); }
        catch (e) { console.error('[Store] snapshotHook error', e); }
      });
    }

    // Drain any pending-queue events that are now current.
    // The store — not the reducer — is the right caller because it owns the version advance.
    this.#eventBus?.onStateAdvanced(this.#state.version);

    return true;
  }

  /**
   * Hard-reset to initial state. Preserves threads.registry if requested.
   * Only valid source: 'SYSTEM'.
   */
  resetSession(preserveThreadRegistry = true) {
    const threads = preserveThreadRegistry
      ? JSON.parse(JSON.stringify(this.#state.threads))
      : undefined;

    this.#state      = createInitialState();
    this.#writeCount = 0;

    if (IS_DEV) this.#devHash = this.#hash(this.#state);

    if (threads) {
      this.update(() => ({ threads }), { sourceLayer: 'SYSTEM' });
    } else {
      this.#notify(this.#state);
    }
  }

  /** Subscribe to all state changes. Returns unsubscribe fn. */
  subscribe(listener) {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  get writeCount() { return this.#writeCount; }

  getViolationLog() { return [...this.#violationLog]; }

  // ── Internal ──────────────────────────────────────────────────────────────

  #notify(state) {
    this.#eventBus?.emit(EVENT.SESSION_STATE_UPDATED, { version: state.version });
    this.#listeners.forEach(fn => {
      try { fn(state); } catch (e) { console.error('[Store] listener error', e); }
    });
  }

  #assertNoMutation() {
    if (this.#devHash === null) return;
    const current = this.#hash(this.#state);
    if (current !== this.#devHash) {
      const err = new Error(
        '[Store] ILLEGAL MUTATION DETECTED — state was modified outside of update().\n' +
        'All writes must go through store.update(). Check for direct object mutation.',
      );
      console.error(err);
      throw err;
    }
  }

  #hash(state) {
    // Fast structural hash: JSON serialization is sufficient for dev-mode detection
    try { return JSON.stringify(state); } catch { return String(Date.now()); }
  }

  #logViolation(sourceLayer, reason, currentVersion, capturedVersion = null) {
    this.#violationLog.push({ timestamp: Date.now(), sourceLayer, reason, currentVersion, capturedVersion });
    if (this.#violationLog.length > 200) this.#violationLog.splice(0, 50);
  }
}

// ── Dev-mode Proxy ─────────────────────────────────────────────────────────────

function _mutationProxy(obj, path) {
  if (!obj || typeof obj !== 'object') return obj;

  return new Proxy(obj, {
    set(target, key) {
      throw new TypeError(
        `[Store] MUTATION BLOCKED at "${path}.${String(key)}" — ` +
        'SESSION_STATE is read-only. Use store.update() to change state.',
      );
    },
    deleteProperty(target, key) {
      throw new TypeError(
        `[Store] DELETE BLOCKED at "${path}.${String(key)}" — SESSION_STATE is read-only.`,
      );
    },
    get(target, key) {
      const val = target[key];
      if (val && typeof val === 'object' && key !== '__proto__') {
        return _mutationProxy(val, `${path}.${String(key)}`);
      }
      return val;
    },
  });
}
