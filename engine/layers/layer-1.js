/**
 * MIDAN Engine — Layer 1: Raw Input Stream
 *
 * SCHEDULING: synchronous — fires immediately on every keydown, no defer.
 * WRITES: zero direct store writes. Emits INPUT_RAW_UPDATED; StateReducer applies it.
 * PRIORITY: highest — JavaScript synchronous path, no event loop delay.
 *
 * Outputs (via event):
 *   raw, length, last_change, change_type
 */

import { CHANGE_TYPE, EVENT } from '../core/constants.js';

export class RawInputLayer {
  #getState;
  #eventBus;
  #prevRaw;

  /**
   * @param {Function} getState  - () => current SESSION_STATE (read-only)
   * @param {import('../core/event-bus.js').EventBus} eventBus
   */
  constructor(getState, eventBus) {
    this.#getState = getState;
    this.#eventBus = eventBus;
    this.#prevRaw  = '';
  }

  /**
   * Process a raw input change.
   * Called synchronously from the input element's 'input' event handler.
   * Emits event; does NOT write to store.
   *
   * @param {string} currentValue
   * @param {InputEvent|null} domEvent
   */
  process(currentValue, domEvent = null) {
    const state       = this.#getState();
    const change_type = this._detectChangeType(this.#prevRaw, currentValue, domEvent);
    this.#prevRaw     = currentValue;

    // Emit with the current state version as origin version
    this.#eventBus.emit(
      EVENT.INPUT_RAW_UPDATED,
      {
        raw:         currentValue,
        length:      currentValue.length,
        timestamp:   Date.now(),
        change_type,
        prev_length: this.#prevRaw.length,
      },
      state.version, // origin version = current state version at emit time
    );
  }

  reset() { this.#prevRaw = ''; }

  _detectChangeType(prev, current, domEvent) {
    if (domEvent?.inputType) {
      const t = domEvent.inputType;
      if (t === 'insertFromPaste' || t === 'insertFromDrop') return CHANGE_TYPE.PASTE;
      if (t.startsWith('delete'))                             return CHANGE_TYPE.DELETE;
      if (t === 'insertReplacementText')                      return CHANGE_TYPE.REPLACE;
      return CHANGE_TYPE.ADD;
    }
    const delta = current.length - prev.length;
    if (delta > 10) return CHANGE_TYPE.PASTE;
    if (delta < 0)  return CHANGE_TYPE.DELETE;
    if (delta === 0) return CHANGE_TYPE.REPLACE;
    return CHANGE_TYPE.ADD;
  }
}
