/**
 * MIDAN Engine — InterruptionClassifier
 * Classifies new input after PAUSED into:
 *   MERGING   — continuation, shares ≥2 signal types OR adds detail without contradiction
 *   DEEPENING — targets specific element
 *   CLOSING   — new core entity OR explicit redirect language
 *
 * Used to determine how to resume the pipeline and whether to archive the thread.
 */

import { INTERRUPTION_TYPE, EVENT } from '../core/constants.js';
import { hasRedirectLanguage, detectTier1, detectTier2 } from '../signals/detector.js';

export class InterruptionClassifier {
  #eventBus;

  constructor(eventBus) {
    this.#eventBus = eventBus;
  }

  /**
   * Classify the interruption based on current state and new input.
   *
   * @param {Object} sessionState  - SESSION_STATE at interruption moment
   * @param {string} newInput      - new text entered by user
   * @returns {{ type: string, newSignals: Object, targetComponent: string|null }}
   */
  classify(sessionState, newInput) {
    const currentRegistry  = sessionState.signals.registry;
    const currentTypes     = new Set(currentRegistry.map(s => s.type));

    const newTier1  = detectTier1(newInput);
    const newTier2  = detectTier2(newInput);
    const allNew    = [...newTier1, ...newTier2];
    const newTypes  = new Set(allNew.map(s => s.type));

    // Rule: CLOSING — new core entity OR explicit redirect language
    if (hasRedirectLanguage(newInput)) {
      return this._emit(INTERRUPTION_TYPE.CLOSING, { newTier1, newTier2, targetComponent: null });
    }

    // CLOSING: new types that don't overlap with any current type (entirely new topic)
    const overlap = [...newTypes].filter(t => currentTypes.has(t));
    const isEntirelyNew = allNew.length > 0 && overlap.length === 0;
    if (isEntirelyNew && newTier1.some(s => s.type === 'geographic' || s.type === 'domain')) {
      return this._emit(INTERRUPTION_TYPE.CLOSING, { newTier1, newTier2, targetComponent: null });
    }

    // Rule: DEEPENING — new input targets a specific component
    const targetComponent = this._detectTargetComponent(newInput, sessionState);
    if (targetComponent && overlap.length <= 1) {
      return this._emit(INTERRUPTION_TYPE.DEEPENING, { newTier1, newTier2, targetComponent });
    }

    // Rule: MERGING — shares ≥2 signal types OR adds detail without contradiction
    const sharedTypes = overlap.length;
    if (sharedTypes >= 2 || (sharedTypes >= 1 && allNew.length > 0)) {
      return this._emit(INTERRUPTION_TYPE.MERGING, {
        newTier1,
        newTier2,
        targetComponent: null,
        newPrimary: newTier1.some(s => s.type === 'problem'),
        newTier1:   newTier1.length > 0,
        newTier2:   newTier2.length > 0,
      });
    }

    // Default to MERGING for any continuation input
    return this._emit(INTERRUPTION_TYPE.MERGING, { newTier1, newTier2, targetComponent: null });
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  _detectTargetComponent(text, state) {
    const lower = text.toLowerCase();

    // Specific element targeting patterns
    const componentHints = {
      problem:   ['the problem', 'the issue', 'actually the pain', 'what i mean by'],
      market:    ['target market', 'target audience', 'the market', 'customer is', 'targeting'],
      mechanism: ['the solution', 'how it works', 'the platform', 'the product', 'mechanism'],
      traction:  ['we have', 'already', 'customers so far', 'revenue'],
    };

    for (const [component, hints] of Object.entries(componentHints)) {
      if (hints.some(h => lower.includes(h))) return component;
    }

    return null;
  }

  _emit(type, meta) {
    this.#eventBus.emit(EVENT.INTERRUPTION_CLASSIFIED, { type, ...meta });
    return { type, ...meta };
  }
}
