/**
 * MIDAN Engine — IntentClassifier
 * Frontend-side pre-classification before API call.
 * Mirrors the backend's intent classification to minimize round trips.
 * Does NOT recreate backend ML logic — only pattern-based routing.
 *
 * Classes (in priority order, matching backend):
 *   DIRECTIVE   — explicit command, bypasses classification
 *   CLARIFICATION — post-analysis follow-up
 *   CASUAL      — greeting, meta, no idea content
 *   ANALYSIS_REQUEST — sufficient signals
 *   PARTIAL_IDEA — some signals, missing components
 */

import { INPUT_CLASS } from '../core/constants.js';
import { isReady } from '../signals/completeness.js';

// Directive commands (must bypass all checks)
const DIRECTIVE_PATTERNS = [
  /^(analyze|run|evaluate|assess|go|check|scan|look at|tell me|what do you think|rate|score)\b/i,
  /\b(analyze now|run it|run analysis|evaluate this|give me results|just analyze)\b/i,
  /^!/, // starts with ! (command prefix)
];

// Casual inputs (greetings, meta, off-topic)
const CASUAL_PATTERNS = [
  /^(hi|hello|hey|sup|what'?s up|greetings|good morning|good evening)\b/i,
  /^(how are you|who are you|what are you|what is midan|what can you do)\b/i,
  /^(thanks|thank you|ok|okay|sure|got it|nice|cool|great)\b/i,
  /^\s*$/, // empty
];

// Language indicating explicit redirect
const REDIRECT_PATTERNS = [
  /\b(different idea|new idea|forget (that|the previous)|start over|scratch that|never mind)\b/i,
];

export class IntentClassifier {
  /**
   * Classify the latest user message.
   *
   * @param {string}  message      - latest user message text
   * @param {Object}  sessionState - current SESSION_STATE
   * @returns {{ class: string, confidence: number, meta: Object }}
   */
  classify(message, sessionState) {
    const text     = message?.trim() ?? '';
    const registry = sessionState.signals.registry;
    const hasAnalysis = Object.keys(sessionState.output.conclusions).length > 0;

    // Priority 1: DIRECTIVE
    if (DIRECTIVE_PATTERNS.some(p => p.test(text))) {
      return { class: INPUT_CLASS.DIRECTIVE, confidence: 0.95, meta: {} };
    }

    // Priority 2: INLINE (chip click, component edit — handled externally, not via text)
    // Not applicable here — inline inputs bypass this classifier

    // Priority 3: Post-analysis follow-up → CLARIFICATION (routes to /chat)
    if (hasAnalysis && sessionState.pipeline_state.status === 'complete') {
      return {
        class: INPUT_CLASS.COMPLETE, // will be routed to /chat by API adapter
        confidence: 0.85,
        meta: { isPostAnalysis: true },
      };
    }

    // Priority 4: CASUAL
    if (CASUAL_PATTERNS.some(p => p.test(text))) {
      return { class: INPUT_CLASS.CASUAL, confidence: 0.90, meta: {} };
    }

    // Priority 5: REDIRECT (explicit redirect language)
    if (REDIRECT_PATTERNS.some(p => p.test(text))) {
      return { class: INPUT_CLASS.DIRECTIVE, confidence: 0.80, meta: { isRedirect: true } };
    }

    // Priority 6: ANALYSIS_REQUEST — sufficient signal (ready)
    if (isReady(registry)) {
      return { class: INPUT_CLASS.COMPLETE, confidence: 0.80, meta: {} };
    }

    // Priority 7: PARTIAL_IDEA — some content but insufficient signal
    if (text.length > 10 && !CASUAL_PATTERNS.some(p => p.test(text))) {
      return { class: INPUT_CLASS.PARTIAL, confidence: 0.70, meta: {} };
    }

    return { class: INPUT_CLASS.CASUAL, confidence: 0.60, meta: {} };
  }
}
