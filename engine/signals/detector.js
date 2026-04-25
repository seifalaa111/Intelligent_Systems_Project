/**
 * MIDAN Engine — SignalDetector
 * Pattern-based Tier 1 and Tier 2 signal detection.
 * Runs on raw text; returns detected signal descriptors.
 * Does NOT write to state — that is the layer's responsibility.
 */

import { SIGNAL_TYPE, SIGNAL_TIER } from '../core/constants.js';

// ── Tier 1 Patterns ──────────────────────────────────────────────────────────

const GEOGRAPHIC_TERMS = [
  'uae','dubai','sharjah','abu dhabi',
  'saudi','riyadh','jeddah','ksa',
  'egypt','cairo','alexandria',
  'africa','mena','gcc','kenya','nigeria','morocco',
  'qatar','kuwait','bahrain','oman','jordan','lebanon',
  'pakistan','bangladesh','india','indonesia',
];

const PROBLEM_TERMS = [
  "can't",'cannot','problem','issue','struggle',
  'hard','difficult','slow','expensive','costly',
  'broken','inefficient','manual','frustrating',
  'pain','failing','fails','broken','tedious','annoying',
  'waste','wasted','complex','complicated','confusing',
];

const DOMAIN_TERMS = {
  fintech:     ['fintech','payment','payments','banking','loan','lending','insurance','wallet','remittance'],
  healthtech:  ['health','medical','doctor','clinic','hospital','patient','diagnosis','healthcare','pharma'],
  edtech:      ['education','learning','school','course','training','tutor','tutoring','edtech','e-learning'],
  logistics:   ['logistics','shipping','delivery','warehouse','freight','supply chain','transport','fleet'],
  saas:        ['saas','software','platform','dashboard','api','b2b software','crm','erp','workflow'],
  marketplace: ['marketplace','marketplace','listing','gig','freelance','on-demand','two-sided'],
  agritech:    ['agritech','farm','farming','crop','agriculture','harvest','irrigation','livestock'],
  ecommerce:   ['e-commerce','ecommerce','online store','shop','retail','dropship','d2c','dtc'],
};

const AUDIENCE_TERMS = {
  founders:      ['founders','startups','startup','entrepreneurs'],
  students:      ['students','student','learners','graduates'],
  smes:          ['smes','small business','small businesses','merchants'],
  enterprise:    ['enterprise','corporate','large companies','b2b'],
  freelancers:   ['freelancers','freelancer','independents','contractors'],
  professionals: ['professionals','professionals','practitioners','experts'],
  developers:    ['developers','developer','engineers','engineering teams'],
  retailers:     ['retailers','retailer','merchants','sellers'],
  consumers:     ['consumers','users','customers','people','individuals','households'],
};

// Monetization keywords — absence triggers signal
const MONETIZATION_KEYWORDS = [
  'subscription','revenue','pricing','price','fee','fees','charge','charges',
  'pay','paid','monetize','business model','saas','commission','freemium',
  'premium','sell','sales','earn','income','profit',
];

// ── Tier 2 Patterns ──────────────────────────────────────────────────────────

const FRICTION_PATTERNS = [
  /takes? (too )?long/i,
  /slow(ly)?/i,
  /wait(ing)?/i,
  /manual(ly)?/i,
  /approval/i,
  /back and forth/i,
  /submit(ting)?/i,
  /follow.?up/i,
  /\b(is|are|gets?) (handled|done|managed) by hand\b/i,
];

// Structural complexity: ≥2 distinct actor types + ownership language absent + connectors
const STRUCTURAL_CONNECTORS = /\b(then|after that|which then|followed by|before|subsequently)\b/i;
const ACTOR_PATTERNS = [
  /\b(buyer|seller|vendor|supplier|provider|user|customer|client|agent|broker|driver|rider|operator|admin|manager|employee|employer|landlord|tenant|doctor|patient)\b/ig,
];

// Value ambiguity: ≥50 chars, no outcome language, only vague benefit words
const OUTCOME_LANGUAGE = /\b(so that|which means|this allows|this enables|resulting in|leading to|helping (them|users|people) to|in order to)\b/i;
const VAGUE_BENEFITS   = /\b(better|easier|faster|simpler|smoother|more efficient|improved|enhanced)\b/i;

// Behavioral: repetitive/forced behavior language
const BEHAVIORAL_PATTERNS = [
  /people (keep|always|every time|constantly|regularly)/i,
  /always (do|doing|happens|occurs)/i,
  /every time/i,
  /manually (do|doing|manage|track)/i,
  /forced to/i,
  /workaround/i,
  /hack(s)?/i,
  /\b(repeatedly|repetitive|routine)\b/i,
];

// Redirect language (used by InterruptionClassifier)
export const REDIRECT_LANGUAGE = [
  /actually,? (let'?s?|i want to|i('d| would) like to)/i,
  /forget (that|the previous|what i said)/i,
  /start(ing)? (over|fresh|new)/i,
  /different (idea|direction|concept)/i,
  /instead,? (let'?s?|i want)/i,
  /never mind/i,
  /scratch that/i,
];

// ── Detector ─────────────────────────────────────────────────────────────────

/**
 * Run Tier 1 detection on raw text (synchronous, word-boundary triggered).
 * @param {string} text
 * @returns {Array<SignalDescriptor>}
 */
export function detectTier1(text) {
  if (!text || typeof text !== 'string') return [];
  const lower   = text.toLowerCase();
  const results = [];

  // Problem
  const problemMatch = PROBLEM_TERMS.find(t => lower.includes(t));
  if (problemMatch) {
    results.push(_signal(SIGNAL_TYPE.PROBLEM, SIGNAL_TIER.TIER1, { trigger: problemMatch, label: 'Problem signal detected' }));
  }

  // Geographic / Market
  const geoMatch = GEOGRAPHIC_TERMS.find(t => lower.includes(t));
  if (geoMatch) {
    results.push(_signal(SIGNAL_TYPE.GEOGRAPHIC, SIGNAL_TIER.TIER1, { trigger: geoMatch, label: `${_titleCase(geoMatch)} market signal` }));
  }

  // Domain
  for (const [domain, terms] of Object.entries(DOMAIN_TERMS)) {
    const match = terms.find(t => lower.includes(t));
    if (match) {
      results.push(_signal(SIGNAL_TYPE.DOMAIN, SIGNAL_TIER.TIER1, { trigger: match, domain, label: `${_titleCase(domain)} pattern matched` }));
      break; // one domain signal at a time
    }
  }

  // Audience
  for (const [segment, terms] of Object.entries(AUDIENCE_TERMS)) {
    const match = terms.find(t => lower.includes(t));
    if (match) {
      results.push(_signal(SIGNAL_TYPE.AUDIENCE, SIGNAL_TIER.TIER1, { trigger: match, segment, label: `Audience signal: ${segment}` }));
      break;
    }
  }

  // Monetization absence (only fires after sufficient length)
  if (text.length > 60) {
    const hasMonetization = MONETIZATION_KEYWORDS.some(k => lower.includes(k));
    if (!hasMonetization) {
      results.push(_signal(SIGNAL_TYPE.MONETIZATION_ABSENT, SIGNAL_TIER.TIER1, { trigger: 'length_threshold', label: 'Monetization unclear' }));
    }
  }

  return results;
}

/**
 * Run Tier 2 detection (runs 180ms after word boundary).
 * @param {string} text
 * @returns {Array<SignalDescriptor>}
 */
export function detectTier2(text) {
  if (!text || typeof text !== 'string') return [];
  const results = [];

  // Friction
  const hasFriction = FRICTION_PATTERNS.some(p => p.test(text));
  if (hasFriction) {
    results.push(_signal(SIGNAL_TYPE.FRICTION, SIGNAL_TIER.TIER2, { label: 'Workflow friction detected' }));
  }

  // Structural complexity: ≥2 distinct actor types + "then/after" connectors
  const actors = new Set();
  ACTOR_PATTERNS.forEach(p => {
    const matches = text.matchAll(p);
    for (const m of matches) actors.add(m[0].toLowerCase());
  });
  const hasConnectors = STRUCTURAL_CONNECTORS.test(text);
  if (actors.size >= 2 && hasConnectors) {
    results.push(_signal(SIGNAL_TYPE.STRUCTURAL_COMPLEXITY, SIGNAL_TIER.TIER2, { label: 'System complexity signal', actorCount: actors.size }));
  }

  // Value ambiguity: ≥50 chars, no outcome language, only vague benefits
  if (text.length >= 50 && !OUTCOME_LANGUAGE.test(text) && VAGUE_BENEFITS.test(text)) {
    results.push(_signal(SIGNAL_TYPE.VALUE_AMBIGUITY, SIGNAL_TIER.TIER2, { label: 'Value clarity weak' }));
  }

  // Behavioral
  const hasBehavioral = BEHAVIORAL_PATTERNS.some(p => p.test(text));
  if (hasBehavioral) {
    results.push(_signal(SIGNAL_TYPE.BEHAVIORAL, SIGNAL_TIER.TIER2, { label: 'Repetitive behavior pattern detected' }));
  }

  return results;
}

/**
 * Detects if text contains an explicit redirect (CLOSING) signal.
 * @param {string} text
 * @returns {boolean}
 */
export function hasRedirectLanguage(text) {
  return REDIRECT_LANGUAGE.some(p => p.test(text));
}

/**
 * Extracts interpretation components from raw text (Layer 2 NLP extraction).
 * Returns only components that have changed vs. current interpretation.
 *
 * @param {string} text
 * @param {Object} currentInterp - current interpretation from SESSION_STATE
 * @returns {Object} partial interpretation update (only changed components)
 */
export function extractInterpretation(text, currentInterp) {
  const lower    = text.toLowerCase();
  const changes  = {};
  const now      = Date.now();

  // Problem component
  if (!currentInterp.problem?.user_override) {
    const problemKw = PROBLEM_TERMS.find(t => lower.includes(t));
    const problemSentence = problemKw ? _extractSentenceAround(text, problemKw) : null;
    if (problemSentence !== currentInterp.problem?.value) {
      changes.problem = { value: problemSentence, confidence: problemSentence ? 'MODERATE' : null, version: now, user_override: false };
    }
  }

  // Market component (geographic + audience)
  if (!currentInterp.market?.user_override) {
    const geoKw   = GEOGRAPHIC_TERMS.find(t => lower.includes(t));
    const audMatch = Object.values(AUDIENCE_TERMS).flat().find(t => lower.includes(t));
    const marketVal = [geoKw, audMatch].filter(Boolean).join(' · ') || null;
    if (marketVal !== currentInterp.market?.value) {
      changes.market = { value: marketVal, confidence: marketVal ? 'MODERATE' : null, version: now, user_override: false };
    }
  }

  // Mechanism component (domain signals + solution language)
  if (!currentInterp.mechanism?.user_override) {
    let mechVal = null;
    for (const [domain, terms] of Object.entries(DOMAIN_TERMS)) {
      if (terms.some(t => lower.includes(t))) {
        mechVal = domain;
        break;
      }
    }
    if (mechVal !== currentInterp.mechanism?.value) {
      changes.mechanism = { value: mechVal, confidence: mechVal ? 'LOW' : null, version: now, user_override: false };
    }
  }

  // Context component (temporal/situational language)
  if (!currentInterp.context?.user_override) {
    const ctxKw = _findContextKeyword(lower);
    if (ctxKw !== currentInterp.context?.value) {
      changes.context = { value: ctxKw, confidence: ctxKw ? 'LOW' : null, version: now, user_override: false };
    }
  }

  // Traction component
  if (!currentInterp.traction?.user_override) {
    const tractionVal = _extractTraction(lower);
    if (tractionVal !== currentInterp.traction?.value) {
      changes.traction = { value: tractionVal, confidence: tractionVal ? 'LOW' : null, version: now, user_override: false };
    }
  }

  return changes;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function _signal(type, tier, meta = {}) {
  return {
    id:       `${type}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    type,
    tier,
    added_at: Date.now(),
    ...meta,
  };
}

function _titleCase(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function _extractSentenceAround(text, keyword) {
  const idx = text.toLowerCase().indexOf(keyword);
  if (idx === -1) return null;
  const start = Math.max(0, text.lastIndexOf('.', idx) + 1);
  const end   = text.indexOf('.', idx);
  return text.slice(start, end === -1 ? text.length : end + 1).trim();
}

function _findContextKeyword(lower) {
  const contextKw = ['currently','now','today','existing','at the moment','right now','at present'];
  return contextKw.find(k => lower.includes(k)) ?? null;
}

function _extractTraction(lower) {
  const tractionKw = ['users','customers','revenue','validated','traction','pilots','beta','sign-ups','signups','downloads','paying'];
  return tractionKw.find(k => lower.includes(k)) ?? null;
}
