/**
 * MIDAN Engine — Master Constants
 * All values derived from spec. No ad hoc values permitted.
 */

// Rhythm grid: all animation durations must be multiples of 80ms
export const TIMING = Object.freeze({
  UNIT:  80,   // 1u
  U2:   160,
  U3:   240,
  U4:   320,
  U5:   400,   // INTERPRETING perception floor
  U6:   480,
  U7:   560,
  U8:   640,
  U10:  800,   // ANALYZING perception floor
  U15: 1200,   // ANALYZING perception ceiling
});

// Layer processing timings (spec-exact values)
export const LAYER_TIMINGS = Object.freeze({
  LAYER_2_NORMAL:       80,   // 1u debounce, normal state
  LAYER_2_PAUSED:      160,   // 2u debounce during PAUSED
  LAYER_3_TIER2_NORMAL: 180,  // spec-defined (between 2u-3u, not on grid — spec override)
  LAYER_3_TIER2_PAUSED: 260,  // spec-defined during PAUSED
  SIGNAL_STABLE:        160,  // 2u — registry must be stable before SP1
  INTERP_LOCK:          160,  // 2u — block stable before SP2
  ANALYSIS_TRIGGER:    1200,  // interpretation stable → ANALYZING
  LAYER1_TO_LAYER2_GAP: 640,  // L1→L2 inter-layer gap
  LAYER2_TO_LAYER4_GAP: 480,  // L2→L4 inter-layer gap
  UNCERTAINTY_RESTORE:  400,  // 5u — motion restores after first Tier 1 confirmed
});

export const SYSTEM_STATE = Object.freeze({
  IDLE:         'IDLE',
  LISTENING:    'LISTENING',
  RECEIVING:    'RECEIVING',
  INTERPRETING: 'INTERPRETING',
  ANALYZING:    'ANALYZING',
  REVEALING:    'REVEALING',
  WAITING:      'WAITING',
  PAUSED:       'PAUSED',
  REDIRECTED:   'REDIRECTED',
  COMPLETED:    'COMPLETED',
  UNCERTAIN:    'UNCERTAIN',
});

export const SIGNAL_TIER = Object.freeze({ TIER1: 'TIER1', TIER2: 'TIER2' });
export const SIGNAL_RANK = Object.freeze({ PRIMARY: 'PRIMARY', SECONDARY: 'SECONDARY', TERTIARY: 'TERTIARY' });
export const CONFIDENCE_TIER = Object.freeze({ HIGH: 'HIGH', MODERATE: 'MODERATE', LOW: 'LOW' });
export const INTERRUPTION_TYPE = Object.freeze({ MERGING: 'MERGING', DEEPENING: 'DEEPENING', CLOSING: 'CLOSING' });

export const INPUT_CLASS = Object.freeze({
  CASUAL:    'CASUAL',
  PARTIAL:   'PARTIAL',
  COMPLETE:  'COMPLETE',
  DIRECTIVE: 'DIRECTIVE',
  INLINE:    'INLINE',
});

export const CHANGE_TYPE = Object.freeze({
  ADD:     'add',
  DELETE:  'delete',
  PASTE:   'paste',
  REPLACE: 'replace',
});

export const INTERP_COMPONENT = Object.freeze({
  PROBLEM:   'problem',
  MARKET:    'market',
  MECHANISM: 'mechanism',
  CONTEXT:   'context',
  TRACTION:  'traction',
});

export const PIPELINE_STATUS = Object.freeze({
  IDLE:        'idle',
  RUNNING:     'running',
  INTERRUPTED: 'interrupted',
  COMPLETE:    'complete',
});

// Signal type identifiers (matches detector output)
export const SIGNAL_TYPE = Object.freeze({
  PROBLEM:               'problem',
  GEOGRAPHIC:            'geographic',
  DOMAIN:                'domain',
  AUDIENCE:              'audience',
  MONETIZATION_ABSENT:   'monetization_absent',
  FRICTION:              'friction',
  STRUCTURAL_COMPLEXITY: 'structural_complexity',
  VALUE_AMBIGUITY:       'value_ambiguity',
  BEHAVIORAL:            'behavioral',
});

// Maps signal type → which INTERP_COMPONENT it primarily populates
export const SIGNAL_TO_COMPONENT = Object.freeze({
  problem:               'problem',
  geographic:            'market',
  audience:              'market',
  domain:                'mechanism',
  friction:              'context',
  structural_complexity: 'context',
  value_ambiguity:       'mechanism',
  behavioral:            'context',
  monetization_absent:   'traction',
});

// Signal priority for rank hierarchy (lower = higher priority)
export const SIGNAL_PRIORITY_ORDER = Object.freeze({
  problem:               1,
  geographic:            2,
  friction:              2,
  domain:                3,
  audience:              3,
  structural_complexity: 3,
  monetization_absent:   4,
  value_ambiguity:       4,
  behavioral:            4,
});

// Completeness weights per signal TYPE category (not instance)
// Sum of all non-negative = 1.05 → capped at 1.0
export const COMPLETENESS_WEIGHT = Object.freeze({
  problem:               0.30,
  geographic:            0.25,
  domain:                0.15,
  audience:              0.12,
  friction:              0.10,
  structural_complexity: 0.08,
  value_ambiguity:       0.06,
  behavioral:            0.05,
  monetization_absent:  -0.05, // penalizes: >60 chars, no monetization language
});

// Confidence tier weights for compound rules
export const CONFIDENCE_WEIGHTS = Object.freeze({
  problem_absent_cap:    'LOW',   // Tier 1 absent → cap at LOW
  market_absent_cap:     'MODERATE', // Tier 2 absent → cap at MODERATE
  load_bearing_cap:      'LOW',   // any load-bearing assumption → that conclusion LOW
});

export const SYNC_POINT = Object.freeze({
  SP1: 'SP1_SIGNAL_STABILIZATION',
  SP2: 'SP2_INTERPRETATION_LOCK',
  SP3: 'SP3_ANALYSIS_START',
  SP4: 'SP4_REVEAL_START',
});

export const PERFORMANCE_BUDGET = Object.freeze({
  INTERPRETING_FLOOR:   400,  // 5u
  INTERPRETING_CEILING: 800,  // 10u
  ANALYZING_FLOOR:      800,  // 10u
  ANALYZING_CEILING:   1200,  // 15u
  HEARTBEAT_UNIT:        80,  // 1u between heartbeats
  MAX_HEARTBEATS:         5,
  SUSTAINED_LOAD_COUNT:   3,  // consecutive extended → raise baseline
  RECOVERY_COUNT:         5,  // consecutive on-time → lower baseline 1u
  FLOOR_DELTA:           80,  // 1u step for floor adjustment
  LATENCY_WINDOW:        10,  // rolling average window size
  MAX_FLOOR_EXTENSION:    5,  // extension_units cap
});

export const UNCERTAINTY = Object.freeze({
  LOW_COMPLETENESS:       0.30,
  LONG_NO_SIGNAL_CHARS:   60,
  MOTION_DURATION_SCALE:  1.4, // all durations × 1.4 during UNCERTAIN
  TRANSLATE_SCALE:        0.6, // all translates × 0.60 during UNCERTAIN
});

// Incremented when the thread snapshot schema changes (new fields, renamed keys).
// Restored threads with a different schema_version trigger partial recompute.
export const ENGINE_SCHEMA_VERSION = 1;

export const THREAD_COMPACTION_THRESHOLD = 50;
export const SYNC_MAX_DELAY_MS = 240; // 3u max before force-clear
export const REVERSAL_GUARD_MS = 200; // signal removed within 200ms → TRANSIENT

export const EVENT = Object.freeze({
  INPUT_RAW_UPDATED:              'INPUT_RAW_UPDATED',
  INTERPRETATION_UPDATED:         'INTERPRETATION_UPDATED',
  SIGNAL_DETECTED:                'SIGNAL_DETECTED',
  SIGNAL_REMOVED:                 'SIGNAL_REMOVED',
  SIGNAL_PROMOTED:                'SIGNAL_PROMOTED',
  SIGNAL_DEMOTED:                 'SIGNAL_DEMOTED',
  COMPLETENESS_UPDATED:           'COMPLETENESS_UPDATED',
  STATE_TRANSITION:               'STATE_TRANSITION',
  PIPELINE_STARTED:               'PIPELINE_STARTED',
  PIPELINE_CHECKPOINT:            'PIPELINE_CHECKPOINT',
  PIPELINE_INTERRUPTED:           'PIPELINE_INTERRUPTED',
  PIPELINE_COMPLETED:             'PIPELINE_COMPLETED',
  OUTPUT_LAYER_REVEALED:          'OUTPUT_LAYER_REVEALED',
  CONCLUSION_RETRACTED:           'CONCLUSION_RETRACTED',
  CONCLUSION_UPDATED:             'CONCLUSION_UPDATED',
  CONFIDENCE_CHANGED:             'CONFIDENCE_CHANGED',
  ASSUMPTION_CONFIRMED:           'ASSUMPTION_CONFIRMED',
  ASSUMPTION_REJECTED:            'ASSUMPTION_REJECTED',
  THREAD_CREATED:                 'THREAD_CREATED',
  THREAD_ARCHIVED:                'THREAD_ARCHIVED',
  THREAD_SWITCHED:                'THREAD_SWITCHED',
  HEARTBEAT:                      'HEARTBEAT',
  SYNC_POINT_CLEARED:             'SYNC_POINT_CLEARED',
  ANIMATION_REGISTERED:           'ANIMATION_REGISTERED',
  ANIMATION_COMPLETED:            'ANIMATION_COMPLETED',
  SESSION_STATE_UPDATED:          'SESSION_STATE_UPDATED',
  STALE_WRITE_REJECTED:           'STALE_WRITE_REJECTED',
  UNCERTAIN_ACTIVATED:            'UNCERTAIN_ACTIVATED',
  UNCERTAIN_DEACTIVATED:          'UNCERTAIN_DEACTIVATED',
  INTERRUPTION_CLASSIFIED:        'INTERRUPTION_CLASSIFIED',
  THREAD_SNAPSHOT_COMPACTED:      'THREAD_SNAPSHOT_COMPACTED',

  // ── Perception layer outputs ──────────────────────────────────────────────
  // Emitted by PerceptionSyncManager. Consumed by the UI layer.
  // These are NOT state events — they carry timing metadata for presentation.

  // A timed sequence of perception items is ready to begin.
  // Payload: { id, items, totalDuration, isInterrupt, redirectsFrom }
  PERCEPTION_SEQUENCE:            'PERCEPTION_SEQUENCE',

  // The system is about to make a meaningful reveal (advisory — not guaranteed).
  // Payload: { for, window }
  PERCEPTION_ANTICIPATE:          'PERCEPTION_ANTICIPATE',

  // A previously fired PERCEPTION_ANTICIPATE will not be followed by the anticipated event.
  // The UI must exit the "prepared" state it entered on ANTICIPATE.
  // Payload: { for, reason }
  PERCEPTION_ANTICIPATE_CANCELLED: 'PERCEPTION_ANTICIPATE_CANCELLED',

  // An active sequence was interrupted and motion should redirect gracefully.
  // Payload: { sequenceId, progress, reason, startedAt, totalDuration, interruptedAt }
  //   progress:      [0,1] — how far through the sequence at interrupt time
  //   startedAt:     ms timestamp — sequence start (for frame-accurate recalculation)
  //   totalDuration: ms — sequence's planned total duration
  //   interruptedAt: ms timestamp — exact moment of interruption
  PERCEPTION_INTERRUPT:           'PERCEPTION_INTERRUPT',
});

/**
 * Weight tiers for perception items — determines timing, duration, and visual prominence.
 *
 * PRIMARY:   Structural changes. New layer visible, analysis starts, thought redirected.
 *            Feel deliberate. Duration: 3u (240ms). Stagger: 1u between items.
 *
 * SECONDARY: Refinements. Interpretation updates, conclusions revising, confidence shifting.
 *            Feel responsive but not disruptive. Duration: 2u (160ms). No stagger.
 *
 * TERTIARY:  Micro adjustments. Signal weight, completeness ticks, background signals.
 *            Nearly invisible — support the narrative without drawing attention.
 *            Duration: 1u (80ms). No stagger.
 */
export const PERCEPTION_WEIGHT = Object.freeze({
  PRIMARY:   'PRIMARY',
  SECONDARY: 'SECONDARY',
  TERTIARY:  'TERTIARY',
});
