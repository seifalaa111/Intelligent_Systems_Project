/**
 * MIDAN UI — Perception renderer.
 *
 * Translates engine-emitted PERCEPTION_SEQUENCE items into DOM motion.
 * The engine owns timing decisions (rhythm grid, weight, stagger, anticipation,
 * interruption) — this module is intentionally dumb: it schedules a class
 * change at `delay` ms from "now" and removes it after `duration` ms.
 *
 * Targeting convention (from PerceptionSyncManager):
 *   `state:<lowercase_state>`   → e.g. 'state:analyzing'
 *   `layer:<layer_id>`          → e.g. 'layer:summary'
 *   `signal:<id>`               → individual signal chip
 *   `assumption:<id>`           → assumption card
 *   `conclusion:<id>`           → output conclusion
 *   `confidence`                → confidence badge
 *   `pipeline`                  → top-rail level pulse
 */

let activeSequence = null;
let scheduledTimers = [];

export function applySequence(seq) {
  // Cancel any already-running schedule from a previous sequence: the engine
  // guarantees the new sequence supersedes it (interrupt/redirect-aware).
  cancelActive();
  activeSequence = seq.id;

  for (const item of seq.items) {
    const t = setTimeout(() => apply(item), Math.max(0, item.delay || 0));
    scheduledTimers.push(t);
  }
}

export function cancelActive() {
  scheduledTimers.forEach(clearTimeout);
  scheduledTimers = [];
  activeSequence  = null;
}

function apply(item) {
  const el = resolveTarget(item.targetId);
  if (!el) return;

  const dur = item.duration || 240;

  switch (item.change) {
    case 'reveal':
    case 'show':
      // Status is the engine's contract; don't override layer status when it's already PENDING/VISIBLE.
      if (el.dataset.status === 'hidden' || el.dataset.status === 'stale') {
        el.dataset.status = 'visible';
      }
      el.dataset.weight = item.weight;
      pulse(el, 'reveal', dur);
      break;

    case 'update':
    case 'updating':
      el.dataset.status = 'updating';
      pulse(el, 'updating', dur);
      break;

    case 'retract':
    case 'stale':
      el.dataset.status = 'stale';
      pulse(el, 'retract', dur);
      break;

    case 'pulse':
      pulse(el, 'pulse', dur);
      break;

    case 'demote':
    case 'promote':
      el.dataset.pulse = '1';
      setTimeout(() => { el.dataset.pulse = '0'; }, dur);
      break;

    case 'state':
      // Body-level state hint — already painted by app.js on STATE_TRANSITION.
      // We add a transient data flag for state-specific micro motion.
      document.body.dataset.transitioning = item.meta?.to || '';
      setTimeout(() => { document.body.dataset.transitioning = ''; }, dur);
      break;

    default:
      pulse(el, 'generic', dur);
  }
}

function pulse(el, kind, dur) {
  el.dataset.perceptKind = kind;
  setTimeout(() => {
    if (el.dataset.perceptKind === kind) el.dataset.perceptKind = '';
  }, dur);
}

function resolveTarget(targetId) {
  if (!targetId) return null;
  const [kind, id] = targetId.split(':');
  switch (kind) {
    case 'layer':       return document.querySelector(`[data-layer-id="${id}"]`);
    case 'state':       return document.body;
    case 'signal':      return document.querySelector(`.chip[data-id="${id}"]`);
    case 'assumption':  return document.querySelector(`[data-kind="assumption"][data-id="${id}"]`);
    case 'conclusion':  return document.querySelector(`[data-conclusion-id="${id}"]`);
    case 'confidence':  return document.getElementById('confidence-badge');
    case 'pipeline':    return document.getElementById('top-rail');
    default:            return document.querySelector(`[data-target="${targetId}"]`);
  }
}
