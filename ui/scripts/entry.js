/**
 * MIDAN UI — Entry experience.
 *
 * Six states. Spec timings in ms. The entry exists to make the user feel
 * the system was already running when they arrived. Every delay below
 * is intentional — do not collapse them without re-reading the entry spec.
 */

import { EVENT, SYSTEM_STATE } from '../../engine/index.js';

const HOOK_LINE = 'Still guessing if your idea is worth building?';

const PAIN_LINES = [
  'Weeks of research… still unsure?',
  'Building first, validating later?',
  "Spending time on ideas that don't hold?",
];

export function runEntry({ engine, input, cta, onTransition }) {
  const hookEl     = document.getElementById('entry-hook');
  const painEls    = [...document.querySelectorAll('.entry__pain-line')];
  const teaserWrap = document.getElementById('entry-teasers');
  const inputWrap  = document.getElementById('entry-input-wrap');
  const tensionEl  = document.getElementById('entry-tension');

  hookEl.textContent = HOOK_LINE;
  painEls[0].textContent = PAIN_LINES[0];
  painEls[1].textContent = PAIN_LINES[1];
  painEls[2].textContent = PAIN_LINES[2];

  // STATE 0 → STATE 1: hook enters
  setTimeout(() => { hookEl.dataset.visible = '1'; }, 600);

  // STATE 2: hook reduces, pain lines stagger in
  setTimeout(() => {
    hookEl.dataset.mode = 'reduced';
    painEls[0].dataset.visible = '1';
  }, 2000);
  setTimeout(() => {
    painEls[0].dataset.dimmed = '1';
    painEls[1].dataset.visible = '1';
  }, 3200);
  setTimeout(() => {
    painEls[1].dataset.dimmed = '1';
    painEls[2].dataset.visible = '1';
  }, 4400);

  // STATE 3: pain dims out, hook minimizes, teasers float in
  setTimeout(() => {
    painEls.forEach(p => { p.dataset.visible = '0'; });
    hookEl.dataset.mode = 'small';
    teaserWrap.querySelectorAll('.teaser').forEach(t => { t.dataset.visible = '1'; });
  }, 5400);

  // STATE 4: input rises from center
  setTimeout(() => {
    teaserWrap.dataset.dim = '1';
    inputWrap.dataset.visible = '1';
    input.focus();
    engine.attach(input);
  }, 8200);

  // ── Live signal projection during typing ─────────────────────────────────
  const signalZone = document.getElementById('signal-zone-entry');
  const chipNodes  = new Map(); // signalId → element

  engine.on(EVENT.SIGNAL_DETECTED, ({ payload }) => {
    const sig = payload.signal;
    if (!sig || chipNodes.has(sig.id)) return;
    const node = makeChip(sig);
    signalZone.appendChild(node);
    requestAnimationFrame(() => { node.dataset.visible = '1'; });
    chipNodes.set(sig.id, node);
  });

  engine.on(EVENT.SIGNAL_REMOVED, ({ payload }) => {
    const id   = payload?.id || payload?.type;
    const node = chipNodes.get(id);
    if (!node) return;
    node.dataset.visible = '0';
    setTimeout(() => node.remove(), 240);
    chipNodes.delete(id);
  });

  engine.on(EVENT.SIGNAL_PROMOTED, ({ payload }) => {
    const node = chipNodes.get(payload?.id);
    if (node) {
      node.dataset.rank  = payload.toRank;
      node.dataset.pulse = '1';
      setTimeout(() => { node.dataset.pulse = '0'; }, 380);
    }
  });

  // Caret + shadow tier glow follow completeness.
  engine.on(EVENT.COMPLETENESS_UPDATED, ({ payload }) => {
    const c = payload.completeness;
    const tier = c >= 0.65 ? '3' : c >= 0.40 ? '2' : c > 0 ? '1' : '0';
    input.dataset.glow = tier;
    if (tier === '3' && !tensionEl.dataset.armed) {
      armTension();
    }
  });

  // ── Pre-CTA tension (after signals stable) ────────────────────────────────
  let tensionTimer = null;
  function armTension() {
    tensionEl.dataset.armed = '1';
    clearTimeout(tensionTimer);
    tensionEl.dataset.visible = '1';
    tensionEl.dataset.step    = '1';
    tensionTimer = setTimeout(() => { tensionEl.dataset.step = '2'; }, 1600);
    tensionTimer = setTimeout(() => { tensionEl.dataset.step = '3'; }, 2400);
  }

  cta.addEventListener('click', () => {
    onTransition?.();
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && input.value.trim().length > 0) {
      e.preventDefault();
      onTransition?.();
    }
  });
}

// ── Helpers ─────────────────────────────────────────────────────────────────

const ICON = {
  problem:               '⚠️',
  geographic:            '📍',
  domain:                '🧠',
  audience:              '👥',
  friction:              '⚠️',
  structural_complexity: '⚠️',
  value_ambiguity:       '⚠️',
  behavioral:            '🧠',
  monetization_absent:   '⚠️',
};

const CLASS_BY_TYPE = {
  problem:               'PROBLEM',
  friction:              'RISK',
  structural_complexity: 'RISK',
  value_ambiguity:       'RISK',
  monetization_absent:   'RISK',
  geographic:            'MARKET',
  audience:              'AUDIENCE',
  domain:                'DOMAIN',
  behavioral:            'DOMAIN',
};

function labelFor(sig) {
  const v = sig.value || sig.type;
  switch (sig.type) {
    case 'geographic':            return `${v} market signal`;
    case 'problem':               return 'Problem signal detected';
    case 'domain':                return `${v} pattern matched`;
    case 'audience':              return `Audience: ${v}`;
    case 'friction':              return 'Workflow friction';
    case 'structural_complexity': return 'System complexity';
    case 'value_ambiguity':       return 'Value clarity weak';
    case 'behavioral':            return 'Repetitive behavior';
    case 'monetization_absent':   return 'Monetization unclear';
    default:                      return v;
  }
}

function makeChip(sig) {
  const el = document.createElement('div');
  el.className = 'chip';
  el.dataset.tier  = sig.tier || 'TIER1';
  el.dataset.rank  = sig.rank || 'TERTIARY';
  el.dataset.class = CLASS_BY_TYPE[sig.type] || 'DOMAIN';
  el.dataset.id    = sig.id;
  el.textContent   = `${ICON[sig.type] || ''} ${labelFor(sig)}`.trim();
  return el;
}
