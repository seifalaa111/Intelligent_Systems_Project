/**
 * MIDAN UI — App orchestrator.
 *
 * Wires the engine to the DOM. Responsibilities:
 *   - Boot the engine
 *   - Run the entry experience, then transition into the interior
 *   - Bind interior interactions back to the engine
 *   - Apply perception sequences to DOM (motion is owned by the engine; we only render)
 *
 * Architecture rule: this file does NOT decide system state, signal classification,
 * confidence, or what to reveal next. It listens, projects, and responds.
 */

import { MidanEngine, EVENT, SYSTEM_STATE, LAYER_ID, LAYER_STATUS, PERCEPTION_WEIGHT } from '../../engine/index.js';
import { runEntry }       from './entry.js';
import { mountInterior }  from './interior.js';
import { applySequence }  from './perception.js';

// ── Detect API base ───────────────────────────────────────────────────────────
// Default to localhost:8001 (matches start_servers.ps1). Override via
// `?api=https://...` or `window.MIDAN_API`.
const params = new URLSearchParams(location.search);
const API_BASE =
  params.get('api') ||
  window.MIDAN_API ||
  (location.hostname === 'localhost' ? 'http://localhost:8001' : '');

const engine = new MidanEngine({ apiBaseUrl: API_BASE });
window.__midan = engine; // for debugging

// ── DOM refs ──────────────────────────────────────────────────────────────────
const body          = document.body;
const entryInput    = document.getElementById('entry-input');
const interiorInput = document.getElementById('interior-input');
const cta           = document.getElementById('entry-cta');
const diag          = document.getElementById('diag');

// ── Diagnostic console (toggle with ` key) ───────────────────────────────────
const diagLog = [];
function logDiag(...args) {
  if (diag.hidden) return;
  const ts = new Date().toISOString().slice(11, 23);
  diagLog.push(`[${ts}] ` + args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' '));
  if (diagLog.length > 80) diagLog.shift();
  diag.textContent = diagLog.join('\n');
}
window.addEventListener('keydown', (e) => {
  if (e.key === '`' && (e.ctrlKey || e.metaKey)) { diag.hidden = !diag.hidden; }
});

// ── Cross-cutting state projection ────────────────────────────────────────────
// Keep <body> data-* in sync with system state so CSS can react globally.
engine.on(EVENT.STATE_TRANSITION, ({ payload }) => {
  body.dataset.systemState = payload.to;
  logDiag('state', payload.from, '→', payload.to, payload.trigger || '');
});
engine.on(EVENT.UNCERTAIN_ACTIVATED,   () => { body.dataset.uncertain = '1'; });
engine.on(EVENT.UNCERTAIN_DEACTIVATED, () => { body.dataset.uncertain = '0'; });

// Cursor parallax — drives ambient + teaser drift via CSS custom props.
let parallaxRaf = 0, mouseX = 0, mouseY = 0;
window.addEventListener('mousemove', (e) => {
  mouseX = (e.clientX / window.innerWidth)  - 0.5;
  mouseY = (e.clientY / window.innerHeight) - 0.5;
  if (!parallaxRaf) parallaxRaf = requestAnimationFrame(applyParallax);
});
function applyParallax() {
  parallaxRaf = 0;
  document.querySelectorAll('[data-parallax]').forEach(el => {
    const rate = parseFloat(el.dataset.parallax);
    const sign = el.classList.contains('teaser--b') ? -1 : 1;
    const rotX = parseFloat(getComputedStyle(el).getPropertyValue('--rot-x') || 0);
    const rotZ = parseFloat(getComputedStyle(el).getPropertyValue('--rot-z') || 0);
    const tx = mouseX * window.innerWidth  * rate * sign;
    const ty = mouseY * window.innerHeight * rate * sign;
    if (rotX || rotZ) {
      el.style.transform = `translate3d(${tx}px, ${ty}px, 0) rotateX(${rotX}deg) rotateZ(${rotZ}deg)`;
    } else {
      el.style.transform = `translate3d(${tx}px, ${ty}px, 0)`;
    }
  });
}

// ── Perception → DOM ──────────────────────────────────────────────────────────
// The engine emits timed sequences. We faithfully apply them; we do not invent timing.
engine.onPerceptionSequence((seq) => {
  logDiag('perception:seq', seq.id, seq.items.length, 'items', `${seq.totalDuration}ms`);
  applySequence(seq);
});
engine.onPerceptionAnticipate(({ for: target, window }) => {
  logDiag('perception:anticipate', target, `${window}ms`);
  // Element-specific "readying" state is applied by perception.js.
});
engine.onPerceptionAnticipateCancelled(({ for: target, reason }) => {
  logDiag('perception:anticipate-cancelled', target, reason);
});
engine.onPerceptionInterrupt((info) => {
  logDiag('perception:interrupt', info.sequenceId, `progress=${info.progress.toFixed(2)}`, info.reason);
});

// ── Boot: run the entry experience, then mount interior ──────────────────────
const interior = mountInterior(engine, { interiorInput });

// Entry orchestrates timing + signal-zone rendering during typing.
runEntry({
  engine,
  input: entryInput,
  cta,
  onTransition: () => {
    // The CTA was pressed. The input doesn't disappear — it grows into the rail.
    transitionToInterior();
  },
});

function transitionToInterior() {
  // Carry the entry text into the interior input — the user's intent is preserved.
  const carry = entryInput.value;
  interiorInput.value = carry;

  // Choreography per spec: deepen background → fade tension/teasers → expand input → compose interior.
  document.getElementById('entry-tension').dataset.visible = '0';
  document.getElementById('entry-teasers').dataset.hidden  = '1';

  setTimeout(() => {
    body.dataset.stage = 'interior';
    interior.activate({ carry });

    // Re-attach engine to the rail input — the engine continues observing.
    engine.detach();
    engine.attach(interiorInput);

    // Briefly focus so the user feels continuity without being yanked.
    setTimeout(() => interiorInput.focus(), 320);
  }, 480);
}

// Cleanup on unload (defensive — engine releases its own resources).
window.addEventListener('beforeunload', () => { try { engine.destroy(); } catch (_) {} });
