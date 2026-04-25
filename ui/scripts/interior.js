/**
 * MIDAN UI — System interior.
 *
 * Mounts the interior view: top rail (input continuation), signal band,
 * 4 output layers, context rail. Treats the engine as the source of truth
 * for what is visible vs. pending vs. stale — never decides on its own.
 */

import { EVENT, SYSTEM_STATE, LAYER_ID, LAYER_STATUS } from '../../engine/index.js';

export function mountInterior(engine, { interiorInput }) {
  const signalChips    = document.getElementById('signal-band-chips');
  const signalCount    = document.getElementById('signal-band-count');
  const confBadge      = document.getElementById('confidence-badge');
  const railToggle     = document.getElementById('rail-toggle');
  const railSignals    = document.getElementById('rail-signals');
  const railAssump     = document.getElementById('rail-assumptions');
  const railOpen       = document.getElementById('rail-open-threads');
  const railThreads    = document.getElementById('rail-threads');
  const expandSignals  = document.getElementById('expand-signals');
  const reasoningWrap  = document.getElementById('reasoning');
  const reasoningTgl   = document.getElementById('toggle-reasoning');
  const deepenBtn      = document.getElementById('deepen');

  const layerEls = {
    [LAYER_ID.INTERPRETATION]:    document.querySelector('[data-layer-id="interpretation"]'),
    [LAYER_ID.SUMMARY]:           document.querySelector('[data-layer-id="summary"]'),
    [LAYER_ID.SIGNAL_EXPANSION]:  document.querySelector('[data-layer-id="signal_expansion"]'),
    [LAYER_ID.ACTION]:            document.querySelector('[data-layer-id="action"]'),
  };

  const chipNodes = new Map();

  function activate({ carry } = {}) {
    // Initial paint of state.
    paintAll(engine.getState());
    paintOutput(engine.getOutputState());
  }

  // ── Engine subscriptions ────────────────────────────────────────────────
  engine.onStateChange(paintAll);
  engine.onOutputChange(paintOutput);

  engine.on(EVENT.SIGNAL_DETECTED, ({ payload }) => {
    const sig = payload.signal;
    if (!sig || chipNodes.has(sig.id)) return;
    const node = makeChip(sig);
    signalChips.appendChild(node);
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

  engine.on(EVENT.CONFIDENCE_CHANGED, ({ payload }) => {
    const tier = payload.tier || payload.confidence_tier || 'LOW';
    confBadge.dataset.tier = tier;
    confBadge.textContent  = `${tier} CONFIDENCE`;
  });

  engine.on(EVENT.INTERPRETATION_UPDATED, ({ payload }) => {
    const components = payload?.components || Object.keys(payload?.interpretation || {});
    components.forEach((c, i) => {
      setTimeout(() => morphInterpRow(c), i * 80);
    });
  });

  // ── User interactions wired back to the engine ──────────────────────────
  document.querySelectorAll('.interp-value').forEach(el => {
    el.addEventListener('click', () => beginEditComponent(el));
  });

  expandSignals.addEventListener('click', () => {
    layerEls[LAYER_ID.SIGNAL_EXPANSION].scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  reasoningTgl.addEventListener('click', () => {
    const open = reasoningWrap.dataset.open === 'true';
    reasoningWrap.dataset.open = open ? 'false' : 'true';
  });
  deepenBtn.addEventListener('click', () => {
    const state = engine.getState();
    const action = state.output.conclusions?.action;
    const suggestion = action?.deeperPrompt || action?.value || '';
    if (suggestion) typeInto(interiorInput, suggestion);
  });

  railToggle.addEventListener('click', () => {
    const rail = document.getElementById('context-rail');
    const hidden = rail.style.display === 'none';
    rail.style.display = hidden ? '' : 'none';
  });

  // ── Painters ────────────────────────────────────────────────────────────
  function paintAll(state) {
    paintInterpretation(state.interpretation);
    paintRail(state);
    paintSignalBand(state);
  }

  function paintInterpretation(interp) {
    if (!interp) return;
    for (const comp of ['problem', 'market', 'mechanism', 'context']) {
      const row = document.querySelector(`.interp-row[data-component="${comp}"]`);
      if (!row) continue;
      const valEl = row.querySelector('.interp-value');
      const v = interp[comp]?.value;
      if (v && v.length) {
        valEl.textContent = v;
        valEl.dataset.empty = '0';
      } else {
        valEl.textContent = 'Not detected';
        valEl.dataset.empty = '1';
      }
    }
  }

  function paintSignalBand(state) {
    signalCount.textContent =
      state.signals.registry.length
        ? `${state.signals.registry.length} signals · ${state.signals.tier || ''}`
        : '';
  }

  function paintRail(state) {
    // Confirmed signals
    railSignals.innerHTML = '';
    state.signals.registry.forEach(s => {
      const li = document.createElement('li');
      li.textContent = labelFor(s);
      railSignals.appendChild(li);
      requestAnimationFrame(() => { li.dataset.visible = '1'; });
    });

    // Active assumptions
    railAssump.innerHTML = '';
    (state.assumptions?.active || []).forEach(a => {
      const li = document.createElement('li');
      li.dataset.kind = 'assumption';
      li.innerHTML = `
        <div>${escapeHtml(a.value || a.id)}</div>
        ${a.supports_conclusion ? `<div style="font-size:11px;color:var(--text-tertiary);margin-top:4px">supports: ${escapeHtml(a.supports_conclusion)}</div>` : ''}
        <div class="actions">
          <button data-act="confirm">Confirm</button>
          <button data-act="reject">Reject</button>
        </div>`;
      li.querySelector('[data-act="confirm"]').addEventListener('click', () => engine.confirmAssumption(a.id));
      li.querySelector('[data-act="reject"]').addEventListener('click', () => engine.rejectAssumption(a.id));
      railAssump.appendChild(li);
      requestAnimationFrame(() => { li.dataset.visible = '1'; });
    });

    // Open threads
    railOpen.innerHTML = '';
    (state.output?.open_threads || []).forEach(q => {
      const li = document.createElement('li');
      li.dataset.kind = 'open';
      li.textContent = q.question || q;
      li.addEventListener('click', () => typeInto(interiorInput, q.question || q));
      railOpen.appendChild(li);
      requestAnimationFrame(() => { li.dataset.visible = '1'; });
    });

    // Inactive threads
    railThreads.innerHTML = '';
    (engine.getInactiveThreads?.() || []).slice(0, 5).forEach(t => {
      const li = document.createElement('li');
      li.textContent = t.summary || t.id;
      li.addEventListener('click', () => engine.switchThread(t.id));
      railThreads.appendChild(li);
      requestAnimationFrame(() => { li.dataset.visible = '1'; });
    });
  }

  function paintOutput(out) {
    if (!out?.layers) return;
    Object.entries(out.layers).forEach(([id, info]) => {
      const el = layerEls[id];
      if (!el) return;
      el.dataset.status = info.status;

      if (id === LAYER_ID.SUMMARY && info.data?.summary) {
        document.getElementById('layer-summary').textContent = info.data.summary;
      }
      if (id === LAYER_ID.SIGNAL_EXPANSION && info.data?.expansions) {
        renderExpansions(info.data.expansions);
      }
      if (id === LAYER_ID.ACTION && info.data) {
        const body = document.getElementById('layer-action');
        body.textContent = info.data.action || info.data.value || '';
      }
    });

    // Reasoning content (Layer 3 lives inside signal expansion)
    const reasoning = out.layers[LAYER_ID.SIGNAL_EXPANSION]?.data?.reasoning;
    if (Array.isArray(reasoning)) renderReasoning(reasoning);
  }

  function renderExpansions(expansions) {
    const host = document.getElementById('layer-signal-expansion');
    host.innerHTML = '';
    expansions.forEach((ex, i) => {
      const wrap = document.createElement('div');
      wrap.className = 'signal-expand-item';
      wrap.dataset.open = '0';
      wrap.innerHTML = `
        <div class="signal-expand-item__head">
          <div class="signal-expand-item__title">${escapeHtml(ex.title || ex.signal || 'Signal')}</div>
          <span aria-hidden="true">▾</span>
        </div>
        <div class="signal-expand-item__body">
          <div class="signal-expand-item__row"><span class="label">Found</span><span>${escapeHtml(ex.found || '—')}</span></div>
          <div class="signal-expand-item__row"><span class="label">Why it matters</span><span>${escapeHtml(ex.why || '—')}</span></div>
          <div class="signal-expand-item__row"><span class="label">Confidence</span><span>${escapeHtml(ex.confidence || '—')}</span></div>
        </div>`;
      wrap.querySelector('.signal-expand-item__head').addEventListener('click', () => {
        // one-at-a-time
        host.querySelectorAll('.signal-expand-item').forEach(s => { if (s !== wrap) s.dataset.open = '0'; });
        wrap.dataset.open = wrap.dataset.open === '1' ? '0' : '1';
      });
      host.appendChild(wrap);
    });
  }

  function renderReasoning(rows) {
    reasoningWrap.innerHTML = '';
    rows.forEach(r => {
      const div = document.createElement('div');
      div.className = 'reasoning__row';
      div.textContent = typeof r === 'string' ? r : (r.text || JSON.stringify(r));
      reasoningWrap.appendChild(div);
    });
  }

  function morphInterpRow(component) {
    const row = document.querySelector(`.interp-row[data-component="${component}"]`);
    if (!row) return;
    row.dataset.morphing = '1';
    row.dataset.flashing = '1';
    setTimeout(() => { row.dataset.morphing = '0'; }, 240);
    setTimeout(() => { row.dataset.flashing = '0'; }, 400);
  }

  function beginEditComponent(valEl) {
    const row = valEl.closest('.interp-row');
    const component = row?.dataset.component;
    if (!component) return;
    const original = valEl.textContent.trim();
    valEl.contentEditable = 'true';
    valEl.focus();
    const range = document.createRange();
    range.selectNodeContents(valEl);
    document.getSelection().removeAllRanges();
    document.getSelection().addRange(range);

    const finish = () => {
      valEl.contentEditable = 'false';
      const next = valEl.textContent.trim();
      if (next && next !== original) engine.editInterpretationComponent(component, next);
    };
    valEl.addEventListener('blur', finish, { once: true });
    valEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter')  { e.preventDefault(); valEl.blur(); }
      if (e.key === 'Escape') { valEl.textContent = original; valEl.blur(); }
    }, { once: true });
  }

  return { activate };
}

// ── Helpers ────────────────────────────────────────────────────────────────

function makeChip(sig) {
  const el = document.createElement('div');
  el.className = 'chip';
  el.dataset.tier  = sig.tier || 'TIER1';
  el.dataset.rank  = sig.rank || 'TERTIARY';
  el.dataset.class = classOf(sig.type);
  el.dataset.id    = sig.id;
  el.textContent   = labelFor(sig);
  return el;
}

function classOf(type) {
  if (['problem','friction','structural_complexity','value_ambiguity','monetization_absent'].includes(type)) return 'RISK';
  if (['geographic'].includes(type)) return 'MARKET';
  if (['audience'].includes(type))   return 'AUDIENCE';
  return 'DOMAIN';
}

function labelFor(sig) {
  const v = sig.value || sig.type;
  switch (sig.type) {
    case 'geographic':            return `📍 ${v}`;
    case 'problem':               return '⚠️ Problem';
    case 'domain':                return `🧠 ${v}`;
    case 'audience':              return `👥 ${v}`;
    case 'friction':              return '⚠️ Friction';
    case 'structural_complexity': return '⚠️ Complexity';
    case 'value_ambiguity':       return '⚠️ Value unclear';
    case 'behavioral':            return '🧠 Repetition';
    case 'monetization_absent':   return '⚠️ No monetization';
    default:                      return v;
  }
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function typeInto(input, text) {
  input.focus();
  input.value = '';
  let i = 0;
  const total = 640;
  const step  = Math.max(20, Math.floor(total / Math.max(text.length, 1)));
  const timer = setInterval(() => {
    input.value = text.slice(0, ++i);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    if (i >= text.length) clearInterval(timer);
  }, step);
}
