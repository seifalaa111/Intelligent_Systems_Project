# MIDAN UI

The thinking interface. Built from zero — not a website, not a chatbot.
The UI is a thin projection of the engine; the engine owns state, motion,
and timing. The DOM listens, projects, and responds.

## Layout

```
ui/
  index.html              entry + interior in one document, two stages
  styles/
    tokens.css            sensory tokens (colors, spacing, timing, easing, shadows)
    main.css              layout + components — every value resolves to a token
  scripts/
    app.js                orchestrator: boots engine, runs entry, mounts interior
    entry.js              entry sequence — hook → pain → tease → input → tension → CTA
    interior.js           interior view: top rail, signal band, output layers, context rail
    perception.js         engine-emitted PERCEPTION_SEQUENCE → DOM motion (no invented timing)
```

## Run locally

The UI is static ESM — no build step. Serve the project root (so
`../engine/index.js` resolves from `/ui/index.html`):

```bash
python -m http.server 3000
# open http://localhost:3000/ui/
```

Backend (FastAPI) on port 8001:

```bash
uvicorn api:app --port 8001
```

To point the UI at a different API:

```
http://localhost:3000/ui/?api=https://your-api.example.com
```

or set `window.MIDAN_API` before `<script src="./scripts/app.js">`.

## Architecture rule

The UI never decides:

- system state (engine StateMachine owns it)
- signal classification, rank, tier (engine signal layer owns it)
- confidence (engine ConfidenceEngine owns it)
- what to reveal next (engine OutputLayerController owns it)
- when to move (engine PerceptionSyncManager owns it)

The UI only:

- attaches an input element to the engine (`engine.attach(el)`)
- projects state changes to the DOM (`engine.onStateChange`, `engine.onOutputChange`)
- applies perception sequences to motion (`engine.onPerceptionSequence`)
- forwards user interactions back (`engine.editInterpretationComponent`,
  `engine.confirmAssumption`, `engine.rejectAssumption`, `engine.switchThread`)

If a behavior feels wrong, fix it in the engine, not here.

## Diagnostic console

Press `Ctrl+\`` (or `Cmd+\``) to toggle a live event tail at the bottom right.

## Reduced motion

`@media (prefers-reduced-motion: reduce)` collapses all transitions to ~0ms.
Engine timing is unchanged — only the DOM presentation flattens.
