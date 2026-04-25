/**
 * MIDAN Engine — Public API
 * MidanEngine: the execution layer between existing backend intelligence and future UI.
 *
 * Usage:
 *   const engine = new MidanEngine({ apiBaseUrl: 'http://localhost:8000' });
 *   engine.attach(inputElement);
 *   engine.on('state', (state) => renderUI(state));
 *   engine.on(EVENT.HEARTBEAT, ({ payload }) => doHeartbeat(payload));
 *   engine.submit();
 */

// Core
import { EventBus }             from './core/event-bus.js';
import { SessionStore }         from './core/store.js';
import { StateReducer }         from './core/reducer.js';
import { StateMachine }         from './core/state-machine.js';
import { SYSTEM_STATE, EVENT, INTERRUPTION_TYPE, INPUT_CLASS, SYNC_POINT } from './core/constants.js';

// Layers
import { RawInputLayer }        from './layers/layer-1.js';
import { InterpretedStructureLayer } from './layers/layer-2.js';
import { SignalLayer }           from './layers/layer-3.js';

// Analysis
import { AnalysisPipeline }     from './analysis/pipeline.js';
import { RecomputationEngine }  from './analysis/recomputation.js';
import { ConfidenceEngine }     from './analysis/confidence.js';
import { syncCompleteness }     from './signals/completeness.js';

// Threads
import { ThreadManager }        from './threads/manager.js';

// Adapters
import { ApiAdapter }           from './adapters/api-adapter.js';
import { IntentClassifier }     from './adapters/intent-classifier.js';
import { InterruptionClassifier } from './adapters/interruption.js';

// Sync
import { PhaseAnimationRegistry } from './sync/animation-registry.js';
import { PerformanceSyncManager } from './sync/performance.js';
import { SyncPointGate }        from './sync/sync-points.js';

// Output
import { OutputLayerController, LAYER_ID, LAYER_STATUS } from './output/layer-controller.js';

// Perception
import { PerceptionSyncManager } from './perception/perception-sync.js';
import { PERCEPTION_WEIGHT }     from './core/constants.js';

// Re-export constants for consumers
export { EVENT, SYSTEM_STATE, INTERRUPTION_TYPE, INPUT_CLASS, SYNC_POINT, LAYER_ID, LAYER_STATUS, PERCEPTION_WEIGHT };

export class MidanEngine {
  // Core
  #eventBus;
  #store;
  #reducer;
  #stateMachine;

  // Layers
  #layer1;
  #layer2;
  #layer3;

  // Analysis
  #pipeline;
  #recomputation;
  #confidence;

  // Threads
  #threadManager;

  // Adapters
  #apiAdapter;
  #intentClassifier;
  #interruptionClassifier;

  // Sync
  #animationRegistry;
  #performanceSync;
  #syncGate;

  // Output
  #outputLayerController;

  // Perception
  #perceptionSync;

  // State
  #inputElement;
  #boundHandlers;
  #isDestroyed;
  #analysisRun;   // { id: Symbol, abortController: AbortController } | null

  /**
   * @param {Object} config
   * @param {string} config.apiBaseUrl - backend base URL, e.g. 'http://localhost:8000'
   */
  constructor({ apiBaseUrl = '' } = {}) {
    this.#isDestroyed = false;
    this.#analysisRun = null;

    // Instantiation order matters: eventBus → store → everything else
    this.#eventBus  = new EventBus(() => this.#store?.getState().version ?? 0);
    this.#store     = new SessionStore(this.#eventBus);

    // Adapters (before layers — layers may need adapter reference via engine)
    this.#apiAdapter            = new ApiAdapter(this.#store, this.#eventBus, apiBaseUrl);
    this.#intentClassifier      = new IntentClassifier();
    this.#interruptionClassifier = new InterruptionClassifier(this.#eventBus);

    // Sync (before pipeline — pipeline references performanceSync)
    this.#animationRegistry = new PhaseAnimationRegistry(this.#eventBus);
    this.#performanceSync   = new PerformanceSyncManager(this.#store, this.#eventBus);
    this.#syncGate          = new SyncPointGate(this.#animationRegistry, this.#eventBus);

    // Layers — receive read-only getState function, never direct store reference.
    // This enforces Fix 3: layers are emit-only; StateReducer owns all store writes.
    const getState = () => this.#store.getState();
    this.#layer1 = new RawInputLayer(getState, this.#eventBus);
    this.#layer2 = new InterpretedStructureLayer(getState, this.#eventBus);
    this.#layer3 = new SignalLayer(getState, this.#eventBus);

    // StateReducer — the ONLY entity that calls store.update() in response to layer events.
    // Must be instantiated after eventBus and store, before state machine.
    this.#reducer = new StateReducer(this.#store, this.#eventBus);

    // Analysis
    this.#confidence    = new ConfidenceEngine(this.#store, this.#eventBus);
    this.#recomputation = new RecomputationEngine(this.#store, this.#eventBus, this.#apiAdapter);
    this.#pipeline      = new AnalysisPipeline(
      this.#store, this.#eventBus, this.#apiAdapter, this.#performanceSync,
    );

    // Threads
    this.#threadManager = new ThreadManager(this.#store, this.#eventBus);

    // Output — must come after pipeline (depends on PIPELINE_* events) and store
    this.#outputLayerController = new OutputLayerController(this.#store, this.#eventBus);

    // Perception — sits on top of all system layers, translates events into
    // timed sequences for the UI layer. Must come after OutputLayerController
    // so OUTPUT_LAYER_REVEALED is already in flight when perception processes it.
    this.#perceptionSync = new PerceptionSyncManager(this.#store, this.#eventBus);

    // State machine (wires up cross-layer event reactions)
    this.#stateMachine = new StateMachine(this.#store, this.#eventBus);

    this.#boundHandlers = {};
    this._bindEngineWiring();
  }

  // ── Public: Attach to DOM ─────────────────────────────────────────────────

  /**
   * Bind the engine to an input element (textarea or input[type=text]).
   * The engine drives all three layers from this element's events.
   *
   * @param {HTMLInputElement|HTMLTextAreaElement} inputElement
   */
  attach(inputElement) {
    this.detach(); // clean up previous binding if any

    this.#inputElement = inputElement;

    const onFocus = () => {
      if (this.#stateMachine.current === SYSTEM_STATE.IDLE) {
        this.#stateMachine.transition(SYSTEM_STATE.LISTENING, { trigger: 'focus' });
      }
    };

    const onBlur = () => {
      if (this.#stateMachine.current === SYSTEM_STATE.LISTENING) {
        this.#stateMachine.transition(SYSTEM_STATE.IDLE, { trigger: 'blur' });
      }
    };

    const onInput = (e) => {
      const value = inputElement.value;
      // Layer 1: immediate, no debounce
      this.#layer1.process(value, e);
      // Layer 2: debounced
      this.#layer2.schedule();
      // Layer 3: word-boundary + timer
      this.#layer3.process(value);
    };

    const onKeydown = (e) => {
      // Enter (without Shift) = submit
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.submit();
      }
    };

    inputElement.addEventListener('focus',  onFocus);
    inputElement.addEventListener('blur',   onBlur);
    inputElement.addEventListener('input',  onInput);
    inputElement.addEventListener('keydown', onKeydown);

    this.#boundHandlers = { onFocus, onBlur, onInput, onKeydown };
  }

  /** Remove DOM event bindings. */
  detach() {
    if (this.#inputElement && this.#boundHandlers.onInput) {
      const el = this.#inputElement;
      el.removeEventListener('focus',   this.#boundHandlers.onFocus);
      el.removeEventListener('blur',    this.#boundHandlers.onBlur);
      el.removeEventListener('input',   this.#boundHandlers.onInput);
      el.removeEventListener('keydown', this.#boundHandlers.onKeydown);
    }
    this.#inputElement  = null;
    this.#boundHandlers = {};
  }

  // ── Public: Submit ────────────────────────────────────────────────────────

  /**
   * Trigger analysis. Classifies intent and routes accordingly.
   * Called on Enter or CTA button press.
   */
  async submit() {
    if (this.#isDestroyed) return;

    const state   = this.#store.getState();
    const message = state.input.raw;

    if (!message.trim()) return;

    const intent  = this.#intentClassifier.classify(message, state);
    const current = this.#stateMachine.current;

    // Handle PAUSED + new submit → classify interruption
    if (current === SYSTEM_STATE.PAUSED) {
      return this._handlePausedSubmit(message, intent);
    }

    // Route by intent
    switch (intent.class) {
      case INPUT_CLASS.DIRECTIVE:
        if (intent.meta?.isRedirect) {
          return this._handleRedirect(message);
        }
        this._forceAnalyzing(current);
        return this._runAnalysis({ inputClass: INPUT_CLASS.DIRECTIVE });

      case INPUT_CLASS.COMPLETE:
        if (intent.meta?.isPostAnalysis) {
          return this._handlePostAnalysisChat(message);
        }
        this._forceAnalyzing(current);
        return this._runAnalysis({ inputClass: INPUT_CLASS.COMPLETE });

      case INPUT_CLASS.PARTIAL:
        this._forceAnalyzing(current);
        return this._runAnalysis({ inputClass: INPUT_CLASS.PARTIAL });

      case INPUT_CLASS.CASUAL:
        return this._handleCasualInput(message);

      default:
        this._forceAnalyzing(current);
        return this._runAnalysis({ inputClass: INPUT_CLASS.PARTIAL });
    }
  }

  // ── Public: State + Events ────────────────────────────────────────────────

  /**
   * Returns current frozen SESSION_STATE.
   * @returns {Readonly<Object>}
   */
  getState() {
    return this.#store.getState();
  }

  /**
   * Subscribe to engine events.
   * Returns unsubscribe function.
   *
   * @param {string}   eventType - use EVENT constants
   * @param {Function} handler   - ({ payload, version, eventType }) => void
   */
  on(eventType, handler) {
    return this.#eventBus.on(eventType, handler);
  }

  /**
   * Subscribe to ALL state changes (convenience).
   * Returns unsubscribe function.
   */
  onStateChange(handler) {
    return this.#store.subscribe(handler);
  }

  /**
   * Register a debug/replay/audit hook called synchronously after every state commit.
   *
   * @param {Function|null} fn
   *   fn(version: number, state: Readonly<Object>, meta: { sourceLayer, capturedVersion, eventType }) => void
   *   Pass null to clear.
   *
   * @example
   *   engine.onStateSnapshot((version, state, meta) => {
   *     console.log(`[v${version}] ${meta.sourceLayer} / ${meta.eventType}`, state.system_state);
   *   });
   */
  onStateSnapshot(fn) {
    this.#store.setSnapshotHook(fn);
  }

  /**
   * Returns the current output layer state — which layers are visible, pending,
   * updating, or stale, and what data each layer contains.
   *
   * Shape:
   *   {
   *     layers: {
   *       interpretation:   { status, data, revealedAt, updatedAt },
   *       summary:          { status, data, revealedAt, updatedAt },
   *       signal_expansion: { status, data, revealedAt, updatedAt },
   *       action:           { status, data, revealedAt, updatedAt },
   *     }
   *   }
   *
   * Use LAYER_ID and LAYER_STATUS constants to compare values.
   */
  getOutputState() {
    return this.#outputLayerController.getOutputState();
  }

  /**
   * Subscribe to output layer changes.
   * Called whenever any layer's status or data changes.
   * Returns unsubscribe function.
   *
   * @param {Function} fn - (outputState) => void
   */
  onOutputChange(fn) {
    return this.#outputLayerController.subscribe(fn);
  }

  // ── Public: Inline Interactions ───────────────────────────────────────────

  /**
   * User edited an interpretation component (Type A correction).
   * Sets user_override = true on that component.
   * Layer 2 will never overwrite it again.
   *
   * @param {string} component - 'problem' | 'market' | 'mechanism' | 'context' | 'traction'
   * @param {string} value     - corrected text
   */
  editInterpretationComponent(component, value) {
    const state = this.#store.getState();
    this.#store.update(
      (current) => ({
        interpretation: {
          ...current.interpretation,
          [component]: {
            ...current.interpretation[component],
            value,
            user_override: true,
            version: Date.now(),
          },
        },
      }),
      { sourceLayer: 'USER' },
    );

    this.#eventBus.emit(EVENT.INTERPRETATION_UPDATED, {
      components: [component],
      interpretation: { [component]: { value, user_override: true } },
    });

    // Recompute dependency chain for this component via the forward graph
    this.#recomputation.onComponentUpdated(component);
  }

  /**
   * User confirmed a signal assumption (Type C correction).
   * Promotes assumption, upgrades signal confidence, recomputes affected conclusions.
   *
   * @param {string} assumptionId
   */
  confirmAssumption(assumptionId) {
    const state      = this.#store.getState();
    const assumption = state.assumptions.active.find(a => a.id === assumptionId);
    if (!assumption) return;

    const affectedConclusions = [assumption.supports_conclusion].filter(Boolean);

    this.#store.update(
      (current) => ({
        assumptions: {
          active: current.assumptions.active.map(a =>
            a.id === assumptionId
              ? { ...a, confirmed: true, confirmed_type: 'knowledge' }
              : a,
          ),
        },
      }),
      { sourceLayer: 'USER' },
    );

    this.#confidence.onAssumptionConfirmed(assumptionId, affectedConclusions, state.version);
    this.#eventBus.emit(EVENT.ASSUMPTION_CONFIRMED, { assumptionId, affectedConclusions });
  }

  /**
   * User rejected a signal assumption.
   * Signal returns to ABSENT. Retraction protocol triggered if conclusion loses all support.
   *
   * @param {string} assumptionId
   */
  rejectAssumption(assumptionId) {
    const state      = this.#store.getState();
    const assumption = state.assumptions.active.find(a => a.id === assumptionId);
    if (!assumption) return;

    const affectedConclusions = [assumption.supports_conclusion].filter(Boolean);

    this.#store.update(
      (current) => ({
        assumptions: {
          active: current.assumptions.active.filter(a => a.id !== assumptionId),
        },
      }),
      { sourceLayer: 'USER' },
    );

    this.#confidence.onAssumptionRejected(assumptionId, affectedConclusions, state.version);

    // Check if any conclusion loses all support → retraction
    for (const conclusionId of affectedConclusions) {
      const conclusion  = state.output.conclusions[conclusionId];
      const hasSupport  = (conclusion?.signals?.length > 0);
      if (!hasSupport) {
        this.#recomputation.retract(conclusionId, `Assumption rejected: ${assumption.value}`);
      }
    }

    this.#eventBus.emit(EVENT.ASSUMPTION_REJECTED, { assumptionId, affectedConclusions });
  }

  // ── Public: Perception ───────────────────────────────────────────────────

  /**
   * Subscribe to PERCEPTION_SEQUENCE events.
   * Called when a timed sequence of perception items is ready to begin.
   * Returns unsubscribe function.
   *
   * Sequence shape:
   *   {
   *     id:            string,             // unique sequence ID
   *     items:         PerceptionItem[],   // ordered by delay
   *     totalDuration: number,             // ms until last item settles
   *     isInterrupt:   boolean,            // true if redirecting from a prior sequence
   *     redirectsFrom: string | null,      // ID of the interrupted sequence
   *   }
   *
   * PerceptionItem shape:
   *   {
   *     targetId:  string,               // e.g. 'layer:summary', 'state:analyzing'
   *     change:    string,               // e.g. 'reveal', 'update', 'retract', 'pulse'
   *     weight:    PERCEPTION_WEIGHT,    // 'PRIMARY' | 'SECONDARY' | 'TERTIARY'
   *     delay:     number,               // ms from sequence start — on 80ms grid
   *     duration:  number,               // ms for the change to settle — on 80ms grid
   *     meta:      Object,               // source event payload
   *   }
   *
   * @param {Function} fn - (sequence) => void
   */
  onPerceptionSequence(fn) {
    return this.#eventBus.on(EVENT.PERCEPTION_SEQUENCE, ({ payload }) => fn(payload));
  }

  /**
   * Subscribe to PERCEPTION_ANTICIPATE events.
   * Fires 2u (160ms) before a predicted major reveal — advisory, not guaranteed.
   * Use to prepare layout or show a subtle "readying" state.
   * Returns unsubscribe function.
   *
   * Payload: { for: string, window: number }
   *   `for`    — what is being anticipated (e.g. 'analysis', 'reveal:summary')
   *   `window` — ms remaining before the anticipated event (always 160)
   *
   * @param {Function} fn - ({ for, window }) => void
   */
  onPerceptionAnticipate(fn) {
    return this.#eventBus.on(EVENT.PERCEPTION_ANTICIPATE, ({ payload }) => fn(payload));
  }

  /**
   * Subscribe to PERCEPTION_ANTICIPATE_CANCELLED events.
   * Fires when a previously emitted PERCEPTION_ANTICIPATE will not be followed
   * by the anticipated event. The UI MUST exit any "prepared" state on this signal.
   * Returns unsubscribe function.
   *
   * Without this, the UI could be indefinitely stuck in a preparation posture
   * if the anticipated reveal is blocked (e.g., pipeline cancelled, stability
   * window never closes due to rapid edits).
   *
   * Payload: { for: string, reason: string }
   *   `for`    — what was anticipated (matches the preceding ANTICIPATE payload)
   *   `reason` — 'timeout' | 'user_paused' | 'idle_reset' | 'pipeline_interrupted'
   *
   * @param {Function} fn - ({ for, reason }) => void
   */
  onPerceptionAnticipateCancelled(fn) {
    return this.#eventBus.on(EVENT.PERCEPTION_ANTICIPATE_CANCELLED, ({ payload }) => fn(payload));
  }

  /**
   * Subscribe to PERCEPTION_INTERRUPT events.
   * Fires when an active sequence is interrupted and should redirect gracefully.
   * Use to smoothly redirect in-progress motion rather than snapping to new state.
   * Returns unsubscribe function.
   *
   * Payload: { sequenceId: string, progress: number, reason: string }
   *   `progress` — [0,1] how far through the sequence was when interrupted
   *   `reason`   — 'user_paused' | 'idle_reset' | 'pipeline_interrupted'
   *
   * @param {Function} fn - ({ sequenceId, progress, reason }) => void
   */
  onPerceptionInterrupt(fn) {
    return this.#eventBus.on(EVENT.PERCEPTION_INTERRUPT, ({ payload }) => fn(payload));
  }

  // ── Public: Animation registry ───────────────────────────────────────────

  /**
   * Register an animation with the phase animation registry.
   * The future UI layer calls this when starting animations,
   * so sync points know to wait.
   *
   * @param {string} animationId
   * @param {number} estimatedDuration - ms
   * @returns {Function} complete — call when animation ends
   */
  registerAnimation(animationId, estimatedDuration) {
    this.#animationRegistry.register(animationId, estimatedDuration);
    return () => this.#animationRegistry.complete(animationId);
  }

  /**
   * Wait for sync point to clear (used by future UI layer).
   * @param {string} syncPoint - SYNC_POINT constant
   */
  waitForSyncPoint(syncPoint) {
    return this.#syncGate.waitForClear(syncPoint);
  }

  // ── Public: Thread Management ─────────────────────────────────────────────

  /** Switch to an existing inactive thread. */
  switchThread(threadId) {
    this.#threadManager.switchTo(threadId);
  }

  /** Get all inactive threads for context rail rendering. */
  getInactiveThreads() {
    return this.#threadManager.getInactiveThreads();
  }

  // ── Public: Lifecycle ────────────────────────────────────────────────────

  /** Tear down the engine. Releases all timers, listeners, bindings. */
  destroy() {
    this.#isDestroyed = true;
    this.detach();
    this.#reducer.destroy();
    this.#stateMachine.destroy();
    this.#confidence.destroy();
    this.#performanceSync.destroy();
    this.#outputLayerController.destroy();
    this.#perceptionSync.destroy();
    this.#layer2.cancel();
    this.#layer3.cancel();
    this.#pipeline.interrupt();
    this.#eventBus.destroy();
  }

  // ── Internal: Engine Wiring ───────────────────────────────────────────────

  /**
   * Internal event subscriptions that orchestrate cross-module behavior.
   * This is the three-layer → analysis → output coordination layer.
   */
  _bindEngineWiring() {
    // Signal changes → confidence re-evaluation + recomputation routing.
    // Completeness is NOT synced here — it is derived from state snapshot in the
    // store subscriber below, so event ordering cannot affect its correctness.
    this.#eventBus.on(EVENT.SIGNAL_DETECTED, ({ payload }) => {
      this.#confidence.scheduleReEvaluation();
      if (payload?.signal?.type) {
        this.#recomputation.onSignalAdded(payload.signal.type);
      }
    });
    this.#eventBus.on(EVENT.SIGNAL_REMOVED, ({ payload }) => {
      this.#confidence.scheduleReEvaluation();
      if (payload?.type) {
        this.#recomputation.onSignalRemoved(payload.type);
      }
    });

    // Completeness sync — derived exclusively from state snapshot, never from event order.
    // Runs after every state write; only re-computes when signals.registry actually changed.
    // This guarantees completeness is correct regardless of how or when signals were written.
    let _lastRegistryKey = '';
    this.#store.subscribe((state) => {
      // Registry identity: sorted type list is sufficient (types are the weight keys).
      const registryKey = state.signals.registry.map(s => s.type).sort().join('\0');
      if (registryKey === _lastRegistryKey) return;
      _lastRegistryKey = registryKey;
      syncCompleteness(this.#store, this.#eventBus, state.version);
    });

    // Pipeline complete → SP4 gate.
    // OutputLayerController's store subscriber fires when conclusions land in state
    // and handles progressive revelation — no manual OUTPUT_LAYER_REVEALED emit here.
    this.#eventBus.on(EVENT.PIPELINE_COMPLETED, async () => {
      // SP4: wait for any active reveal animations before output layers can re-evaluate.
      // The OutputLayerController's stability window (400ms) absorbs most of this naturally.
      await this.#syncGate.waitForClear(SYNC_POINT.SP4);
    });

    // State transitions → interrupt pipeline, manage sync gates, auto-trigger analysis
    this.#eventBus.on(EVENT.STATE_TRANSITION, ({ payload }) => {
      if (payload.to === SYSTEM_STATE.PAUSED) {
        this.#pipeline.interrupt();
        this.#syncGate.clearAll();
      }
      if (payload.to === SYSTEM_STATE.REDIRECTED) {
        this.#syncGate.reset();
      }
      // Phase 6B: stability timer fired → auto-run pipeline (the three-layer loop closes here)
      if (payload.to === SYSTEM_STATE.ANALYZING && payload.trigger === 'interpretation_stable') {
        const state     = this.#store.getState();
        const intent    = this.#intentClassifier.classify(state.input.raw, state);
        const inputClass = intent.class === INPUT_CLASS.DIRECTIVE
          ? INPUT_CLASS.DIRECTIVE
          : INPUT_CLASS.COMPLETE;
        this._runAnalysis({ inputClass });
      }
    });

    // Thread restore with schema mismatch → partial recompute of stale conclusions
    this.#eventBus.on(EVENT.THREAD_SWITCHED, ({ payload }) => {
      if (payload?.needsRecompute && payload.staleConclusions?.length > 0) {
        this.#recomputation.recompute(payload.staleConclusions, { trigger: 'thread_restore' });
      }
    });

    // State version changes → check stale revealed layers
    this.#store.subscribe((state) => {
      if (state.output.layers_revealed.length > 0) {
        this.#recomputation.checkStaleLayers();
      }
    });
  }

  // ── Internal: Flow Handlers ───────────────────────────────────────────────

  // Ensure state machine is in ANALYZING before _runAnalysis() is called by submit().
  // Does NOT cancel the in-flight run — _runAnalysis() handles that via #analysisRun.
  _forceAnalyzing(current) {
    if (current === SYSTEM_STATE.RECEIVING || current === SYSTEM_STATE.INTERPRETING) {
      this.#stateMachine.transition(SYSTEM_STATE.ANALYZING, { trigger: 'user_submit' });
      this.#eventBus.emit(EVENT.PIPELINE_STARTED, {});
    }
    // If already ANALYZING: #analysisRun abort in _runAnalysis() handles cancellation.
  }

  _runAnalysis({ inputClass = INPUT_CLASS.COMPLETE } = {}) {
    // Hard-cancel any in-flight run before starting a new one.
    // This is the single enforcement point — no other code calls pipeline.run().
    if (this.#analysisRun) {
      this.#analysisRun.abortController.abort();
    }

    const abortController = new AbortController();
    const runId           = Symbol('analysisRun');
    this.#analysisRun     = { id: runId, abortController };

    return this.#syncGate.waitForClear(SYNC_POINT.SP3)
      .then(() => this.#pipeline.run({ inputClass, signal: abortController.signal }))
      .finally(() => {
        if (this.#analysisRun?.id === runId) this.#analysisRun = null;
      });
  }

  async _handlePausedSubmit(message, intent) {
    const state = this.#store.getState();
    const classification = this.#interruptionClassifier.classify(state, message);

    switch (classification.type) {
      case INTERRUPTION_TYPE.MERGING: {
        // Restore opacity (visual layer hook)
        this.#eventBus.emit(EVENT.INTERRUPTION_CLASSIFIED, {
          type: INTERRUPTION_TYPE.MERGING,
          ...classification,
        });
        // Resume from checkpoint
        const resumePhase = this.#pipeline.determineResumePhase(
          INTERRUPTION_TYPE.MERGING,
          { newPrimary: classification.newPrimary, newTier1: classification.newTier1, newTier2: classification.newTier2 },
        );
        this.#stateMachine.transition(SYSTEM_STATE.ANALYZING, { trigger: 'merging_resume' });
        await this.#pipeline.run({ resumeFromPhase: resumePhase, inputClass: INPUT_CLASS.COMPLETE });
        break;
      }

      case INTERRUPTION_TYPE.DEEPENING: {
        // Only re-run for targeted component
        const affected = this.#recomputation.computeAffected(
          `${classification.targetComponent}_signal`,
        );
        this.#eventBus.emit(EVENT.INTERRUPTION_CLASSIFIED, {
          type: INTERRUPTION_TYPE.DEEPENING,
          ...classification,
        });
        this.#stateMachine.transition(SYSTEM_STATE.ANALYZING, { trigger: 'deepening' });
        await this.#recomputation.recompute(affected, {
          changedSource: classification.targetComponent,
          trigger: 'deepening',
        });
        this.#stateMachine.transition(SYSTEM_STATE.COMPLETED, { trigger: 'deepening_done' });
        break;
      }

      case INTERRUPTION_TYPE.CLOSING:
        await this._handleRedirect(message);
        break;
    }
  }

  async _handleRedirect(message) {
    // Per spec: announce "Starting fresh on [detected topic]", archive thread, new thread
    const state = this.#store.getState();

    this.#threadManager.archiveActive();
    const newThreadId = this.#threadManager.createNew();

    // Reset active session state (preserves threads.registry)
    this.#store.resetSession(true);
    this.#layer1.reset();
    this.#layer2.cancel();
    this.#layer3.cancel();
    this.#syncGate.reset();

    this.#stateMachine.transition(SYSTEM_STATE.REDIRECTED, {
      trigger:      'redirect',
      newThreadId,
      fromMessage:  message,
    });

    this.#eventBus.emit(EVENT.THREAD_CREATED, { threadId: newThreadId });
  }

  async _handlePostAnalysisChat(message) {
    await this.#apiAdapter.interact(message);
  }

  async _handleCasualInput(message) {
    await this.#apiAdapter.interact(message);
  }
}
