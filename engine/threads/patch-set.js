/**
 * MIDAN Engine — JSON Patch (RFC 6902)
 * Implements the minimal subset required for PATCH_SET thread diffs:
 *   add, remove, replace
 * COMPACT_BASE: non-default SESSION_STATE fields only.
 * Omits: input.raw, system_state, threads.registry (per spec).
 */

// Fields excluded from COMPACT_BASE
const BASE_OMIT_KEYS = new Set(['system_state', 'threads']);
const BASE_DEEP_OMIT = { input: new Set(['raw']) };

/**
 * Generate a COMPACT_BASE from SESSION_STATE at thread creation time.
 * Stores only non-default fields.
 *
 * @param {Object} state - frozen SESSION_STATE
 * @returns {Object} compact base (plain JS object, not frozen)
 */
export function createCompactBase(state) {
  const base = {};
  for (const [key, value] of Object.entries(state)) {
    if (BASE_OMIT_KEYS.has(key)) continue;
    if (BASE_DEEP_OMIT[key]) {
      const sub = {};
      for (const [sk, sv] of Object.entries(value)) {
        if (!BASE_DEEP_OMIT[key].has(sk)) sub[sk] = deepClone(sv);
      }
      base[key] = sub;
    } else {
      base[key] = deepClone(value);
    }
  }
  return base;
}

/**
 * Generate an RFC 6902-compatible patch array from base → current.
 * Only generates: add, remove, replace operations.
 *
 * @param {Object} base    - COMPACT_BASE object
 * @param {Object} current - current SESSION_STATE
 * @returns {Array} RFC 6902 patch operations
 */
export function generatePatch(base, current) {
  const ops = [];
  diffObjects(base, current, '', ops, BASE_OMIT_KEYS, BASE_DEEP_OMIT);
  return ops;
}

/**
 * Apply an RFC 6902 patch array to a base object.
 * Returns a new object (non-mutating).
 *
 * @param {Object} base    - COMPACT_BASE
 * @param {Array}  patches - PATCH_SET
 * @returns {Object} reconstructed state
 */
export function applyPatch(base, patches) {
  let doc = deepClone(base);

  for (const op of patches) {
    const parts = parsePath(op.path);
    switch (op.op) {
      case 'add':
      case 'replace':
        setPath(doc, parts, op.value);
        break;
      case 'remove':
        removePath(doc, parts);
        break;
      default:
        console.warn('[Patch] unsupported op:', op.op);
    }
  }

  return doc;
}

/**
 * Reconstruct a full SESSION_STATE from COMPACT_BASE + PATCH_SET.
 * Fills defaults for omitted fields.
 * Sets system_state → 'COMPLETED', input.raw → ''.
 *
 * @param {Object} base    - COMPACT_BASE
 * @param {Array}  patches - PATCH_SET
 * @param {Object} defaults - createInitialState() for default fill
 * @returns {Object} restored SESSION_STATE (plain, not frozen)
 */
export function restoreFromSnapshot(base, patches, defaults) {
  // Start with defaults for omitted fields
  const restored = deepClone(defaults);

  // Apply compact base on top of defaults
  deepMergeInto(restored, base);

  // Apply patches
  const withPatches = applyPatch(restored, patches);

  // Override spec-required restore fields
  withPatches.system_state = 'COMPLETED';
  withPatches.input.raw    = '';

  return withPatches;
}

// ── Internal utilities ────────────────────────────────────────────────────────

function diffObjects(a, b, prefix, ops, omitKeys = new Set(), deepOmit = {}) {
  // Keys in B but not in A → add
  for (const key of Object.keys(b)) {
    if (omitKeys.has(key) && prefix === '') continue;
    const path = prefix ? `${prefix}/${key}` : `/${key}`;
    const av   = a?.[key];
    const bv   = b[key];

    if (deepOmit[key] && prefix === '') {
      // Recurse with deep-omit restrictions
      diffObjects(av ?? {}, bv, `/${key}`, ops, deepOmit[key]);
      continue;
    }

    if (!(key in (a ?? {}))) {
      ops.push({ op: 'add', path, value: deepClone(bv) });
    } else if (typeof bv === 'object' && bv !== null && !Array.isArray(bv) &&
               typeof av === 'object' && av !== null && !Array.isArray(av)) {
      diffObjects(av, bv, path, ops);
    } else if (!deepEqual(av, bv)) {
      ops.push({ op: 'replace', path, value: deepClone(bv) });
    }
  }

  // Keys in A but not in B → remove
  for (const key of Object.keys(a ?? {})) {
    if (omitKeys.has(key) && prefix === '') continue;
    const path = prefix ? `${prefix}/${key}` : `/${key}`;
    if (!(key in b)) {
      ops.push({ op: 'remove', path });
    }
  }
}

function parsePath(path) {
  // RFC 6902 path: "/foo/bar/0" → ['foo', 'bar', '0']
  return path.split('/').filter(Boolean).map(p => p.replace(/~1/g, '/').replace(/~0/g, '~'));
}

function setPath(doc, parts, value) {
  let obj = doc;
  for (let i = 0; i < parts.length - 1; i++) {
    const key = parts[i];
    if (obj[key] == null) obj[key] = {};
    obj = obj[key];
  }
  obj[parts[parts.length - 1]] = value;
}

function removePath(doc, parts) {
  let obj = doc;
  for (let i = 0; i < parts.length - 1; i++) {
    obj = obj?.[parts[i]];
    if (obj == null) return;
  }
  delete obj[parts[parts.length - 1]];
}

function deepMergeInto(target, source) {
  for (const key of Object.keys(source)) {
    const sv = source[key];
    const tv = target[key];
    if (sv && typeof sv === 'object' && !Array.isArray(sv) && tv && typeof tv === 'object') {
      deepMergeInto(tv, sv);
    } else {
      target[key] = deepClone(sv);
    }
  }
}

function deepClone(val) {
  if (val === null || typeof val !== 'object') return val;
  return JSON.parse(JSON.stringify(val));
}

function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (typeof a !== 'object' || a === null) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every(k => deepEqual(a[k], b[k]));
}
