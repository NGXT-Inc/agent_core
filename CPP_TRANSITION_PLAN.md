# Transition Plan: `agent_core` Python → Python+C++

This plan describes the staged migration of `agent_core`'s stateful core (session
registry, history storage, compaction, cache management) from Python into a
compiled C++ extension, while keeping provider implementations, the agent
orchestration loop, hooks, and events in Python.

## North Star — what a script looks like when we're done

Before laying out phases, fix what "hyper clean" means concretely. Every other
decision serves this:

```python
# any_script.py
from agent_core import Agent

class DesignerAgent(Agent):
    name = "designer"
    system_prompt = "..."

    def __init__(self, session_id, **kw):
        super().__init__(session_id=session_id, **kw)
        self.register_tool(self.draw)

    def draw(self, what: str) -> str:
        """Draw something."""
        ...

# Run it. No registry bookkeeping, no persistence config, no cleanup.
designer = DesignerAgent(session_id="user-42")
designer.run("draw a circle")

# Spawn sub-agents — hierarchical session_id is automatic
explorer = designer.spawn(ExplorerAgent)         # → "user-42:explorer-7f3a"
explorer.run("...")

# Any other script in the same process sees the same registry
from agent_core import registry
registry.list_active()                            # ["user-42", "user-42:explorer-7f3a"]
registry.cancel_subtree("user-42")
```

Three invariants for every phase:

1. **Public Python API stays stable.** `test_papyrus_compat.py` keeps passing
   throughout. Existing consumers don't change a line.
2. **The C++ registry is a process-global module singleton.** Scripts import it;
   they never construct or destroy it.
3. **Sub-agent spawning produces hierarchical session_ids automatically.**
   Scripts never assemble them by hand.

## Architecture target

```
   ┌────────────────────────────────────────────────────────────────┐
   │  Python                                                        │
   │  ────────────────────────────────────────────────────────      │
   │                                                                │
   │   agent_core/                                                  │
   │     __init__.py        ←  re-exports, including `registry`     │
   │     agent.py           ←  Agent class (thinned)                │
   │     spawning.py        ←  Agent.spawn() helper                 │
   │     providers/         ←  unchanged: gemini, openai, openrouter│
   │     events.py          ←  unchanged: EventBus                  │
   │                                                                │
   └────────────────────────┬───────────────────────────────────────┘
                            │  nanobind boundary
   ┌────────────────────────▼───────────────────────────────────────┐
   │  C++  (agent_core._native, compiled extension)                 │
   │                                                                │
   │     class Registry           ← global singleton                │
   │     struct Session                                             │
   │     class HistoryStore       ← per-session, owned by Session   │
   │     class CompactionEngine   ← policy + summarize callback     │
   │     class CacheManager       ← create/delete callbacks         │
   │     class SQLiteWriter       ← background async writer thread  │
   │                                                                │
   └────────────────────────────────────────────────────────────────┘
```

---

## Phase 0 — Build infrastructure

**Goal:** Ship a wheel that exposes an empty `agent_core._native` extension. No
behavior change.

**Why first:** This is the most likely thing to bog you down for an entire week
if you push it later. Get the build green before writing any real code.

**Deliverables:**

- `pyproject.toml` switches from `hatchling` to `scikit-build-core` as the build
  backend.
- `CMakeLists.txt` for the native extension.
- `nanobind` chosen over pybind11 — smaller wrappers, faster, better
  `py::object` handling. (For a perf-motivated rewrite this is the right pick.)
- `cibuildwheel` configured in CI for Linux (x86_64, aarch64) and macOS
  (x86_64, arm64). Skip Windows unless required.
- `src/agent_core/_native.cpp` exports one no-op function (`_native.hello()`
  returns `"ok"`).
- `from agent_core import _native; _native.hello()` works in an editable install
  and in a built wheel.

**Verification:**

- Existing test suite runs identically (everything still passes — the extension
  exists but nothing uses it).
- Wheels build green on CI matrix.

**Risk:** Toolchain headaches across CI runners (macOS arm64 + Linux aarch64
cross-compile). Mitigate by using cibuildwheel's defaults; only customize when
something breaks.

---

## Phase 1 — Canonical message format

**Goal:** Lock the wire format that crosses the Python↔C++ boundary forever.
Land it as pure Python first so it's testable before any C++ depends on it.

**Why now:** Every later phase needs this. Get it wrong here and every
dependency rewrites.

**What changes:**

- Formalize the dual-slot model in Python type definitions:

  ```python
  @dataclass(frozen=True, slots=True)
  class CanonicalMessage:
      role: Literal["user", "assistant", "tool", "system"]
      provider_tag: str            # "gemini" | "openai"
      canonical_json: str          # what gets persisted
      approx_tokens: int           # cached at construction
      provider_native: Any         # opaque — the provider's own message object
  ```

- Add `provider.to_canonical(native_msg) -> CanonicalMessage` and
  `provider.from_canonical(c) -> native_msg` to the `LLMProvider` Protocol.
  Implement in `GeminiProvider` and `OpenAIProvider`. These are thin wrappers
  around the existing `serialize_message` / `deserialize_message`.
- Add an `approx_tokens(text: str) -> int` helper exposed by each provider
  (default: `len/4` like `compaction.py` does today).

**Deliverables:**

- New `CanonicalMessage` dataclass in `providers/types.py`.
- Two new methods on each provider.
- Roundtrip tests: `provider.from_canonical(provider.to_canonical(m)) == m` for
  every message shape (text, tool_call, tool_result, multimodal placeholder,
  thought).

**Verification:**

- All existing tests still pass (this phase only adds API; nothing uses it yet).
- New tests cover roundtrip for Gemini, OpenAI, OpenRouter.

**Risk:** Provider-specific edge cases (multimodal output parts,
reasoning_details). Address them in tests now — finding them later in C++ is
much more painful.

---

## Phase 2 — `HistoryStore` in C++

**Goal:** Move `Agent._history` and `_save_history()` into C++. Nothing else
moves yet. This is the lowest-risk migration that validates the whole approach.

**Why second:** It's the simplest stateful component. If the dual-slot design
doesn't work in practice, you find out here, before three more components depend
on it.

**What changes in C++:**

- `MessageSlot` struct (Python object handle + canonical_json + role +
  approx_tokens).
- `HistoryStore` class:
  - `append(py::object native, std::string canonical_json, uint32_t tokens, std::string role)`
  - `snapshot_native() -> std::vector<py::object>` — for sending to provider
  - `snapshot_canonical() -> std::vector<std::string>` — for compaction's
    transcript rendering (later)
  - `replace_prefix(size_t n, MessageSlot summary)` — for compaction (later)
  - `clear()`
  - `size()`, `total_approx_tokens()`
- `SQLiteWriter` class — single background thread, write queue, idempotent
  coalescing. The queue absorbs the "4-8 saves per turn" write amplification.

**What changes in Python:**

- `Agent._history` becomes a thin Python wrapper that holds a `HistoryStore`
  handle.
- `_save_history()` becomes a no-op (the C++ writer handles it).
- Persistence — `SQLiteConversationStore` is replaced for the `Agent.run()`
  path by C++'s writer. The Python `SQLiteConversationStore` class stays in the
  API for backward compatibility with code that constructs it directly (Papyrus
  does), routing to the same underlying C++ writer.

**Deliverables:**

- `_native.HistoryStore` exposed.
- Agent constructor uses `HistoryStore` when `session_id` is provided; falls
  back to a pure-Python list otherwise (for `run_stateless` and ephemeral
  agents).
- Benchmark: a session with 200 tool-call turns, compare end-to-end wall-clock
  + SQLite write count vs current code. Target: ≥3× fewer fsyncs.

**Verification:**

- All existing tests pass (Python API unchanged).
- New micro-benchmark in `tests/perf/` showing the SQLite improvement.
- Crash-recovery test: kill mid-loop, restart, verify history loads correctly.

**Risk:** GIL handling around `py::object` refs in the message slot. Every
append/snapshot must hold the GIL; the SQLite writer must release it. nanobind
makes this easier than pybind11 but you still need to write
`nb::gil_scoped_release` in the right places.

---

## Phase 3 — Registry as global singleton  *(the main objective)*

**Goal:** Introduce `Registry` as a module-global. Every `Agent` holds a
`SessionHandle` from it. Hierarchical session_ids are automatic.

**Why now:** With history already in C++, the session struct is mostly built —
wrapping it in a registry is mostly plumbing. Doing it before this phase would
mean migrating the same data twice.

**What changes in C++:**

- `Session` struct (as sketched in earlier design discussion): identity,
  history, config, cache slot (empty until phase 4), cancellation atomic,
  refcount, per-session shared_mutex.
- `Registry` class:
  - Global singleton, accessible from any module.
  - `acquire(session_id, config) -> SessionHandle` — creates or
    resurrects-from-SQLite or refcount++.
  - `release(session_id)` — refcount--.
  - `get(session_id) -> optional<SessionHandle>` — no creation.
  - `cancel_subtree(root_session_id)` — prefix scan, atomic flag.
  - `clear_cancellation(session_id, recursive=False)`.
  - `descendants(root)` / `list_active()`.
  - Background reaper thread: idle TTL eviction, ref_count==0.
- `SessionHandle` — RAII handle. Drop = release.

**What changes in Python:**

- `Agent.__init__` calls `_native.registry.acquire(session_id, config)`. The
  returned handle replaces `_history`, `_session_id`, `_cancel_event`,
  `_compaction_count`, `_instance_id` — they all become properties of the
  session.
- `Agent.spawn(ChildClass, **kw)` — new helper. Composes
  `child_session_id = parent.session_id + ":" + child_name + "-" + uuid_hex[:8]`,
  passes to `ChildClass.__init__`. The hyper-clean piece.
- `Agent.cancel(recursive=True)` calls `registry.cancel_subtree(self.session_id)`.
- `Agent.close()` drops the handle. `__del__` does the same defensively.
- A new top-level export: `from agent_core import registry`. It's a Python
  proxy over the C++ singleton with the methods `list_active`,
  `cancel_subtree`, `get`, `descendants`. Read-only by design — no
  `acquire`/`release` from Python (those belong to `Agent`'s lifecycle).

**Deliverables:**

- Working hierarchical spawning.
- A demo script (`examples/multi_script_shared_registry.py`) showing two imports
  of the same module sharing one session view.
- Migration of `cancel_event=parent._cancel_event` to subtree cancellation in
  all tests.

**Verification:**

- All existing tests pass. The `test_shared_cancel_event_propagates` test
  rewrites to use `spawn()` instead of explicit event sharing — both should
  produce identical behavior.
- New test: `spawn()` produces correct hierarchical session_ids.
- New test: `registry.cancel_subtree` propagates across all descendants.
- New test: resurrection from SQLite — kill process, restart,
  `registry.acquire(same_sid)` finds the same history.

**Risks:**

- **Hidden lifecycle bugs.** Refcount mismanagement (double-release, missed
  release) will manifest as ghost sessions or premature eviction. Mitigate with
  strict RAII: `SessionHandle` is move-only, never copyable; Python wrapper
  holds exactly one reference.
- **Lock ordering.** Registry mutex must always be acquired before session
  mutex, never the reverse. Establish this convention in code review.
- **Separator collision.** Validate that `agent_type` names and ID suffixes
  can't contain `:`. Reject at acquire time.

---

## Phase 4 — `CacheManager` in C++

**Goal:** Move `ContextCacheRegistry` state into C++. The Vertex API calls stay
in Python (callback).

**Why now (not earlier):** Cache logic is independent of compaction, so this can
ship independently. It's also the second-smallest component, and it lets you
validate the "Python callback from C++ background thread" pattern before
compaction needs it for summarization.

**What changes in C++:**

- `CacheSlot` (was already a placeholder in `Session`): ready_name, ready_offset,
  ready_created_at, pending future state, last_cache_token_count,
  config_fingerprint.
- `CacheManager` class:
  - `get_advice(session_id, system_prompt, tool_names) -> CacheAdvice`
  - `notify(session_id, token_count)` — submits to background work queue.
  - `invalidate(session_id)` — also called by `HistoryStore::clear()` and on
    tool registration changes.
  - Reaper thread.
- Background work queue with a thread pool. The "create cache" task calls back
  to Python with the GIL acquired:
  `py_create_cb(session_id, model, contents, system_prompt, tools, ttl)`.
- SHA-256 fingerprinting moves to C++ (use any libcrypto / SHA implementation;
  the input is short).

**What changes in Python:**

- `Agent` no longer holds `_cache_enabled`, `_instance_cache_registry`. It calls
  `session.cache_advice()` and `session.cache_notify(tokens)` directly.
- `Agent.init_cache_registry()` / `shutdown_cache_registry()` become no-ops with
  a deprecation warning. The registry is global and managed by the module.
- A new module-level function:
  `agent_core.configure_caching(create_callback, delete_callback)`. Called once
  at app start, wires the Python callbacks that hit Vertex. For Gemini
  providers this is auto-wired when the provider is first instantiated.
- OpenAI/OpenRouter providers still return `False` from
  `supports_context_cache_registry` — the C++ manager simply never queues work
  for sessions whose provider doesn't support it.

**Deliverables:**

- `_native.CacheManager` exposed via the registry.
- All `test_caching.py` tests pass.
- New stress test: 100 concurrent agents acquiring/releasing cache advice.
  Validate no deadlocks, no leaked Vertex caches.

**Verification:**

- Existing `test_caching.py` passes without modification (the API is the same).
- Vertex create/delete callbacks fire the same number of times as before.
- A new test exercises the case where a cache create callback raises — the
  manager handles it gracefully.

**Risk:** Calling Python callbacks from C++ background threads. nanobind's
`nb::gil_scoped_acquire` handles this, but if you forget it once you get a hard
crash. Lint rule + test that runs under `PYTHONDEVMODE=1` to catch GIL
violations.

---

## Phase 5 — `CompactionEngine` in C++

**Goal:** Move compaction policy logic to C++. The summary model call
round-trips back to Python.

**Why last:** Compaction depends on history (phase 2) and the registry
(phase 3). It also needs the cache to be invalidatable atomically (phase 4).
Doing it before the others would require interim shims.

**What changes in C++:**

- `CompactionConfig` — same shape as today.
- `CompactionPolicy` virtual interface. Ship one implementation,
  `FlatSummaryPolicy`, that matches current behavior exactly. Sets up the
  abstraction so adding `HierarchicalPolicy`, `KeyEventPolicy`, etc. later is
  just a new subclass.
- `CompactionEngine` class:
  - `maybe_compact(session_id, reason)` — returns true if compaction ran.
  - Token estimation in C++ using cached `approx_tokens` from message slots.
    Optional callback for accurate provider-side counting.
  - Tail selection in C++ using cached canonical_json for transcript rendering.
  - Calls registered
    `py_summarize_cb(session_id, system_prompt, prompt_text, max_tokens)` for
    the actual model invocation. The callback runs `provider.generate(...)` and
    returns the summary string.
  - Atomic prefix swap via `HistoryStore::replace_prefix()`.
  - Invalidates the cache slot after a successful swap.
  - Emits `CONTEXT_COMPACTION` events through a registered Python event
    callback (or, by phase 6, through the EventBus binding).
- Provider's `adjust_compaction_tail_start` becomes a registered C callback on
  the engine, keyed by `provider_tag`. The engine queries it before finalizing
  the tail boundary.

**What changes in Python:**

- The compaction summary loop inside `_run_with_function_loop` becomes a single
  call: `session.maybe_compact(reason="context_limit")`.
- All the `_maybe_compact_history`, `_generate_compaction_summary`,
  `build_compaction_summary_prompt`, `build_compacted_summary_message`,
  `_emit_compaction_event` methods on `Agent` either move to C++ or become thin
  shims that the engine invokes.
- `get_compaction_config()` stays in Python (Agent subclasses override it). On
  `Agent.__init__`, the resolved config is pushed into the session.

**Deliverables:**

- `_native.CompactionEngine` exposed.
- Existing compaction tests pass without behavior change.
- A new test exercises a custom policy: register a `MockPolicy` from Python
  that always says "compact everything from index 5"; verify the engine
  respects it.

**Verification:**

- Compaction triggers at the same token thresholds as before.
- The summary call still excludes `system_prompt` and `tool_schemas` (verified
  by an existing test).
- Events emitted are unchanged in shape.

**Risk:** The summarize callback may block for many seconds (an LLM call).
During that time, the session must remain usable for **read** operations (other
agents in the tree might query state), but **not for further compaction** of
the same session. Add a per-session compaction-in-flight flag, drop concurrent
compaction requests on the floor.

---

## Phase 6 — Cleanup and consolidation

**Goal:** Remove all the dead code in `Agent` that the C++ layer made redundant.
Lock in the final public API. Update docs.

**What changes:**

- `Agent` shrinks substantially. Target: under 700 lines.
- Remove `_save_history`, `_invalidate_cache`, `_maybe_compact_history`,
  `_count_context_tokens_for_compaction`, the whole `_compaction_policy_details`
  helper family, `init_cache_registry` / `shutdown_cache_registry`,
  `_owns_cancel_event`, `_cache_registry`.
- `core/persistence.py`: the `SQLiteConversationStore` becomes a thin shim that
  just opens the same database the C++ writer uses (for Papyrus's direct
  calls). `serialize_content` / `deserialize_content` stay for backward
  compatibility.
- `core/caching.py`: deleted. Replaced by C++.
- `agents/compaction.py`: stays in Python but reduces to just `CompactionConfig`
  (the dataclass). All helper functions move to C++.
- README updates: explain the registry, document `Agent.spawn`, document
  `agent_core.registry`.

**Deliverables:**

- A diff showing the LOC reduction in `agents/base.py`.
- Updated README with the new "hyper-clean script" examples.
- A migration guide for downstream consumers (mostly: nothing to do, but new
  APIs noted).

**Verification:**

- Full test suite passes.
- Papyrus compat suite passes without modification.
- A fresh checkout build-test-runs in under N minutes on CI.

---

## Cross-cutting concerns

### Testing strategy

The existing test suite is the safety net. Its job through every phase is to
**catch behavior regressions** while internals move under it. Three rules:

1. **No test file moves to C++.** All tests stay in Python and exercise the
   public API. The C++ extension is tested through its Python bindings.
2. **Add C++-specific tests as new files**, never replacing existing ones. By
   phase 6 you'll have `tests/test_native_registry.py`,
   `tests/test_native_history.py`, etc., alongside the originals.
3. **Run the full suite in CI after every phase**, not just at the end. If a
   Papyrus compat test breaks, you know the phase you're on did it.

### Build and distribution

- Ship pre-built wheels for every supported platform. Source distribution stays
  available but should not be the common path.
- Editable installs need to recompile on header changes. scikit-build-core
  handles this but it's slower than pure-Python editable installs. Acknowledge
  the dev-loop cost; mitigate by keeping C++ headers small.
- Pin nanobind version. Their API has minor breaking changes occasionally.

### Threading and GIL

Three rules, enforced in code review:

1. **Every `py::object` interaction holds the GIL.** This includes constructing,
   refcounting, calling methods.
2. **Pure C++ work releases the GIL.** SQLite writes, mutex acquisitions,
   atomic operations, std::string manipulation — all under
   `nb::gil_scoped_release`.
3. **Background threads acquire the GIL only at call-back points.** The SQLite
   writer thread, cache reaper, and any future workers run GIL-free and only
   re-acquire when calling Python.

### Observability for "different scripts"

Since multiple scripts share one registry, you need to be able to inspect it
from any script:

- `registry.list_active()` — already in the plan.
- `registry.session_info(sid) -> dict` — returns
  `{agent_type, message_count, total_tokens, last_access, ref_count, has_cache, cancelled}`.
- `registry.dump()` — full snapshot for debugging.
- Optional: a tiny `agent_core.cli` module so
  `python -m agent_core.cli list-sessions` works from a shell, even attaching
  to a running process via a unix socket if you decide that's worth it. (Defer
  — phase 7 or later.)

### Public API freeze

By end of phase 1, the Python public API should be considered frozen except for
additions. Every later phase changes internals only. This is the single most
important rule for keeping consumer code clean — and it's what makes
"hyper clean scripts" possible.

---

## Risk register

| Risk | Phase | Mitigation |
|---|---|---|
| Build matrix breaks on CI | 0 | cibuildwheel defaults; minimize customization |
| Canonical format misses an edge case | 1 | Exhaustive roundtrip tests across all part types |
| GIL violation crashes randomly | 2, 4, 5 | Test under `PYTHONDEVMODE=1` + `-X dev`; nanobind's GIL helpers |
| Refcount leaks → ghost sessions | 3 | RAII handles, move-only types, leak test in CI |
| Lock ordering deadlock | 3, 4, 5 | Single rule (registry → session, never reverse); document on every lock acquisition |
| Summary callback blocks indefinitely | 5 | Per-session in-flight flag; drop concurrent requests; configurable timeout |
| Compaction races with mutation | 5 | Snapshot tail before callback; verify unchanged before swap; otherwise abort |
| Cross-process expectation creep | any | Be explicit: single-process registry; SQLite is the cross-process truth |

---

## Timeline shape (not commitments — relative weight)

If phases were units of effort: **0 ≈ 1 < 2 ≈ 3 > 4 > 5 > 6**.

Phase 3 (the registry) is the biggest single milestone because it's where the
architecture's character changes. Phases 2 and 4 are lower-risk because they
have narrow scopes with clear before/after benchmarks. Phase 5 is medium risk
despite its scope — the summarization callback contract is subtle.

A reasonable sequencing target: ship phase 2 as the first "real" milestone where
consumers see a benefit (faster persistence). Ship phase 3 next as the major
architectural change. Phases 4-6 then sequence in over several smaller releases.

---

## Phase-by-phase shipping criteria, in one paragraph

**Phase 0:** wheels build, no behavior. **Phase 1:** roundtrip tests pass, no
behavior. **Phase 2:** SQLite write count drops, all tests pass. **Phase 3:**
hierarchical spawning works, registry visible across scripts, cancellation
subtrees work. **Phase 4:** cache tests still pass, Vertex callbacks unchanged.
**Phase 5:** compaction tests still pass, summary callbacks unchanged.
**Phase 6:** LOC reduction visible, README updated, Papyrus still works
untouched.

Each is independently shippable. Each leaves the public API unchanged. That's
the discipline that makes the whole transition tractable.
