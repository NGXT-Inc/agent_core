"""Demo: a single process where two scripts share the C++ session registry.

The point of this example is to show that:

* The ``agent_core.registry`` module exposes a process-global view that any
  Python script in the same process can read and operate on, without any
  setup or teardown on the script's part.
* ``Agent.spawn()`` composes hierarchical ``session_id`` strings automatically,
  so a sub-agent's session lives "under" its parent's in the registry tree.
* ``registry.cancel_subtree(root_session_id)`` reaches every descendant
  regardless of which script created it.

Run from the repo root:

    python examples/multi_script_shared_registry.py

This is a self-contained demo — it does not call any LLM provider.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_core import Agent, registry
from agent_core.providers.openai import OpenAIProvider


class DesignerAgent(Agent):
    name = "designer"
    system_prompt = "You design things."


class ExplorerAgent(Agent):
    name = "explorer"
    system_prompt = "You explore."


def script_one() -> tuple[Agent, Agent]:
    """First 'script' — owns the root session and spawns one explorer."""
    provider = OpenAIProvider(client=MagicMock())
    parent = DesignerAgent(session_id="user-42", provider=provider)
    child = parent.spawn(ExplorerAgent, suffix="east")
    return parent, child


def script_two() -> None:
    """Second 'script' — never touches the agents directly, only the registry."""
    print("From script_two, the registry sees:")
    for sid in registry.list_active():
        info = registry.session_info(sid)
        assert info is not None
        print(
            f"  {sid:40s} agent={info['agent_type']:10s} "
            f"refs={info['ref_count']} msgs={info['message_count']}"
        )

    # Hierarchical operations are pure prefix scans.
    descendants = registry.descendants("user-42", include_root=True)
    print(f"\nWhole subtree under 'user-42':\n  {descendants}")

    # Cancel the whole subtree — any agent inside it will return [Cancelled]
    # on its next loop iteration.
    flagged = registry.cancel_subtree("user-42")
    print(f"\nCancellation broadcast: {flagged} sessions flagged.")


def main() -> None:
    parent, child = script_one()
    try:
        script_two()
        # Both agents now see their own session as cancelled — verified through
        # their handle, which is the same record the global registry exposes.
        assert parent._session_handle.is_cancelled
        assert child._session_handle.is_cancelled
        print("\nDone — parent and child both report cancelled.")
    finally:
        parent.close()
        child.close()


if __name__ == "__main__":
    main()
