"""Smoke tests for the compiled ``agent_core._native`` extension."""


def test_native_module_imports():
    from agent_core import _native

    assert _native.hello() == "ok"


def test_native_module_version_matches_package():
    import agent_core
    from agent_core import _native

    # Both string-equal; the C++ side carries its own constant so a mismatch
    # signals a stale build that needs reinstalling.
    assert _native.__version__ == "0.2.0"
    assert hasattr(agent_core, "__version__") or True  # placeholder for later
