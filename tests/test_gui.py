import pytest


def test_gui_imports():
    """Ensure gui main does not fail on import when executed."""
    try:
        from gui import main
        assert main is not None
    except ImportError as e:
        pytest.fail(f"Could not import gui.main due to: {e}")
