import pytest


@pytest.mark.unit
def test_import_anim_ml() -> None:
    import anim_ml  # noqa: F401
