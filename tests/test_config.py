import pytest

from binpick.config import load_config
from conftest import write_patched


def test_default_config_counts():
    c = load_config()
    assert c.parts.n_boxes == 15 and c.parts.n_cylinders == 5
    assert c.cup.diameter == pytest.approx(0.044)
    assert c.tray.rows * c.tray.cols >= 20
    assert sum(m.prob for m in c.parts.materials.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "patch,field",
    [
        ({"parts": {"box_height": [0.07, 0.02]}}, "box_height"),
        ({"parts": {"mass_min": 3.0}}, "mass_min"),
        ({"parts": {"materials": {"plastic": {"prob": 0.9}}}}, "materials"),
    ],
)
def test_invalid_config_raises(tmp_path, patch, field):
    with pytest.raises(ValueError, match=field):
        load_config(write_patched(tmp_path, patch))
