import json

from vla_trial.config import RENDER_SPIKE_JSON
from vla_trial.spike.bench_render import bench_render


def test_bench_render_produces_a_number():
    result = bench_render(n_frames=20, width=256, height=256)
    assert result["fps"] > 0
    assert result["n_frames"] == 20
    assert RENDER_SPIKE_JSON.is_file()
    assert json.loads(RENDER_SPIKE_JSON.read_text())["fps"] == result["fps"]
