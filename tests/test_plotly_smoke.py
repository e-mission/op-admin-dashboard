import json

import plotly.io as pio


def test_plotly_figure_serializes(sample_figure):
    payload = pio.to_json(sample_figure)
    parsed = json.loads(payload)

    assert "data" in parsed
    assert "layout" in parsed
    assert len(parsed["data"]) == 1
    assert parsed["data"][0]["type"] == "scatter"


def test_plotly_figure_has_expected_trace(sample_figure):
    assert len(sample_figure.data) == 1
    assert sample_figure.data[0].type == "scatter"
    assert sample_figure.layout.title.text == "Sample figure"
