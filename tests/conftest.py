import plotly.graph_objects as go
import pytest


@pytest.fixture
def sample_figure():
    """Return a simple Plotly figure for smoke tests."""
    figure = go.Figure(
        data=[
            go.Scatter(
                x=[1, 2, 3],
                y=[4, 5, 6],
                mode="lines+markers",
                name="series_a",
            )
        ],
        layout=go.Layout(title="Sample figure"),
    )
    return figure
