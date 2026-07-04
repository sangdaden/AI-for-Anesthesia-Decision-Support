import pandas as pd
import plotly.graph_objects as go
from adaptivedose.dashboard import charts

def _case(cid=1):
    return pd.DataFrame({
        "caseid": [cid] * 3,
        "time_sec": [0, 10, 20],
        "bis": [45.0, 50.0, 55.0],
        "map": [70.0, 72.0, 68.0],
        "propofol_rate": [4.0, 5.0, 5.0],
        "remifentanil_rate": [3.0, 3.0, 4.0],
    })

def test_case_timeseries_figure_has_traces_per_signal():
    fig = charts.case_timeseries_figure(_case())
    assert isinstance(fig, go.Figure)
    names = {t.name for t in fig.data}
    assert {"BIS", "MAP", "propofol_rate", "remifentanil_rate"} <= names

def test_distribution_figure_is_histogram():
    fig = charts.distribution_figure(pd.Series([1.0, 2.0, 2.0, 3.0]), "bis")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "histogram"

def test_compare_bis_figure_one_trace_per_case():
    fig = charts.compare_bis_figure({1: _case(1), 2: _case(2)})
    assert len(fig.data) == 2
    assert {t.name for t in fig.data} == {"case 1", "case 2"}
