from __future__ import annotations
from typing import Dict
import pandas as pd
import plotly.graph_objects as go

def case_timeseries_figure(df: pd.DataFrame) -> go.Figure:
    """Multi-axis time-series: BIS (with target band) + MAP + drug rates."""
    t = df["time_sec"] / 60.0
    fig = go.Figure()
    if "bis" in df.columns:
        fig.add_trace(go.Scatter(x=t, y=df["bis"], name="BIS", yaxis="y1"))
    if "map" in df.columns:
        fig.add_trace(go.Scatter(x=t, y=df["map"], name="MAP", yaxis="y2"))
    for drug in ("propofol_rate", "remifentanil_rate"):
        if drug in df.columns:
            fig.add_trace(go.Scatter(x=t, y=df[drug], name=drug, yaxis="y3"))
    fig.add_hrect(y0=40, y1=60, line_width=0, fillcolor="green", opacity=0.08,
                  yref="y1")
    fig.update_layout(
        xaxis=dict(title="time (min)"),
        yaxis=dict(title="BIS", range=[0, 100]),
        yaxis2=dict(title="MAP (mmHg)", overlaying="y", side="right"),
        yaxis3=dict(title="rate", overlaying="y", side="right", position=0.95,
                    showgrid=False),
        legend=dict(orientation="h"),
        margin=dict(l=40, r=60, t=30, b=40),
    )
    return fig

def distribution_figure(values: pd.Series, title: str) -> go.Figure:
    fig = go.Figure(data=[go.Histogram(x=values, nbinsx=40)])
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=40, b=40))
    return fig

def compare_bis_figure(frames: Dict[int, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for cid, df in frames.items():
        fig.add_trace(go.Scatter(x=df["time_sec"] / 60.0, y=df["bis"],
                                 name=f"case {cid}"))
    fig.add_hrect(y0=40, y1=60, line_width=0, fillcolor="green", opacity=0.08)
    fig.update_layout(xaxis=dict(title="time (min)"),
                      yaxis=dict(title="BIS", range=[0, 100]),
                      legend=dict(orientation="h"),
                      margin=dict(l=40, r=20, t=30, b=40))
    return fig
