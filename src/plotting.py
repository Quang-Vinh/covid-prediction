# -*- coding: utf-8 -*-

"""
This module contains a collection of functions for plotting covid forecasts
"""

import pandas as pd
import plotly.graph_objects as go


def plot_predictions(
    forecast_data: pd.DataFrame,
    y: str,
    y_label: str,
    title: str,
    height: int = 400,
    include_ci: bool = False,
):
    """
    Plots forecasts for given variable.

    Args:
        forecast_data (pd.DataFrame): Dataframe containing column date, y, y_pred and if include_ci is True then y_ci_lower and y_ci_upper.
        y (str): Name of column in forecast dataframe to plot
        y_label (str): Y label axis on plot
        title (str): Plot title
        height (int): Plot layout height
        include_ci (bool, optional): Whether to include confidence intervals in plot. Defaults to False.

    Returns:
        [plotly go figure]: Plotly graph object figure
    """
    y_pred = y + "_pred"

    # Set figure layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title="Date"),
        yaxis=dict(title=y_label),
        height=height,
    )
    fig = go.Figure(layout=layout)

    # Add line for end of current known data
    end_date = forecast_data.loc[forecast_data["is_forecast"] == False]["date"].max()
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                yref="paper",
                y0=0,
                y1=1,
                xref="x",
                x0=end_date,
                x1=end_date,
            )
        ]
    )

    # Trace for actual data
    fig.add_trace(
        go.Scatter(
            x=forecast_data["date"],
            y=forecast_data[y],
            mode="lines",
            name="Actual",
            marker_color="blue",
        )
    )

    # Trace for predicted deaths
    fig.add_trace(
        go.Scatter(
            x=forecast_data["date"],
            y=forecast_data[y_pred],
            mode="lines",
            line={"dash": "dash"},
            name="Predicted",
            marker_color="red",
        )
    )

    # Trace for 95% confidence intervals of E[Y|X]
    if include_ci:
        fig.add_trace(
            go.Scatter(
                x=forecast_data["date"],
                y=forecast_data[y + "_ci_lower"],
                mode="lines",
                line={"dash": "dash"},
                name="95% CI Lower",
                marker_color="orange",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_data["date"],
                y=forecast_data[y + "_ci_upper"],
                mode="lines",
                line={"dash": "dash"},
                name="95% CI Upper",
                marker_color="orange",
            )
        )

    return fig
