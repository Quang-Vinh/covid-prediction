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
    include_ci: bool = False,
):
    """
    Plots forecasts for given variable.

    Args:
        forecast_data (pd.DataFrame): Dataframe containing column date, y, y_pred and if include_ci is True then y_ci_lower and y_ci_upper.
        y (str): Name of column in forecast dataframe to plot
        y_label (str): Y label axis on plot
        title (str): Plot title
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
        height=600,
    )
    fig = go.Figure(layout=layout)

    # Add line for end of current known data
    end_date = forecast_data.query("not is_forecast")["date"].max()
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


def plot_mortality_predictions(mortality_pred: pd.DataFrame, title: str = ""):
    """
    Creates a time series plotly plot of the actual and predicted cumulative deaths.

    Args:
        mortality_pred (pd.DataFrame): Dataframe containing actual and predicted mortalities.
                                       Contains columns ['is_forecast', 'date_death_report', 'cumulative_deaths', 'cumulative_deaths_pred']
        title (str): Title of plot. Defaults to ''

    Returns:
        [go.Figure]: Plotly figure
    """

    # Set figure layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title="Date"),
        yaxis=dict(title="Cumulative Deaths"),
        height=600,
    )
    fig = go.Figure(layout=layout)

    # Add line for end of current known data
    end_date = mortality_pred.query("not is_forecast")["date_death_report"].max()
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

    # Trace for actual deaths
    fig.add_trace(
        go.Scatter(
            x=mortality_pred["date_death_report"],
            y=mortality_pred["cumulative_deaths"],
            mode="lines",
            name="Actual",
        )
    )

    # Trace for predicted deaths
    fig.add_trace(
        go.Scatter(
            x=mortality_pred["date_death_report"],
            y=mortality_pred["cumulative_deaths_pred"],
            mode="lines",
            line={"dash": "dash"},
            name="Predicted",
        )
    )

    return fig