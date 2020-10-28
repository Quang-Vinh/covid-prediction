# -*- coding: utf-8 -*-

"""
This module contains a collection of functions for plotting covid forecasts
"""

import pandas as pd
import plotly.graph_objects as go


def plot_predictions():
    return


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