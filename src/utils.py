import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fixed_params import DATE_STR_FMT, DASH_REGIONS


def inv_sigmoid(shift=0, a=1, b=1, c=0):
    """Returns a inverse sigmoid function based on the parameters."""
    return (
        lambda x: b * np.exp(-(a * (x - shift))) / (1 + np.exp(-(a * (x - shift)))) + c
    )


def str_to_date(date_str, fmt=DATE_STR_FMT):
    """Convert string date to datetime object."""
    return datetime.datetime.strptime(date_str, fmt).date()


def date_range(start_date, end_date, interval=1, str_fmt=DATE_STR_FMT):
    """Returns range of datetime dates from start_date to end_date (inclusive).

    start_date/end_date can be either str (with format str_fmt) or datetime objects.
    """

    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, str_fmt).date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, str_fmt).date()
    return [
        start_date + datetime.timedelta(n)
        for n in range(0, (end_date - start_date).days + 1, interval)
    ]


def remove_space_region(region):
    return region.replace(" ", "-")


def add_space_region(region):
    if region in DASH_REGIONS:
        return region


def plot_predictions(mortality_pred: pd.DataFrame, title: str = ""):
    """
    Creates a time series plotly plot of the actual and predicted cumulative deaths.

    Args:
        mortality_pred (pd.DataFrame): Dataframe containing actual and predicted mortalities.
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