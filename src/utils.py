# -*- coding: utf-8 -*-

"""
Module for getting data from https://github.com/ishaberry/Covid19Canada
"""


# Built-ins
from pathlib import Path

# Other
import pandas as pd

# Current file path
cur_dir = Path(__file__).parent


def get_covid_data(
    type: str, level: str = "canada", preprocess: bool = False
) -> pd.DataFrame:
    """
    Gets up to date Canada covid data from https://github.com/ishaberry/Covid19Canada

    Args:
        type (str): Type of data to retrieve. Options are active, cases, mortality, recovered, and testing
        level (str, optional): Level of data to retrieve. Options are prov or canada. Defaults to "canada".
        preprocess (bool, optional): Whether to clean errors in data. Defaults to False

    Returns:
        pd.DataFrame: Covid19 time series data
    """
    repo_url = "https://raw.githubusercontent.com/ishaberry/Covid19Canada/master"
    data_url = f"{repo_url}/timeseries_{level}/{type}_timeseries_{level}.csv"
    covid_data = pd.read_csv(data_url)

    # Convert date to date type
    format = "%d-%m-%Y"
    date_col = covid_data.filter(regex="^date").columns[0]
    covid_data = covid_data.assign(
        **{date_col: (lambda x: pd.to_datetime(x[date_col], format=format).dt.date)}
    ).rename(columns={date_col: "date"})

    # Optionally removed negative recovered values
    if preprocess and type == "recovered":
        covid_data = covid_data.assign(recovered=lambda x: x["recovered"].clip(lower=0))

    return covid_data


def get_all_covid_data(level: str = "canada", preprocess: bool = False) -> pd.DataFrame:
    """
    Gets all covid data and variables from https://github.com/ishaberry/Covid19Canada

    Args:
        level (str, optional): Level of data to retrieve either prov or canada. Defaults to "canada".
        preprocess (bool, optional): Whether to clean errors in data. Defaults to False

    Returns:
        pd.DataFrame: Covid19 time series data
    """
    # Read in data
    cases_data = get_covid_data(type="cases", level=level, preprocess=preprocess)
    active_cases_data = get_covid_data(
        type="active", level=level, preprocess=preprocess
    )
    mortality_data = get_covid_data(
        type="mortality", level=level, preprocess=preprocess
    )
    recovered_data = get_covid_data(
        type="recovered", level=level, preprocess=preprocess
    )

    # Province population data
    prov_map = {
        "British Columbia": "BC",
        "Newfoundland and Labrador": "NL",
        "Northwest Territories": "NWT",
        "Prince Edward Island": "PEI",
    }

    province_populations = (
        pd.read_csv(cur_dir / "../data/canada_prov_population.csv")
        .rename(columns={"GEO": "province", "VALUE": "population"})
        .replace({"province": prov_map})
        .loc[:, ["province", "population"]]
    )

    # Select columns of dataframes to be merged with
    recovered_data = recovered_data.loc[:, ["province", "date", "recovered"]]
    mortality_data = mortality_data.loc[:, ["province", "date", "deaths"]]
    cases_data = cases_data.loc[:, ["province", "date", "cases"]]

    # Preprocessing
    all_covid_data = (
        active_cases_data
        # Merge deaths and recovered data
        .merge(mortality_data, how="left", on=["province", "date"])
        .merge(recovered_data, how="left", on=["province", "date"])
        .merge(cases_data, how="left", on=["province", "date"])
        .fillna(0)
        # Turn floats back to int
        .assign(
            deaths=lambda x: x["deaths"].astype(int),
            recovered=lambda x: x["recovered"].astype(int),
        )
        # Remove non province data
        .query('province != "Repatriated"')
        # Add population data per province
        .merge(province_populations, how="left", on="province")
        # Add transformed variables
        .assign(
            removed=lambda x: x["recovered"] + x["deaths"],
            cumulative_removed=lambda x: x["cumulative_recovered"]
            + x["cumulative_deaths"],
            susceptible=lambda x: x["population"] - x["cumulative_cases"],
            percent_susceptible=lambda x: x["susceptible"] / x["population"],
        )
    )

    return all_covid_data