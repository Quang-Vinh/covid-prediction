# -*- coding: utf-8 -*-

"""
Module for getting data from https://github.com/ishaberry/Covid19Canada
"""


# Other
import pandas as pd


def get_covid_data(type: str, level: str = "canada") -> pd.DataFrame:
    """
    Gets up to date Canada covid data from https://github.com/ishaberry/Covid19Canada

    Args:
        type (str): Type of data to retrieve. Options are active, cases, mortality, recovered, and testing
        level (str, optional): Level of data to retrieve. Options are prov or canada. Defaults to "canada".

    Returns:
        pd.DataFrame: Covid19 time series data
    """
    repo_url = "https://raw.githubusercontent.com/ishaberry/Covid19Canada/master"
    data_url = f"{repo_url}/timeseries_{level}/{type}_timeseries_{level}.csv"
    covid_data = pd.read_csv(data_url)
    return covid_data


def get_all_covid_data(level: str = "canada") -> pd.DataFrame:
    """
    Gets all covid data and variables from https://github.com/ishaberry/Covid19Canada

    Args:
        level (str, optional): Level of data to retrieve either prov or canada. Defaults to "canada".

    Returns:
        pd.DataFrame: Covid19 time series data
    """
    # Read in data
    cases_data = get_covid_data(type="cases", level=level)
    active_cases_data = get_covid_data(type="active", level=level)
    mortality_data = get_covid_data(type="mortality", level=level)
    recovered_data = get_covid_data(type="recovered", level=level)

    # Province population data
    prov_map = {
        "British Columbia": "BC",
        "Newfoundland and Labrador": "NL",
        "Northwest Territories": "NWT",
        "Prince Edward Island": "PEI",
    }

    province_populations = (
        pd.read_csv("../data/canada_prov_population.csv")
        .rename(columns={"GEO": "province", "VALUE": "population"})
        .replace({"province": prov_map})
        .loc[:, ["province", "population"]]
    )

    # Preprocessing dataframes to be merged
    recovered_data = recovered_data.rename(columns={"date_recovered": "date"}).loc[
        :, ["province", "date", "recovered"]
    ]
    mortality_data = mortality_data.rename(columns={"date_death_report": "date"}).loc[
        :, ["province", "date", "deaths"]
    ]
    cases_data = cases_data.rename(columns={"date_report": "date"}).loc[
        :, ["province", "date", "cases"]
    ]

    # Preprocessing
    format = "%d-%m-%Y"
    all_covid_data = (
        active_cases_data.rename(columns={"date_active": "date"})
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
        # Format date and remove non province
        .assign(
            date=lambda x: pd.to_datetime(x["date"], format=format).dt.date,
        )
        .query('province != "Repatriated"')
        # Add population data per province
        .merge(province_populations, how="left", on="province")
        # Add transformed variables
        .assign(
            removed=lambda x: x["recovered"] + x["deaths"],
            susceptible=lambda x: x["population"] - x["cumulative_cases"],
            percent_susceptible=lambda x: x["susceptible"] / x["population"],
        )
    )

    return all_covid_data