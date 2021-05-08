# -*- coding: utf-8 -*-

"""
Module for getting data from https://github.com/ishaberry/Covid19Canada and loading Twitter sentiment data. Also contains many other useful functions
"""


# Built-ins
from datetime import datetime
from pathlib import Path
from typing import List

# Other
import numpy as np
import pandas as pd

# File paths
cur_dir = Path(__file__).parent
data_dir = cur_dir / "../data"

# Dictionary to map population and twitter province names to COVID19 province names
prov_map = {
    "British Columbia": "BC",
    "Newfoundland and Labrador": "NL",
    "Northwest Territories": "NWT",
    "Prince Edward Island": "PEI",
}

# Province population data
province_populations = (
    pd.read_csv(data_dir / "canada_prov_population.csv")
    .rename(columns={"GEO": "province", "VALUE": "population"})
    .replace({"province": prov_map})
    .loc[:, ["province", "population"]]
)

province_populations_dict = province_populations.set_index("province").to_dict()[
    "population"
]

# Twitter data
twitter_file_names = [
    "CANADA_longitudinal_data_by_region_2020-01-01_to_2020-07-01.xlsx",
    "CANADA_longitudinal_data_by_region_2020-07-01_to_2021-03-12.xlsx",
]
twitter_dir = data_dir / "twitter_sentiment"
twitter_file_paths = [twitter_dir / file_name for file_name in twitter_file_names]


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up column names of dataframe

    Args:
        df (pd.DataFrame): Dataframe input

    Returns:
        pd.DataFrame: Dataframe with cleaned column names
    """
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    return df


def load_twitter_data() -> pd.DataFrame:
    """
    Load twitter data to dataframe

    Returns:
        [pd.DataFrame]: Dataframe containing twitter sentiment data
    """
    twitter_data = []

    # Loop through each excel file and sheet and concatenate them together
    for file_path in twitter_file_paths:
        for i in range(0, 5):
            topic_data = pd.read_excel(
                io=file_path,
                sheet_name=f"Topic {i + 1}",
                engine="openpyxl",
            )

            # Preprocess
            topic_data = clean_names(topic_data)
            topic_data = (
                topic_data.dropna(subset=["topic_num"])
                .rename(columns={"variable": "province"})
                .assign(date=lambda x: x["date"].dt.date)
                .replace({"province": prov_map})
            )

            twitter_data.append(topic_data)

    # Combine and remove duplicates
    twitter_data = pd.concat(twitter_data)
    twitter_data = twitter_data.drop_duplicates(
        subset=["topic_num", "date", "province"], keep="last"
    ).reset_index(drop=True)

    return twitter_data


def get_covid_data(
    type: str, level: str = "canada", preprocess: bool = False
) -> pd.DataFrame:
    """
    Gets up to date Canada covid data from https://github.com/ishaberry/Covid19Canada

    Args:
        type (str): Type of data to retrieve. Options are active, cases, mortality, recovered, testing, and vaccine_completion
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
    # Read in and merge all data types
    types = ["cases", "mortality", "recovered", "vaccine_completion"]
    all_covid_data = get_covid_data(type="active", level=level, preprocess=preprocess)
    for type in types:
        covid_data = get_covid_data(type=type, level="prov", preprocess=preprocess)
        cols_on = covid_data.columns.intersection(all_covid_data.columns).to_list()
        all_covid_data = all_covid_data.merge(covid_data, how="left", on=cols_on)

    # Preprocessing
    all_covid_data = (
        all_covid_data.fillna(0)
        # Remove non province data
        .query('province != "Repatriated"')
        # Add population data per province
        .merge(province_populations, how="left", on="province")
        # Add transformed variables
        .assign(
            # Removed individuals
            removed=lambda x: x["recovered"] + x["deaths"],
            cumulative_removed=lambda x: x["cumulative_recovered"]
            + x["cumulative_deaths"],
            # Susceptible individuals
            susceptible=lambda x: x["population"] - x["cumulative_cases"],
            percent_susceptible=lambda x: x["susceptible"] / x["population"],
            # Vaccine related
            percent_cvaccine=lambda x: x["cvaccine"] / x["population"],
            percent_cumulative_cvaccine=lambda x: x["cumulative_cvaccine"]
            / x["population"],
        )
    )

    return all_covid_data


def get_prov_gov_policies(province: str = None) -> pd.DataFrame:
    """
    Returns dataframe with government intervention and dates for given province

    Args:

        province (str): Province

    Returns:
        pd.DataFrame: Dataframe with government intervention
    """
    # Read and parse government intervention
    dateparse = lambda x: datetime.strptime(x, "%d-%m-%Y")
    prov_gov_policies = pd.read_csv(
        data_dir / "prov_gov_policies.csv", parse_dates=["date"], date_parser=dateparse
    )

    if province:
        prov_gov_policies = prov_gov_policies.query("province == @province").copy()

    # Sort by date
    prov_gov_policies.sort_values(by="date", axis=0, inplace=True)

    return prov_gov_policies


def sigmoid(x: float, a: float = 1, b: float = 1, shift: float = 0) -> float:
    """
    Sigmoid function represented by b * \frac{1}{1 + e^{-a * (x - shift)}}}

    Args:
        x (float): Input x
        a (float, optional): Rate of inflection. Defaults to 1.
        b (float, optional): Difference of lowest to highest value. Defaults to 1.
        shift (float, optional): Horizontal shift. Defaults to 0.

    Returns:
        float: sigmoid function at x
    """
    result = b * (1 / (1 + np.exp(-a * (x - shift))))
    return result


def combined_sigmoid(
    x: float,
    y_list: List[float],
    splits: List[float],
    a: float,
    bias: int = 0,
) -> float:
    """
    Sum of sigmoid functions.
    Args:
        x (float): Input x
        y_list (List[float]): Value where each sigmoid converges to
        splits (List[float]): Location of inflection point of each sigmoid function (shift)
        a (float, optional): Rate of inflection. Defaults to 1.
    Returns:
        [float]: Sum of sigmoids at x
    """
    result = y_list[0]

    # Loop through each set of parameters and compute sigmoid at x then add to result
    for i in range(1, len(y_list)):
        b = y_list[i] - y_list[i - 1]
        shift = splits[i - 1]
        result += sigmoid(x, a=a, b=b, shift=shift + bias)

    return result
