# -*- coding: utf-8 -*-

"""
Script train a Poisson GAM model to forecast new cases and removed for each province

Arguments: h (int) - Number of days to forecast
"""


# Built ins
import argparse
from datetime import date, timedelta
import joblib
from pathlib import Path
import sys
from timeit import default_timer as timer

# Other
import pandas as pd

# Owned
cur_dir = Path(__file__).resolve().parent
src_path = str(cur_dir.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from src.stem_poisson_gam import preprocess_data, StemPoissonRegressor
from src.utils import get_all_covid_data


# Paths to output results
output_path = cur_dir / "../output/"


def main(h: int):
    """
    Trains poisson models for each province and produces forecast to output folder

    Args:
        h (int): Number of days to forecast
    """
    # Get newest data
    covid_data = get_all_covid_data(level="prov", preprocess=True)

    # Remove outlier for quebec
    covid_data.loc[covid_data["removed"] == 23687, "removed"] = 1

    # Remove data before March 8
    remove_date = date(day=8, month=3, year=2020)
    covid_data = covid_data.query("date >= @remove_date")

    # Focus on only select provinces
    provinces = ["Alberta", "BC", "Manitoba", "Ontario", "Quebec", "Saskatchewan"]
    covid_data = covid_data.query("province in @provinces").reset_index(drop=True)

    # Preprocess all province data
    covid_data_preprocessed = preprocess_data(covid_data, drop_first_day=True)
    X = covid_data_preprocessed
    Y = covid_data_preprocessed[["province", "date", "cases", "removed"]]

    # Fit model and get forecasts for each province
    forecasts = pd.DataFrame()
    models = {}

    for province in provinces:
        # Get province data
        province_data = covid_data_preprocessed.query("province == @province")
        X = province_data
        Y = province_data[["province", "date", "cases", "removed"]]

        # Fit model
        model = StemPoissonRegressor()
        model.fit(X, Y)

        # Get forecasts
        province_forecasts = model.forecast(h=h)
        province_forecasts = province_forecasts.merge(
            province_data, how="left", on=["province", "date"]
        )
        forecasts = pd.concat([forecasts, province_forecasts], ignore_index=True)

        # Save province model
        models[province] = model

    # Save results to output
    current_date = date.today().strftime("%d-%m-%Y")
    joblib.dump(models, output_path / f"poisson_models_{current_date}.pkl")
    forecasts.to_csv(output_path / f"poisson_forecasts_{current_date}.csv", index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, default=40, help="Number of days to forecast")
    args = parser.parse_args()

    try:
        start = timer()
        main(h=args.h)
        end = timer()
        elapsed_time = timedelta(seconds=round(end - start))
        print(
            f"Finished training Poisson regression models  -  elapsed time {elapsed_time}"
        )
    except Exception as e:
        print(f"Error training - {e}")
