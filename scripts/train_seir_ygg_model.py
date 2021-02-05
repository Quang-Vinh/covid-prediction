# -*- coding: utf-8 -*-

"""
Script train a SEIR model to forecast new cases and removed for each province

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

from src.seir_ygg_model import SEIRYGGForecaster
from src.utils import get_all_covid_data, province_populations_dict


# Paths to output results
output_path = cur_dir / "../output/"


def main(h: int):
    """
    Trains poisson models for each province and produces forecast to output folder

    Args:
        h (int): Number of days to forecast
    """

    # Get newest data
    covid_data = get_all_covid_data(level="prov")

    # Remove data before March 8
    remove_date = date(day=8, month=3, year=2020)
    covid_data = covid_data.query("date >= @remove_date")

    # Focus on only select provinces
    provinces = ["Alberta", "BC", "Ontario", "Quebec"]
    covid_data = covid_data.query("province in @provinces").reset_index(drop=True)

    # Fit model and get forecasts for each province
    forecasts = pd.DataFrame()
    models = {}

    for province in provinces:
        province_data = covid_data.query("province == @province")

        # Fit SEIR model
        model = SEIRYGGForecaster(
            method="differential_evolution",
            province=province,
            population=province_populations_dict[province],
            verbose=False,
        )
        model.fit(province_data)

        # Get forecasts
        province_forecasts = model.forecast(h=h)
        province_forecasts.loc[:, "active_cases_pred"] = province_forecasts[
            "infections_pred"
        ]
        forecasts = pd.concat([forecasts, province_forecasts], ignore_index=True)

        # Save province model
        models[province] = model

    # Save results to output
    current_date = date.today().strftime("%d-%m-%Y")
    joblib.dump(models, output_path / f"seir_ygg_model_{current_date}.pkl")
    forecasts.to_csv(
        output_path / f"seir_ygg_forecasts_{current_date}.csv", index=False
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, default=50, help="Number of days to forecast")
    args = parser.parse_args()

    try:
        start = timer()
        main(h=args.h)
        end = timer()
        elapsed_time = timedelta(seconds=round(end - start))
        print(f"Finished training SEIR models  -  elapsed time {elapsed_time}")
    except Exception as e:
        print(f"Error training - {e}")
