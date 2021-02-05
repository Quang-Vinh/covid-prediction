# -*- coding: utf-8 -*-

""" 
SEIR model using YGG simulator.
"""

# TODO: Organize variable_params better instead of hard coding variables and order
# TODO: Remove predict() function from simulation.py eventually

# Built-in
import copy
from datetime import date, timedelta
from pathlib import Path
import sys

# Other
from scipy.optimize import minimize, differential_evolution
import numpy as np
import pandas as pd

# Custom
cur_dir = Path(__file__).parent
ygg_simulator_path = str(cur_dir / "../ygg_seir_simulator")
if ygg_simulator_path not in sys.path:
    sys.path.append(ygg_simulator_path)

from ygg_seir_simulator.fixed_params import *
from ygg_seir_simulator.region_model import RegionModel
from ygg_seir_simulator.learn_simulation import *
from ygg_seir_simulator.simulation import run, predict


# Set YGG simulator region values
skip_hospitalizations = None
quarantine_perc = 0
quarantine_effectiveness = -1
best_params_type = "mean"
country = "Canada"
region = "ALL"
best_params_dir = cur_dir / "../models/best_params/latest"

# YGG simulator parameters
variable_params = [
    "INITIAL_R_0",
    "LOCKDOWN_R_0",
    "RATE_OF_INFLECTION",
    "LOCKDOWN_FATIGUE",
    "DAILY_IMPORTS",
    "MORTALITY_RATE",
    # "REOPEN_DATE",
    "REOPEN_SHIFT_DAYS",
    "REOPEN_R",
    "REOPEN_INFLECTION",
    "POST_REOPEN_EQUILIBRIUM_R",
    "FALL_R_MULTIPLIER",
]


def predict(region_model: RegionModel, mortality_data: pd.DataFrame) -> pd.DataFrame:
    """
    Runs SEIR simulator and produces predictions given a region model

    Args:
        region_model (RegionModel): Region model instance with initialized parameters
        mortality_data (pd.DataFrame): Dataframe of mortality data for given region

    Returns:
        pd.DataFrame: Data frame which includes death forecasts
    """
    # Run SEIR simulation and create dataframe of predictions
    dates, infections, hospitalizations, deaths = run(region_model)
    cumulative_deaths_pred = deaths.cumsum()
    mortality_pred = pd.DataFrame(
        {
            "date": dates,
            "infections_pred": infections,
            "hospitalizations_pred": hospitalizations,
            "cumulative_deaths_pred": cumulative_deaths_pred,
            "deaths_pred": deaths,
        }
    )

    # Cut off days before first death reports
    start_date = mortality_data["date"].min()
    mortality_pred = mortality_pred.query("date >= @start_date")

    # Combine predictions with actual data
    mortality_pred = mortality_pred.merge(mortality_data, how="left", on="date")
    mortality_pred.loc[:, "province"] = mortality_data.iloc[0]["province"]

    # Add flag to check if date is a forecast of not
    mortality_pred["is_forecast"] = mortality_pred["cumulative_deaths"].isnull()

    return mortality_pred


def loss(
    variable_params: Tuple,
    params: dict,
    region_model: RegionModel,
    mortality_data: pd.DataFrame,
) -> float:
    """
    Helper function for calculting root mean squared error for SEIR model.
    The variable params should be (initial_r_0, lockdown_r_0, mortality_rate, daily_imports)
    """
    # Initialize parameters for region model
    (
        params["INITIAL_R_0"],
        params["LOCKDOWN_R_0"],
        params["RATE_OF_INFLECTION"],
        params["LOCKDOWN_FATIGUE"],
        params["DAILY_IMPORTS"],
        params["MORTALITY_RATE"],
        # params["REOPEN_DATE"],
        params["REOPEN_SHIFT_DAYS"],
        params["REOPEN_R"],
        params["REOPEN_INFLECTION"],
        params["POST_REOPEN_EQUILIBRIUM_R"],
        params["FALL_R_MULTIPLIER"],
    ) = variable_params
    params_tups = tuple(params.items())
    region_model_copy = copy.deepcopy(region_model)
    region_model_copy.init_params(params_tups)

    # Run SEIR simulation
    mortality_pred = predict(region_model_copy, mortality_data)

    # Calculate rmse. Use only projections for known dates and not future forecasts
    mortality_pred = mortality_pred.query("cumulative_deaths == cumulative_deaths")
    deaths_true = mortality_pred["cumulative_deaths"]
    deaths_pred = mortality_pred["cumulative_deaths_pred"]

    rmse = np.sqrt(np.mean((deaths_true - deaths_pred) ** 2))

    return rmse


class SEIRYGGForecaster:
    """
    SEIR model for forecasting new deaths and infections by province in Canada. Uses the YGG SEIR simulator to simulate the mathematical model (with extra parameters)
    and uses optimizatio methods in order to estimate the SEIR model parameters.


    Args:
        province (str): Canadian province
        population (int): Population of region
        method (str, optional): Optimization method for SEIR parameters. Defaults to "L-BFGS-B".
        verbose (bool, optional): Display messages when training or not. Defaults to False.

    Attributes:
        X_original (pd.DataFrame): Pandas dataframe used to train the model on fit()
        region_model (RegionModel): RegionModel YGG SEIR model simulator
        optimal (): Result from running optimization methods
    """

    def __init__(
        self,
        province: str,
        population: int,
        method: str = "L-BFGS-B",
        verbose: bool = False,
    ):
        # Input check
        if method not in ("L-BFGS-B", "differential_evolution"):
            raise Exception("Invalid method option")

        self.method = method
        self.province = province
        self.population = population
        self.verbose = verbose

        return

    def fit(self, X: pd.DataFrame):
        """
        Estimate SEIR model parameters for given data

        Args:
            X (pd.DataFrame): Epidemic data with columns date, cumulative_deaths
        """
        self.X_original = X

        # Initialize region model YGG simulator
        first_date = X["date"].min() - timedelta(days=DAYS_BEFORE_DEATH)
        projection_end_date = X["date"].max()

        self.region_model = RegionModel(
            country_str=country,
            region_str=region,
            subregion_str=self.province,
            first_date=first_date,
            projection_create_date=first_date + timedelta(days=1),
            projection_end_date=projection_end_date,
            region_params={"population": self.population},
            compute_hospitalizations=(not skip_hospitalizations),
        )

        # Load best parameters from initial YGG grid search results
        if self.province == "BC":
            prov = "British-Columbia"
        else:
            prov = self.province
        region_param, self.params_dict = load_best_params_province(
            best_params_dir, prov
        )

        # Set initial values and bounds for each parameter for optimization algorithm
        x0 = [0] * len(variable_params)
        bounds = []

        for i, variable_param in enumerate(variable_params):
            x0[i] = self.params_dict.pop(variable_param)

            # if variable_param == "RATE_OF_INFLECTION":
            #     bound = [(1e-6, x0[i])]
            # elif variable_param == "LOCKDOWN_FATIGUE":
            #     bound = [(0.5, x0[i])]
            # elif variable_param == "DAILY_IMPORTS":
            #     bound = [(x0[i] * 0.9, x0[i] * 5)]
            # elif variable_param == "MORTALITY_RATE":
            #     bound = [(5e-3, x0[i] * 2.5)]
            # elif variable_param == "REOPEN_INFLECTION":
            #     bound = [(0.1, x0[i])]
            # elif variable_param == "REOPEN_SHIFT_DAYS":
            #     bound = [(x0[i] * 0.5, x0[i] * 1.5)]
            # else:
            #     bound = [(x0[i] * 0.7, x0[i] * 1.2)]

            if variable_param == "RATE_OF_INFLECTION":
                bound = [(1e-6, 1)]
            elif variable_param == "LOCKDOWN_FATIGUE":
                bound = [(0.5, 1.5)]
            elif variable_param == "DAILY_IMPORTS":
                bound = [(0, 1000)]
            elif variable_param == "MORTALITY_RATE":
                bound = [(1e-6, 0.2 - 1e-6)]
            elif variable_param == "REOPEN_INFLECTION":
                bound = [(0.1, 0.6)]
            elif variable_param == "REOPEN_SHIFT_DAYS":
                bound = [(-28, 28)]
            else:
                bound = [(x0[i] * 0.7, x0[i] * 1.2)]

            bounds += bound

        # Estimate parameters using given method
        if self.method == "L-BFGS-B":
            self.optimal = minimize(
                loss,
                x0=x0,
                args=(self.params_dict.copy(), self.region_model, X),
                method="L-BFGS-B",
                bounds=bounds,
                options={"disp": self.verbose},
            )

        elif self.method == "differential_evolution":
            self.optimal = differential_evolution(
                loss,
                bounds,
                args=(self.params_dict.copy(), self.region_model, X),
                strategy="best1bin",
                maxiter=200,
                popsize=50,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=None,
                callback=None,
                disp=self.verbose,
                polish=True,
                init="latinhypercube",
            )

        return

    def forecast(self, h: int = 21) -> pd.DataFrame:
        """
        Forecast deaths and infections

        Args:
            h (int, optional): Number of days to forecast. Defaults to 21.

        Returns:
            pd.DataFrame: Dataframe containing forecasts
        """
        # Set simulator projection end date for h forecasts
        projection_end_date = self.X_original["date"].max() + timedelta(days=h)
        self.region_model.set_projection_end_date(projection_end_date)

        # Initialize parameters for region_model using estimated params
        params = self.params_dict.copy()
        (
            params["INITIAL_R_0"],
            params["LOCKDOWN_R_0"],
            params["RATE_OF_INFLECTION"],
            params["LOCKDOWN_FATIGUE"],
            params["DAILY_IMPORTS"],
            params["MORTALITY_RATE"],
            # params["REOPEN_DATE"],
            params["REOPEN_SHIFT_DAYS"],
            params["REOPEN_R"],
            params["REOPEN_INFLECTION"],
            params["POST_REOPEN_EQUILIBRIUM_R"],
            params["FALL_R_MULTIPLIER"],
        ) = self.optimal.x
        params_tups = tuple(params.items())
        self.region_model.init_params(params_tups)

        forecasts = predict(self.region_model, self.X_original)
        forecasts["province"] = self.province

        return forecasts
