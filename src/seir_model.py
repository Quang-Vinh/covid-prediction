# -*- coding: utf-8 -*-

"""
Module for SEIR model
"""


# Built ins
from datetime import timedelta
from pathlib import Path
import sys

# Other
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Add parent path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.absolute()
sys.path.append(str(parent_dir))


class SEIRModel:
    """
    SEIR model

    Args:
        lam (float, optional): Lambda value used to combine removed error and infected error. Total error = lambda * removed error + (1 - lambda) * infected error. Defaults to 0.5.

    Attributes:
        S_0 (int): Initial susceptible population
        E_0 (int): Initial exposed population
        I_0 (int): Initial infected population
        R_0 (int): Initial removed population
        N (int): Region population
        province (str): Province name
        X_original (pd.DataFrame): Original dataframe used to fit model
        optimal_params (list): List of parameters for SEIR model in form [alpha, beta, gamma]
    """

    def __init__(self, lam: float = 0.5):
        self.lam = lam
        self.S_0 = None
        self.E_0 = None
        self.I_0 = None
        self.R_0 = 0
        self.N = None
        self.province = None
        return

    def seir_odes(self, t: int, y: list, params: list) -> list:
        """
        Ordinary differential equations of SEIR model.

        Args:
            t (int): Current time t
            y (list): Current state values
            params (list): SEIR parameters

        Returns:
            list: List of ODEs for all values in y
        """
        # Get SEIR parameters
        S, E, I, R = y
        alpha, beta, gamma = params

        # Calculate ODEs
        dSdt = -beta * I * S / self.N
        dEdt = beta * I * S / self.N - alpha * E
        dIdt = alpha * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    def seir_solution(self, params: list, dates: pd.Series) -> pd.DataFrame:
        """
        Calculate solution to initial value problem using initial states and seir odes.

        Args:
            params (list): SEIR parameters
            dates (pd.Series): Series of all dates to get values for

        Returns:
            pd.DataFrame: Dataframe containing solutions for values of all compartments S, E, I, R for given dates
        """
        # Run initial value solver
        size = len(dates)
        solution = solve_ivp(
            self.seir_odes,
            y0=[self.S_0, self.E_0, self.I_0, self.R_0],
            t_span=[0, size],
            t_eval=np.arange(0, size),
            args=[params],
        )

        # Create results dataframe
        y = solution.y
        results = pd.DataFrame(
            {
                "date": dates,
                "susceptible": y[0],
                "exposed": y[1],
                "infected": y[2],
                "removed": y[3],
            }
        )

        return results

    def loss(self, params: list, X: pd.DataFrame) -> float:
        """
        Loss function used for SEIR model. Combines the root mean squared error loss from the removed and infected compartment.
        The loss is calculated as loss = lambda * removed_rmse + (1 - lambda) * infected_rmse

        Args:
            params (list): SEIR parameters
            X (pd.DataFrame): Dataframe for region data containing columns 'date', 'cumulative_deaths', and 'active_cases'

        Returns:
            float: Combined removed and infected rmse
        """
        results = self.seir_solution(params=params, dates=X["date"])
        removed_rmse = np.sqrt(
            np.mean((results["removed"] - X["cumulative_deaths"]) ** 2)
        )
        infected_rmse = np.sqrt(np.mean((results["infected"] - X["active_cases"]) ** 2))
        total_rmse = self.lam * removed_rmse + (1 - self.lam) * infected_rmse
        return total_rmse

    def fit(self, X: pd.DataFrame):
        """
        Estimates best set of SEIR parameters for given data.

        Args:
            X (pd.DataFrame): DataFrame of region infection data. Contains columns 'province', 'population', 'active_cases', 'date', 'cumulative_deaths', 'active_cases'
        """
        self.X_original = X.copy()

        # Initialize values
        self.province = X["province"].iloc[0]
        self.N = X["population"].iloc[0]
        self.I_0 = X["active_cases"].iloc[0]
        self.E_0 = self.I_0
        self.S_0 = self.N - self.E_0 - self.I_0

        # Find optimal parameters
        x0 = 3 * [0.5]
        bounds = 3 * [(1e-4, 50)]
        optimal = minimize(self.loss, x0=x0, bounds=bounds, method="L-BFGS-B", args=(X))
        self.optimal_params = optimal.x
        return

    def forecast(self, h: int = 21) -> pd.DataFrame:
        """
        Calculates forecasted values for h time steps

        Args:
            h (int, optional): Number of days to forecast. Defaults to 21.
        """
        # Get dates to forecast
        start_date = self.X_original["date"].iloc[0]
        end_date = self.X_original["date"].iloc[-1]
        end_date_forecast = end_date + timedelta(days=h)
        dates = pd.Series(
            pd.date_range(start_date, end_date_forecast, name="date")
        ).dt.date

        # Get forecasted values and process columns
        forecasts = self.seir_solution(params=self.optimal_params, dates=dates)
        forecasts["province"] = self.province
        forecasts["is_forecast"] = forecasts["date"] > end_date
        forecasts.rename(
            columns={
                "susceptible": "susceptible_pred",
                "exposed": "active_exposed_pred",
                "infected": "active_cases_pred",
                "removed": "cumulative_deaths_pred",
            },
            inplace=True,
        )

        # Merge forecasted data on the original X dataframe that was fitted on
        forecasts = forecasts.merge(
            self.X_original, how="left", on=["province", "date"]
        )

        return forecasts