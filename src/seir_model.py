# -*- coding: utf-8 -*-

"""
Script ...
"""


# Built ins
import argparse
from datetime import timedelta
from pathlib import Path
import sys
from timeit import default_timer as timer

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
    [summary]
    """

    def __init__(self):
        self.S_0 = None
        self.E_0 = 2
        self.I_0 = 2
        self.R_0 = 0
        self.N = None
        self.province = None
        return

    def seir_odes(self, t: int, y: list, params: list) -> list:
        """
        [summary]

        Args:
            t (int): [description]
            y (list): [description]

        Returns:
            list: [description]
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

    def seir_solution(self, params: list, dates: pd.Series) -> np.ndarray:
        """
        [summary]

        Args:
            params (list): [description]

        Returns:
            np.ndarray: [description]
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
        [summary]

        Args:
            params (list): [description]
            X (pd.DataFrame): [description]

        Returns:
            float: [description]
        """
        results = self.seir_solution(params=params, dates=X["date"])
        rmse = np.sqrt(np.mean((results["removed"] - X["cumulative_deaths"]) ** 2))
        return rmse

    def fit(self, X: pd.DataFrame):
        """
        [summary]

        Args:
            X (pd.DataFrame): [description]
        """
        self.X_original = X.copy()
        self.province = X["province"].iloc[0]
        self.N = X["population"].iloc[0]
        self.S_0 = self.N - self.E_0 - self.I_0

        # Find optimal parameters
        x0 = [0.5, 0.5, 0.5]
        bounds = [(1e-4, 50), (1e-4, 50), (1e-4, 50)]
        optimal = minimize(self.loss, x0=x0, bounds=bounds, method="L-BFGS-B", args=(X))
        self.optimal_params = optimal.x
        return

    def forecast(self, h: int = 21) -> pd.DataFrame:
        """
        [summary]

        Args:
            h (int, optional): [description]. Defaults to 21.
        """
        start_date = self.X_original["date"].iloc[0]
        end_date = self.X_original["date"].iloc[-1]
        end_date_forecast = end_date + timedelta(days=h)
        dates = pd.Series(
            pd.date_range(start_date, end_date_forecast, name="date")
        ).dt.date

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

        forecasts = forecasts.merge(
            self.X_original, how="left", on=["province", "date"]
        )

        return forecasts