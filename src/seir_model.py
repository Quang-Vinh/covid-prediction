# -*- coding: utf-8 -*-

"""
Module for SEIR model 
"""

# TODO: solve_ivp not returning y values for all of domain t sometimes
# TODO: Scale infected and removed loss when combining them for total loss
# TODO: have inflection rate of sigmoids as learnable parameter


# Built ins
from datetime import date, timedelta
from pathlib import Path
from typing import List
import sys

# Other
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, dual_annealing, minimize

# Owned
from .utils import combined_sigmoid

# Add parent path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.absolute()
sys.path.append(str(parent_dir))


class SEIRModel:
    """
    SEIR model time varying parameter for infection rate from Susceptible component to Exposed.

    Args:
        lam (float, optional): Lambda value used to combine removed error and infected error. Total error = lambda * removed error + (1 - lambda) * infected error. Defaults to 0.5.
        method (str, optional): Optimization method for estimating parameters. Defaults to 'L-BFGS-B'.
        date_splits (List[date], optional): List of dates to use in time varying parameters. Defaults to None
        verbose (int, optional): Level of verbose. Defaults to 0.

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

    def __init__(
        self,
        lam: float = 0.5,
        method: str = "L-BFGS-B",
        date_splits: List[date] = None,
        epoch: int = 1000,
        verbose: int = 0,
    ):
        # Input check
        if method not in (
            "L-BFGS-B",
            "Trust-Constr",
            "differential_evolution",
            "dual_annealing",
        ):
            raise Exception("Invalid method option")

        self.lam = lam
        self.method = method
        self.date_splits = date_splits
        self.epoch = epoch
        self.verbose = verbose

        self.S_0 = None
        self.I_0 = None
        self.R_0 = 0
        self.N = None
        self.province = None
        return

    def time_varying_betas(self) -> List[float]:
        """
        Returns list of the betas for all times from fitted data

        Returns:
            List[float]: List of betas
        """
        betas = self.optimal.x[1:-2]
        t = self.X_original["date"].shape[0] + 20
        betas = [combined_sigmoid(i, betas, self.time_splits, a=0.5) for i in range(t)]
        return betas

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
        # Get SEIR parameters. The Beta parameter which controls the movement of S to E can vary with time
        S, E, I, R = y
        alpha, gamma = params[-2:]
        betas = params[:-2]

        # Get beta for current time t using a sigmoid function
        beta = combined_sigmoid(t, y_list=betas, splits=self.time_splits, a=0.5)

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
        # Pop first parameter for E_0
        E_0 = params[0]
        params = params[1:]

        # Run initial value solver
        size = len(dates)
        solution = solve_ivp(
            self.seir_odes,
            y0=[self.S_0, E_0, self.I_0, self.R_0],
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

    def fit(self, X: pd.DataFrame, y=None):
        """
        Estimates best set of SEIR parameters for given data.

        Args:
            X (pd.DataFrame): DataFrame of region infection data. Contains columns 'province', 'population', 'active_cases', 'date', 'cumulative_deaths', 'active_cases'
            y (Any, optional): Empty parameter to fit framework.
        """
        self.X_original = X.copy()

        # Initialize values
        self.province = X["province"].iloc[0]
        self.N = X["population"].iloc[0]
        self.I_0 = X["active_cases"].iloc[0]
        self.S_0 = self.N - self.I_0

        # Set time splits for time varying parameters. Keep only date split points that exist within the given data
        if self.date_splits:
            dates = X["date"].to_list()
            self.time_splits = [
                dates.index(date_split)
                for date_split in self.date_splits
                if date_split in dates
            ]
        else:
            self.time_splits = []

        # First parameter is for E_0 , next n-2 parameters are for betas and last 2 parameters for alpha and gamma
        n_beta = len(self.time_splits) + 1
        n_params = n_beta + 3
        # Initial values and bounds for optimization. Bounds and initial value for E_0 will be different
        x0 = [10] + (n_params - 1) * [0.5]
        bounds = [(0, 100)] + (n_params - 1) * [(1e-4, 10)]

        # Find optimal parameters
        if self.method == "L-BFGS-B":
            self.optimal = minimize(
                self.loss,
                x0=x0,
                bounds=bounds,
                method="L-BFGS-B",
                args=(X),
                options={"disp": self.verbose},
            )
        elif self.method == "Trust-Constr":
            self.optimal = minimize(
                self.loss,
                x0=x0,
                bounds=bounds,
                method="trust-constr",
                args=(X),
                options={
                    "disp": self.verbose,
                    "maxiter": self.epoch,
                    "verbose": self.verbose,
                },
            )
        elif self.method == "differential_evolution":
            self.optimal = differential_evolution(
                self.loss,
                bounds=bounds,
                args=[X],
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
        elif self.method == "dual_annealing":
            self.optimal = dual_annealing(
                self.loss, bounds=bounds, maxiter=10, args=[X]
            )

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
        forecasts = self.seir_solution(params=self.optimal.x, dates=dates)
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


class SEIRModel2:
    """
    SEIR model with compartments Susceptible, Exposed, Infected Detected, Infected Undetected, Recovered, Deaths

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

    def __init__(
        self,
        lam: float = 0.5,
        method: str = "L-BFGS-B",
        date_splits: List[date] = None,
        verbose: bool = False,
        epoch: int = 200,
    ):
        # Input check
        if method not in (
            "L-BFGS-B",
            "differential_evolution",
            "dual_annealing",
            "Trust-Constr",
        ):
            raise Exception("Invalid method option")

        self.lam = lam
        self.method = method
        self.date_splits = date_splits
        self.verbose = verbose
        self.epoch = epoch

        self.S_0 = None
        self.E_0 = None
        self.ID_0 = None
        self.IU_0 = None
        self.R_0 = 0
        self.D_0 = 0
        self.N = None
        self.province = None
        return

    def time_varying_betas(self) -> List[float]:
        """
        Returns list of the betas for all times from fitted data
        Returns:
            List[float]: List of betas
        """
        betas = self.optimal.x[2:-6]
        t = self.X_original["date"].shape[0] + 20
        betas = [combined_sigmoid(i, betas, self.time_splits, a=0.5) for i in range(t)]
        return betas

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
        # Get SEIR parameters. The Beta parameter which controls the movement of S to E can vary with time
        S, E, ID, IU, R, D = y
        alpha, sigma, mu1, gamma1, mu2, gamma2 = params[-6:]
        # alpha, sigma, mu1, mu2 = params[-4:]

        betas = params[:-6]

        # Get beta for current time t using a sigmoid function
        beta = combined_sigmoid(t, y_list=betas, splits=self.time_splits, a=0.5)

        # Calculate ODEs
        dSdt = -beta * (ID + IU) * S / self.N
        dEdt = beta * (ID + IU) * S / self.N - sigma * E
        dIDdt = alpha * sigma * E - gamma1 * ID
        dIUdt = (1 - alpha) * sigma * E - gamma2 * IU
        dRdt = mu1 * gamma1 * ID + mu2 * gamma2 * IU
        dDdt = (1 - mu1) * gamma1 * ID + (1 - mu2) * gamma2 * IU
        return [dSdt, dEdt, dIDdt, dIUdt, dRdt, dDdt]

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
        self.E_0 = self.ID_0 * params[0]
        self.IU_0 = self.ID_0 * params[1]
        self.S_0 = self.N - self.E_0 - self.ID_0 - self.IU_0
        solution = solve_ivp(
            self.seir_odes,
            y0=[self.S_0, self.E_0, self.ID_0, self.IU_0, self.R_0, self.D_0],
            t_span=[0, size],
            t_eval=np.arange(0, size),
            args=[params[2:]],
        )

        # Create results dataframe
        y = solution.y
        results = pd.DataFrame(
            {
                "date": dates,
                "Susceptible": y[0],
                "Exposed": y[1],
                "InfectedDetected": y[2],
                "InfectedUndetected": y[3],
                "Recovered": y[4],
                "Death": y[5],
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
        # p=0.01
        removed_rmse = np.sqrt(
            np.mean((results["Death"] - X["cumulative_deaths"]) ** 2)
        )
        infected_rmse = np.sqrt(
            np.mean((results["InfectedDetected"] - X["active_cases"]) ** 2)
        )
        # total_rmse = self.lam * removed_rmse + (1 - self.lam) * infected_rmse
        # removed_rmse = np.mean((np.log(results["Death"]+p)-np.log(X["cumulative_deaths"]+p))**2)
        # infected_rmse = np.mean((np.log(results["InfectedDetected"]+p)-np.log(X["active_cases"]+p))**2)
        return self.lam * removed_rmse + (1 - self.lam) * infected_rmse

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
        self.ID_0 = X["active_cases"].iloc[0]
        # self.E_0 = self.ID_0*10
        # self.IU_0 = self.ID_0
        # self.S_0 = self.N - self.E_0 - self.ID_0 - self.IU_0

        # Set time splits for time varying parameters. Keep only date split points that exist within the given data
        if self.date_splits:
            dates = X["date"].to_list()
            self.time_splits = [
                dates.index(date_split)
                for date_split in self.date_splits
                if date_split in dates
            ]
        else:
            self.time_splits = []

        # Find optimal parameters
        n_beta = len(self.time_splits) + 1
        n_params = n_beta + 6
        x0 = [7] + [2] + n_beta * [0.2] + [0.8, 1 / 5, 0.9, 0.2, 0.9, 0.2]
        bounds = (
            2 * [(0.01, 20)]
            + n_beta * [(1e-4, 6)]
            + [(0.1, 1)]
            + [(1e-6, 4)]
            + [(1e-1, 1)]
            + [(1e-6, 4)]
            + [(0.1, 1)]
            + [(1e-6, 4)]
        )

        if self.method == "L-BFGS-B":
            self.optimal = minimize(
                self.loss,
                x0=x0,
                bounds=bounds,
                method="L-BFGS-B",
                args=(X),
                options={"disp": self.verbose},
            )
        elif self.method == "Trust-Constr":
            self.optimal = minimize(
                self.loss,
                x0=x0,
                bounds=bounds,
                method="trust-constr",
                args=(X),
                options={"disp": self.verbose, "maxiter": self.epoch, "verbose": 2},
            )
        elif self.method == "differential_evolution":
            self.optimal = differential_evolution(
                self.loss,
                bounds=bounds,
                args=[X],
                strategy="best1bin",
                maxiter=self.epoch,
                popsize=50,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=None,
                callback=None,
                disp=self.verbose,
                polish=True,
                init="latinhypercube",
                updating="deferred",
                workers=-1,
            )

        elif self.method == "dual_annealing":
            self.optimal = dual_annealing(
                self.loss, bounds=bounds, maxiter=self.epoch, args=[X]
            )

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
        forecasts = self.seir_solution(params=self.optimal.x, dates=dates)
        forecasts["province"] = self.province
        forecasts["is_forecast"] = forecasts["date"] > end_date
        forecasts.rename(
            columns={
                "Susceptible": "susceptible_pred",
                "Exposed": "active_exposed_pred",
                "InfectedDetected": "active_cases_pred",
                "Death": "cumulative_deaths_pred",
            },
            inplace=True,
        )

        # Merge forecasted data on the original X dataframe that was fitted on
        forecasts = forecasts.merge(
            self.X_original, how="left", on=["province", "date"]
        )

        return forecasts