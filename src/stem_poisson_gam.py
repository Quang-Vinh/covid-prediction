# -*- coding: utf-8 -*-

from datetime import timedelta
import numpy as np
import pandas as pd
from pygam import PoissonGAM, l


class StemPoissonRegressor:
    """
    Space-Time Epidemic model based on "Spatiotemporal Dynamics, Nowcasting and Forecasting of COVID-19 in the United States" (https://arxiv.org/abs/2004.14103)
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Args:
            verbose (bool, optional): Whether to print messages on fit. Defaults to False.
        """
        self.verbose = verbose
        return

    def fit(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
    ):
        """
        Fit a poisson regression model each for the cases using active_cases and percentage_susceptible at time t-1, and another model
        for removed using active_cases at time t-1.

        Args:
            X (pd.DataFrame): Dataframe for given region of predictor variables containing columns date, active_cases, percent_susceptible
            Y (pd.DataFrame): Dataframe for given region of response variables containing columns cases, cumulative_removed
        """
        self.X_original = X.copy()
        self.Y_original = Y.copy()

        # Add previous day t-1 information to reach row
        previous_day = X.shift(periods=1, axis=0).loc[
            :, ["active_cases", "percent_susceptible"]
        ]
        X = X.assign(
            active_cases_yesterday=previous_day["active_cases"],
            percent_susceptible_yesterday=previous_day["percent_susceptible"],
        )

        # Drop first row since missing t-1 value
        X = X.iloc[1:]
        Y = Y.iloc[1:]

        # Preprocess variables with log transformation
        X = X.assign(
            log_active_cases_yesterday=lambda x: np.log(
                x["active_cases_yesterday"] + 1
            ),
            log_percent_susceptible_yesterday=lambda x: np.log(
                x["percent_susceptible_yesterday"]
            ),
        )

        # Separate data for each model
        self.X_cases = X[
            ["log_active_cases_yesterday", "log_percent_susceptible_yesterday"]
        ]
        self.Y_cases = Y["cases"]
        self.X_cumulative_removed = X[["log_active_cases_yesterday"]]
        self.Y_cumulative_removed = Y["cumulative_removed"]

        # Model new cases data using infections and percentage susceptible at time t-1
        self.poisson_gam_cases = PoissonGAM(l(0) + l(1), verbose=self.verbose)
        self.poisson_gam_cases.fit(self.X_cases, self.Y_cases)

        # Model removed cases using infections at time t-1
        self.poisson_gam_removed = PoissonGAM(l(0), verbose=self.verbose)
        self.poisson_gam_removed.fit(
            self.X_cumulative_removed, self.Y_cumulative_removed
        )

        return

    def forecast(self, C: int, N: int, h: int = 1) -> pd.DataFrame:
        """
        Gives forecasted new cases, active cases, and cumulative number of removed.

        Args:
            C (int): Most recent value of cumulative number of confirmed cases
            N (int): Population
            h (int, optional): Number of h step predictions to make. Defaults to 1.

        Returns:
            pd.DataFrame: [description]
        """
        predictions = pd.DataFrame()

        # Get 1 step predictions for all values in training set
        Y_cases_preds = self.poisson_gam_cases.predict(self.X_cases)
        Y_cumulative_removed_preds = self.poisson_gam_removed.predict(
            self.X_cumulative_removed
        )

        # Add to result dataframe
        predictions["date"] = self.X_original["date"].iloc[1:]
        predictions["cases_pred"] = Y_cases_preds
        predictions["cumulative_removed_pred"] = Y_cumulative_removed_preds
        predictions["active_cases_pred"] = np.nan
        predictions["is_forecast"] = False

        # Get h step predictions iteratively. Start with last actual known values of active cases and percent susceptible
        I = self.X_original["active_cases"].iloc[-1]
        Z = np.log(self.X_original["percent_susceptible"].iloc[-1])
        R_previous = self.Y_cumulative_removed.iloc[-1]
        date = predictions["date"].max()

        for _ in range(h):
            # Get predictions of next step
            log_I = np.log(I)
            Y = self.poisson_gam_cases.predict(np.array([log_I, Z]).reshape(1, 2))[0]
            R = self.poisson_gam_removed.predict(log_I)[0]

            # Update next values of I, Z, C
            R_delta = R - R_previous
            R_previous = R

            I = max(I + Y - R_delta, 1)
            C = C + Y
            Z = np.log((N - C) / N)
            date = date + timedelta(days=1)

            # Append predicted value at time t+h
            predictions = predictions.append(
                {
                    "date": date,
                    "cases_pred": Y,
                    "cumulative_removed_pred": R,
                    "active_cases_pred": I,
                    "is_forecast": True,
                },
                ignore_index=True,
            )

        return predictions