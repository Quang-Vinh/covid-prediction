# -*- coding: utf-8 -*-

from datetime import timedelta
import numpy as np
import pandas as pd
from pygam import PoissonGAM, l


# TODO: Rename all column variables to letter variables Z, I, Y, C


class StemPoissonRegressor:
    """
    Space-Time Epidemic model based on "Spatiotemporal Dynamics, Nowcasting and Forecasting of COVID-19 in the United States" (https://arxiv.org/abs/2004.14103)
    Fits two Poisson regression models to model the new cases and new deaths/recovered at time t.

    The first model for the new cases Y_t is modelled using the active cases I_t-1 and number of susceptible people S_t-1
    Y_t \sim Poisson(\mu_t) \\
    log(\mu_t) = \beta_{1t} + \beta_{2t}log(I_{t-1} + 1) + \alpha_tlog(S_{t-1}/N)   

    The second model for the new deaths/recovered \Delta D_t is modelled using the active cases I_t-1
    \Delta D_t \sim Poisson({\mu_t}^D) \\
    log({\mu_t}^D) = \beta_{1t}^D + \beta_{2t}^D log(I_{t-1} + 1)

    Attributes:
        X_original {pandas dataframe} -- Original X dataframe called on fit()
        Y_original {pandas dataframe} -- Original Y dataframe called on fit()
        X_cases {pandas dataframe} -- Transformed X dataframe used for fitting new cases model
        Y_case {pandas dataframe} -- Y dataframe used for fitting new cases model
        X_removed {pandas dataframe} -- Transformed X dataframe used for fitting new removed model
        Y_removed {pandas dataframe} -- Y dataframe used for fitting new removed model
        poisson_gam_cases {PoissonGAM model} -- Poisson regression model for new cases
        poisson_gam_removed {PoissonGAM model} -- Poisson regression model for new removed 

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
            Y (pd.DataFrame): Dataframe for given region of response variables containing columns cases, removed
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
        self.X_removed = X[["log_active_cases_yesterday"]]
        self.Y_removed = Y["removed"]

        # Model new cases data using infections and percentage susceptible at time t-1
        self.poisson_gam_cases = PoissonGAM(l(0) + l(1), verbose=self.verbose)
        self.poisson_gam_cases.fit(self.X_cases, self.Y_cases)

        # Model removed cases using infections at time t-1
        self.poisson_gam_removed = PoissonGAM(l(0), verbose=self.verbose)
        self.poisson_gam_removed.fit(self.X_removed, self.Y_removed)

        return

    def forecast(self, C: int, N: int, h: int = 1) -> pd.DataFrame:
        """
        Gives forecasted new cases, active cases, and cumulative number of removed.

        Args:
            C (int): Most recent value of cumulative number of confirmed cases
            N (int): Population
            h (int, optional): Number of h step predictions to make. Defaults to 1.

        Returns:
            pd.DataFrame: Dataframe containing 1 step predictions for all data in training set along with h step forecasts
        """
        predictions = pd.DataFrame()

        # Get 1 step predictions for all values in training set
        Y_cases_preds = self.poisson_gam_cases.predict(self.X_cases)
        Y_removed_preds = self.poisson_gam_removed.predict(self.X_removed)

        # Add to result dataframe
        predictions["date"] = self.X_original["date"].iloc[1:]
        predictions["cases_pred"] = Y_cases_preds
        predictions["removed_pred"] = Y_removed_preds
        predictions["active_cases_pred"] = np.nan
        predictions["is_forecast"] = False

        # Get h step predictions iteratively. Start with last actual known values of active cases and percent susceptible
        I = self.X_original["active_cases"].iloc[-1]
        Z = np.log(self.X_original["percent_susceptible"].iloc[-1])
        date = predictions["date"].max()

        for _ in range(h):
            # Get predictions of next step
            log_I = np.log(I + 1)
            Y = self.poisson_gam_cases.predict(np.array([log_I, Z]).reshape(1, 2))[0]
            R = self.poisson_gam_removed.predict(log_I)[0]

            # Update next values of I, Z, C
            I = max(I + Y - R, 1)
            C = C + Y
            Z = np.log((N - C) / N)
            date = date + timedelta(days=1)

            # Append predicted value at time t+h
            predictions = predictions.append(
                {
                    "date": date,
                    "cases_pred": Y,
                    "removed_pred": R,
                    "active_cases_pred": I,
                    "is_forecast": True,
                },
                ignore_index=True,
            )

        return predictions