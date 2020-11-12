# -*- coding: utf-8 -*-


# Modeling
import numpy as np
import pandas as pd
from pygam import PoissonGAM, l

# Built-ins
from datetime import timedelta


def preprocess_data(X: pd.DataFrame, drop_first_day: bool = False) -> pd.DataFrame:
    """
    Preprocess data to be used in StemPoissonRegressor by adding columns for all regions as predictors

    Args:
        X (pd.DataFrame): Dataframe with columns province, date, active_cases, percent_susceptible
        drop_first_day (bool, optional): Whether to drop first day of each province or not. Defaults to False.

    Returns:
        pd.DataFrame: Preprocess dataframe with columns for all provinces
    """
    # Add columns with log transformation
    X = X.assign(
        log_active_cases=lambda x: np.log(x["active_cases"] + 1),
        log_percent_susceptible=lambda x: np.log(x["percent_susceptible"]),
    )

    # Add columns for previous day information
    previous_day = (
        X.groupby("province")
        .shift(periods=1, axis=0)
        .loc[
            :,
            [
                "active_cases",
                "percent_susceptible",
                "log_active_cases",
                "log_percent_susceptible",
            ],
        ]
    )
    X = X.assign(
        active_cases_yesterday=previous_day["active_cases"],
        percent_susceptible_yesterday=previous_day["percent_susceptible"],
        log_active_cases_yesterday=previous_day["log_active_cases"],
        log_percent_susceptible_yesterday=previous_day["log_percent_susceptible"],
    )

    # Add previous day columns for each province
    combined = X.copy()
    provinces = combined["province"].unique()
    for province in provinces:
        # Get province data rows and duplicate n times for concat column wise
        prov_data = X.query("province == @province").loc[
            :,
            [
                "active_cases_yesterday",
                "percent_susceptible_yesterday",
                "log_active_cases_yesterday",
                "log_percent_susceptible_yesterday",
                "active_cases",
                "percent_susceptible",
                "log_active_cases",
                "log_percent_susceptible",
            ],
        ]
        prov_data = pd.concat([prov_data] * len(provinces), ignore_index=True)

        # Append name of province to each column name
        for col in prov_data.columns:
            prov_data.rename(columns={col: f"{province}_{col}"}, inplace=True)

        combined = pd.concat([combined, prov_data], axis=1)

    # Drop first days missing t-1 information
    if drop_first_day:
        combined = combined.query("active_cases_yesterday == active_cases_yesterday")

    combined.reset_index(drop=True, inplace=True)

    return combined


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
        X_cases {pandas dataframe} -- Dictionary containing transformed X dataframe used for fitting new cases model for each province
        Y_case {pandas dataframe} -- Dictionary containing Y dataframe used for fitting new cases model for each province
        X_removed {pandas dataframe} -- Dictionary containing transformed X dataframe used for fitting new removed model for each province
        Y_removed {pandas dataframe} -- Dictionary containing Y dataframe used for fitting new removed model for each province
        poisson_gam_cases {PoissonGAM model} -- Dictionary of Poisson regression model for new cases for each province
        poisson_gam_removed {PoissonGAM model} -- Dictionary of Poisson regression model for new removed for each province

    """

    def __init__(
        self, verbose: bool = False, lam_main: float = 0.6, lam_other: float = 10
    ) -> None:
        """
        Args:
            verbose (bool, optional): Whether to print messages on fit. Defaults to False.
            lam_main (float, optional): Lambda for regularization of main province effects. Defaults to 0.6
            lam_other (float, optional): Lambda for regularization of other province effects. Defaults to 1.
        """
        self.verbose = verbose
        self.lam_main = lam_main
        self.lam_other = lam_other
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
            X (pd.DataFrame): Dataframe for given region of predictor variables containing columns date, province, active_cases, percent_susceptible,
                              and all columns for provinces for {province_name}_active_cases_yesterday, {province_name}_percent_susceptible_yesterday,
                              as well as all log features
            Y (pd.DataFrame): Dataframe for given region of response variables containing columns date, province, cases, removed
        """
        self.X_original = X.copy()
        self.Y_original = Y.copy()
        self.provinces = X["province"].unique()

        # Fit model for each province
        self.X_cases = {}
        self.Y_cases = {}
        self.X_removed = {}
        self.Y_removed = {}
        self.poisson_gam_cases = {}
        self.poisson_gam_removed = {}

        for province in self.provinces:
            # Remove extra columns for given province in form {province}_column_name
            cols_drop = X.filter(regex=province, axis=1).columns
            X_province = X.query(f"province == '{province}'").drop(cols_drop, axis=1)
            Y_province = Y.query(f"province == '{province}'")

            # Store case dataframe used to train model for each province
            self.X_cases[province] = X_province.filter(
                regex=r"(log_active_cases_yesterday|log_percent_susceptible_yesterday)"
            )
            self.Y_cases[province] = Y_province["cases"]

            # Add terms for each province I_t-1 and Z_t-1
            terms = l(0, lam=self.lam_main) + l(1, lam=self.lam_main)
            for i in range(1, len(self.provinces)):
                terms += l(i * 2, lam=self.lam_other) + l(i * 2 + 1, lam=self.lam_other)

            # Fit cases model for province
            cases_model = PoissonGAM(terms, verbose=self.verbose)
            cases_model.fit(self.X_cases[province], self.Y_cases[province])
            self.poisson_gam_cases[province] = cases_model

            # Store remove dataframe used to train model for each province
            self.X_removed[province] = X_province.filter(
                regex=r"log_active_cases_yesterday"
            )
            self.Y_removed[province] = Y_province["removed"]

            # Add terms for each province I_t-1
            terms = l(0, lam=self.lam_main)
            for i in range(1, len(self.provinces)):
                terms += l(i, lam=self.lam_other)

            # Fit removed model for each province
            removed_model = PoissonGAM(terms, verbose=self.verbose)
            removed_model.fit(self.X_removed[province], self.Y_cases[province])
            self.poisson_gam_removed[province] = removed_model

        return

    def forecast(self, h: int = 1) -> pd.DataFrame:
        """
        Gives forecasted new cases, active cases, and cumulative number of removed.

        Args:
            h (int, optional): Number of h step predictions to make. Defaults to 1.

        Returns:
            pd.DataFrame: Dataframe containing 1 step predictions for all data in training set along with h step forecasts
        """
        # Get 1 step predictions for all values in training set for each province
        forecasts = pd.DataFrame(
            columns=[
                "province",
                "date",
                "active_cases_pred",
                "cases_pred",
                "removed_pred",
                "is_forecast",
            ]
        )

        for province in self.provinces:
            # Get cases and removed predictions
            cases_preds = self.poisson_gam_cases[province].predict(
                self.X_cases[province]
            )
            removed_preds = self.poisson_gam_removed[province].predict(
                self.X_removed[province]
            )

            # Create province prediction dataframe
            dates = self.X_original.query(f"province == '{province}'")["date"]
            province_forecasts = pd.DataFrame(
                {
                    "province": province,
                    "date": dates,
                    "active_cases_pred": np.nan,
                    "cases_pred": cases_preds,
                    "removed_pred": removed_preds,
                    "is_forecast": False,
                }
            )

            forecasts = forecasts.append(province_forecasts, ignore_index=True)

        date = forecasts["date"].max()

        # Dictionary of populations
        populations = self.X_original[["province", "population"]].drop_duplicates()
        N = dict(zip(populations["province"], populations["population"]))

        # Keep track of all I, Z, C for all provinces
        I, Z, C = {}, {}, {}
        for province in self.provinces:
            prov_data = self.X_original.query(f"province == '{province}'")
            I[province], Z[province], C[province] = [], [], []
            I[province].append(prov_data["active_cases"].iloc[-1])
            Z[province].append(prov_data["log_percent_susceptible"].iloc[-1])
            C[province].append(prov_data["cumulative_cases"].iloc[-1])

        # Get h step predictions for all provinces at the same time. Get 1 step forecast for all provinces then use that to get 2 step forecast for all and so on
        for i in range(h):
            date = date + timedelta(days=1)
            for province in self.provinces:
                # Initialize X inputs for model predict
                log_I = np.log(I[province][i] + 1)
                x_cases = [log_I, Z[province][i]]
                x_removed = [log_I]

                # Add columns for all other provinces
                for _province in self.provinces:
                    if _province != province:
                        _log_I = np.log(I[_province][i] + 1)
                        x_cases.append(_log_I)
                        x_cases.append(Z[_province][i])
                        x_removed.append(_log_I)

                # Get predictions for current province
                x_cases = np.array(x_cases).reshape(1, len(self.provinces) * 2)
                x_removed = np.array(x_removed).reshape(1, len(self.provinces))
                Y = self.poisson_gam_cases[province].predict(x_cases)[0]
                R = self.poisson_gam_removed[province].predict(x_removed)[0]

                # Update next values of I, Z, C
                I_new = max(I[province][i] + Y - R, 0)
                I[province].append(I_new)
                C[province].append(C[province][i] + Y)
                Z_new = np.log((N[province] - C[province][i + 1]) / N[province])
                Z[province].append(Z_new)

                # Append forecasts for current province
                forecasts = forecasts.append(
                    {
                        "province": province,
                        "date": date,
                        "active_cases_pred": I[province][i + 1],
                        "cases_pred": Y,
                        "removed_pred": R,
                        "is_forecast": True,
                    },
                    ignore_index=True,
                )

        return forecasts
