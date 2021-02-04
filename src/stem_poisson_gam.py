# -*- coding: utf-8 -*-


# Modeling
import numpy as np
import pandas as pd
from pygam import PoissonGAM, s, l

# Built-ins
from datetime import date, timedelta
from typing import List


def preprocess_data(
    X: pd.DataFrame, add_province_columns: bool = False, drop_first_day: bool = False
) -> pd.DataFrame:
    """
    Preprocess data to be used in StemPoissonRegressor by adding columns for previous day information and also adding columns for all regions as predictors

    Args:
        X (pd.DataFrame): Dataframe with columns province, date, active_cases, percent_susceptible
        add_province_columns (bool, optional): If variables active_cases and percent_susceptible should be added as columns for each province. Defaults to False.
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
    X_new = X.copy()
    if add_province_columns:
        provinces = X_new["province"].unique()
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

            X_new = pd.concat([X_new, prov_data], axis=1)

    # Drop first days missing t-1 information
    if drop_first_day:
        X_new = X_new.query("active_cases_yesterday == active_cases_yesterday")

    X_new.reset_index(drop=True, inplace=True)

    return X_new


def split_columns_dates(
    df: pd.DataFrame,
    date_splits: List[date],
    cols: List[str] = None,
    drop_date: bool = False,
):
    """
    Split all columns in df into several columns partitioned by date. For example if date_splits is [January 1 2020, January 10 2020, January 20 2020] then

    each column of df is split into 3 columns where col_0 has values for Jan 1 2020 to Jan 10 2020 and is 0 every else, ..., and col_2 has values for Jan 20 2020 and onwards.

    Args:
        df (pd.DataFrame): Dataframe with at least a date column.
        date_splits (List[date]): List of dates for bounds.
        cols (List[str], optional): List of column to perform partition by date. If none then applies to all columns. Defaults to None
        drop_date (bool): Whether to keep date column in result dataframe. Defaults to False.

    Returns:
        [pd.DataFrame]: Dataframe df with columns split by dates
    """
    df = df.copy()

    # Keep only dates that are within the dates of df
    date_splits = [date for date in date_splits if date < df["date"].max()]

    if not cols:
        cols = list(df.columns)
        cols.remove("date")

    # Split each column by each date bound
    for col in cols:
        for i in range(len(date_splits)):

            # Get bounds for dates
            start_date = date_splits[i]

            if i == len(date_splits) - 1:
                end_date = df["date"].max() + timedelta(days=10)
            else:
                end_date = date_splits[i + 1] - timedelta(days=1)

            # Add new column to df where between date bounds there are values and everywhere else is 0
            col_name = f"{col}_{i}"
            col_values = df[col].copy()
            date_index = ~df["date"].between(start_date, end_date)
            col_values.loc[date_index] = 0

            df.insert(1, col_name, col_values)

        df = df.drop(col, axis=1)

    if drop_date:
        df = df.drop("date", axis=1)

    return df


class StemPoissonRegressor:
    """
    Space-Time Epidemic model based on "Spatiotemporal Dynamics, Nowcasting and Forecasting of COVID-19 in the United States" (https://arxiv.org/abs/2004.14103)
    Fits two Poisson regression models to model the new cases and new deaths/recovered at time t.
    Also has option for time varying parameters by date/

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

    def __init__(
        self,
        verbose: bool = False,
        date_splits: List[date] = None,
        cols_date_splits: List[str] = None,
        use_spline: bool = False,
    ) -> None:
        """
        Args:
            verbose (bool, optional): Whether to print messages on fit. Defaults to False.
            date_splits (List[date], optional): List of dates for bounds if want to use time varying parameters. Defaults to None.
            cols_date_splits (List[str], optional): List of columns to allow time varying parameters for. If none then uses all except the intercept. Defaults to None.
            use_spline (bool, optional): Whether to use splines in the GAM model, if false then linear terms are used instead. Defaults to False.
        """
        self.verbose = verbose
        self.date_splits = date_splits
        self.cols_date_splits = (
            cols_date_splits
            if cols_date_splits
            else ["log_active_cases_yesterday", "log_percent_susceptible_yesterday"]
        )
        self.use_spline = use_spline
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

        # Separate data for each model
        self.X_cases = X[
            ["date", "log_active_cases_yesterday", "log_percent_susceptible_yesterday"]
        ].copy()
        self.Y_cases = Y["cases"]
        self.X_removed = X[["log_active_cases_yesterday"]].copy()
        self.Y_removed = Y["removed"]

        # Add intercept constant
        self.X_cases["intercept"] = 1

        # If time varying parameters then split each column by the date bounds
        if self.date_splits:
            # Keep only dates that are within the dates of df
            self.date_splits = [
                date for date in self.date_splits if date < self.X_cases["date"].max()
            ]

            self.X_cases = split_columns_dates(
                df=self.X_cases,
                date_splits=self.date_splits,
                cols=self.cols_date_splits,
                drop_date=True,
            )
        else:
            self.X_cases = self.X_cases.drop("date", axis=1)

        # Setup terms to use in GLM
        term = s if self.use_spline else l
        terms_removed = term(0)
        terms_cases = term(0)
        for i in range(1, len(self.X_cases.columns)):
            terms_cases = terms_cases + term(i)

        # Model new cases data using infections and percentage susceptible at time t-1
        self.poisson_gam_cases = PoissonGAM(
            terms_cases, fit_intercept=False, verbose=self.verbose
        )
        self.poisson_gam_cases.fit(self.X_cases, self.Y_cases)

        # Model removed cases using infections at time t-1
        self.poisson_gam_removed = PoissonGAM(terms_removed, verbose=self.verbose)
        self.poisson_gam_removed.fit(self.X_removed, self.Y_removed)

        return

    def forecast(self, h: int = 1) -> pd.DataFrame:
        """
        Gives forecasted new cases, active cases, and cumulative number of removed.
        Args:
            h (int, optional): Number of h step predictions to make. Defaults to 1.
        Returns:
            pd.DataFrame: Dataframe containing 1 step predictions for all data in training set along with h step forecasts
        """
        province = self.X_original["province"].iloc[-1]

        # Get 1 step predictions for all values in training set
        cases_preds = self.poisson_gam_cases.predict(self.X_cases)
        removed_preds = self.poisson_gam_removed.predict(self.X_removed)
        # cases_ci = self.poisson_gam_cases.confidence_intervals(self.X_cases)
        # removed_ci = self.poisson_gam_removed.confidence_intervals(self.X_removed)

        # Create result dataframe for training set data
        forecasts = pd.DataFrame(
            {
                "date": self.X_original["date"],
                "province": province,
                "cases_pred": cases_preds,
                "removed_pred": removed_preds,
                "active_cases_pred": np.nan,
                # "cases_ci_lower": cases_ci[:, 0],
                # "cases_ci_upper": cases_ci[:, 1],
                # "removed_ci_lower": removed_ci[:, 0],
                # "removed_ci_upper": removed_ci[:, 1],
                "is_forecast": False,
            }
        )

        # Get h step predictions iteratively. Start with last actual known values of active cases and percent susceptible
        C = self.X_original["cumulative_cases"].iloc[-1]
        N = self.X_original["population"].iloc[-1]
        I = self.X_original["active_cases"].iloc[-1]
        Z = np.log(self.X_original["percent_susceptible"].iloc[-1])
        date = forecasts["date"].max()

        # Keep track of current forecast to be used to predict next value
        x_cases = self.X_cases.iloc[0, :].copy()
        p = self.X_cases.shape[1]

        # Set column names for indexing x_cases series when setting new values
        if self.date_splits:
            i = len(self.date_splits) - 1

            if "intercept" in self.cols_date_splits:
                intercept = f"intercept_{i}"
            else:
                intercept = "intercept"

            if "log_active_cases_yesterday" in self.cols_date_splits:
                col_log_I = f"log_active_cases_yesterday_{i}"
            else:
                col_log_I = "log_active_cases_yesterday"

            if "log_percent_susceptible_yesterday" in self.cols_date_splits:
                col_Z = f"log_percent_susceptible_yesterday_{i}"
            else:
                col_Z = "log_percent_susceptible_yesterday"
        else:
            intercept = "intercept"
            col_log_I = "log_active_cases_yesterday"
            col_Z = "log_percent_susceptible_yesterday"

        for _ in range(h):
            # Set current values to previous forecast values
            log_I = np.log(I + 1)
            x_cases.loc[:] = 0
            x_cases.loc[[intercept, col_log_I, col_Z]] = 1, log_I, Z

            # Get predictions and CI for next step
            Y = self.poisson_gam_cases.predict(x_cases.values.reshape(1, p))[0]
            R = self.poisson_gam_removed.predict(log_I)[0]
            # Y_ci = self.poisson_gam_cases.confidence_intervals(x_cases)[0]
            # R_ci = self.poisson_gam_removed.confidence_intervals(log_I)[0]

            # Update next values of I, Z, C
            I = max(I + Y - R, 1)
            C = C + Y
            Z = np.log((N - C) / N)
            date = date + timedelta(days=1)

            # Append predicted value at time t+h
            forecasts = forecasts.append(
                {
                    "date": date,
                    "province": province,
                    "cases_pred": Y,
                    "removed_pred": R,
                    "active_cases_pred": I,
                    # "cases_ci_lower": Y_ci[0],
                    # "cases_ci_upper": Y_ci[1],
                    # "removed_ci_lower": R_ci[0],
                    # "removed_ci_upper": R_ci[1],
                    "is_forecast": True,
                },
                ignore_index=True,
            )

        # Add cumulative cases and removed predictions
        forecasts = forecasts.assign(
            cumulative_cases_pred=lambda x: x["cases_pred"].cumsum(),
            cumulative_removed_pred=lambda x: x["removed_pred"].cumsum(),
        )

        return forecasts


class StemPoissonRegressorCombined:
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
        self,
        verbose: bool = False,
        use_splines: bool = False,
        lam_main: float = 0.6,
        lam_other: float = 10,
    ) -> None:
        """
        Args:
            verbose (bool, optional): Whether to print messages on fit. Defaults to False.
            use_spline (bool, optional): Whether to use splines in the GAM model, if false then linear terms are used instead. Defaults to False.
            lam_main (float, optional): Lambda for regularization of main province effects. Defaults to 0.6
            lam_other (float, optional): Lambda for regularization of other province effects. Defaults to 1.
        """
        self.verbose = verbose
        self.use_splines = use_splines
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

            # Add terms for each province I_t-1 and Z_t-1. Either splines or linear terms
            if self.use_splines:
                terms = s(0, lam=self.lam_main) + s(1, lam=self.lam_main)
                for i in range(1, len(self.provinces)):
                    terms += s(i * 2, lam=self.lam_other) + s(
                        i * 2 + 1, lam=self.lam_other
                    )
            else:
                terms = l(0, lam=self.lam_main) + l(1, lam=self.lam_other)
                for i in range(1, len(self.provinces)):
                    terms += l(i * 2, lam=self.lam_other) + l(
                        i * 2 + 1, lam=self.lam_other
                    )

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
                try:
                    Y = self.poisson_gam_cases[province].predict(x_cases)[0]
                except:
                    print(x_cases)
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
