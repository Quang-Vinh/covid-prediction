# -*- coding: utf-8 -*-

"""
Module for model evaluation
"""

# Built-in
from typing import Any, Iterable, List, Tuple

# Other
from joblib import delayed, Parallel, parallel
import numpy as np
import pandas as pd


def time_series_cross_val_scores(
    model: Any,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    responses: List[str],
    k: int = 10,
    h_list: List[int] = [1, 3, 5, 7, 14, 21],
    min_train_size: int = 50,
    n_jobs: int = 1,
) -> dict:
    """
    Calculate cross validation scores for time series by sampling k split points without replacement to split into train/test sets. For each response specified and
    it will calculate the 1 step cross val forecast error, 5 step cross val forecast error, etc... based on h_list values.

    Args:
        model (Any): Model class with fit and forecast methods. Forecasts should include columns with name response_pred.
        X (pd.DataFrame): Dataframe of predictors
        Y (pd.DataFrame): Dataframe of responses
        responses (List[str]): List of responses which should be included as column in Y and as columns of model.forecast(X) as response_pred
        k (int, optional): Number of cross validation splits to perform. Defaults to 10.
        h_list (List[str], optional): List of h step forecasts to calculate cross validation error. Defaults to [1, 3, 5, 7, 14, 21].
        min_train_size (int, optional): Minimum size of a training set. Defaults to 50.
        n_jobs (int, optional): Number of jobs to run in parallel, -1 for all processors. Defaults to 1.

    Returns:
        dict: Dictionary of with keys 'all' for all cross validation results and 'summarised' for the summarised rmse results per h and response.
    """

    n = X.shape[0]
    max_h = max(h_list)
    end = n - max_h

    # Sample split points without replacement from the bounds (min_train_size, total points - maximum h forecast)
    split_values = np.arange(start=min_train_size, stop=end + 1)
    split_points = np.random.choice(split_values, size=k, replace=False)

    # Run model for each split point and get scores
    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(
        delayed(fit_and_score)(
            model, X, Y, split_point=split_point, responses=responses, h_list=h_list
        )
        for split_point in split_points
    )
    cv_scores_all = pd.concat(results)

    # Aggregate to get rmse by h and response
    cv_scores_summarised = cv_scores_all.groupby(by=["h", "response"]).apply(
        lambda x: np.sqrt((x["error"] ** 2).mean())
    )
    cv_scores_summarised = pd.DataFrame({"rmse": cv_scores_summarised}).reset_index()
    cv_scores = {"all": cv_scores_all, "summarised": cv_scores_summarised}

    return cv_scores


def fit_and_score(
    model: Any,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    split_point: int,
    responses: List[str],
    h_list: List[int] = [1, 3, 5, 7, 14, 21],
) -> pd.DataFrame:
    """
    Split data into train/test sets using the split_point and then fits the model on the training and calculates scores on the test set.

    Args:
        model (Any): Model class with fit and forecast methods. Forecasts should include columns with name response_pred.
        X (pd.DataFrame): Dataframe of predictors
        Y (pd.DataFrame): Dataframe of responses
        split_point (int): Split point to separate train and test sets
        responses (List[str]): List of responses which should be included as column in Y and as columns of model.forecast(X) as response_pred
        h_list (List[str], optional): List of h step forecasts to calculate cross validation error. Defaults to [1, 3, 5, 7, 14, 21].

    Returns:
        [pd.DataFrame]: Dataframe containing the result scores for the model
    """
    max_h = max(h_list)

    # Split into train/test sets
    X_train, Y_train = X.iloc[:split_point], Y.iloc[:split_point]
    Y_test = Y.iloc[split_point : (split_point + max_h)]

    model.fit(X_train, Y_train)

    # Get forecasts for last max_h values
    forecasts = model.forecast(h=max_h)
    forecasts = forecasts.iloc[-max_h:]

    # Get result for each h forecast and response
    cv_scores = []
    for h in h_list:
        for response in responses:
            forecast = forecasts.iloc[h - 1][response + "_pred"]
            actual = Y_test.iloc[h - 1][response]
            error = forecast - actual
            cv_scores.append(
                {
                    "split_point": split_point,
                    "h": h,
                    "response": response,
                    "forecast": forecast,
                    "actual": actual,
                    "error": error,
                }
            )
    cv_scores = pd.DataFrame(cv_scores)

    return cv_scores
