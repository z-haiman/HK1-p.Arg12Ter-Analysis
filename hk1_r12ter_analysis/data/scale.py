from typing import Literal

from loguru import logger
import numpy as np
import pandas as pd

from hk1_r12ter_analysis.util import check_axis_value, check_method_value


def mean_center_data(
    df: pd.DataFrame, axis: Literal[0, "index", 1, "columns"] = 0
) -> pd.DataFrame:
    """Scale the data by centering each variable to have a mean of zero.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be scaled.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to scale values.
            - 0 or 'index': Scale values across rows (i.e., within each column).
            - 1 or 'columns': Scale values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Scaled data.
    """
    axis = check_axis_value(axis)
    method = "mean"
    logger.debug(f"Scaling data with {method} scaling along the specified axis...")
    df_scaled = df.sub(df.mean(axis=axis), axis=1 - axis)
    logger.info(f"Scaled data with {method} scaling along the specified axis.")
    return df_scaled


def auto_scale_data(df: pd.DataFrame, axis: Literal[0, "index", 1, "columns"] = 0) -> pd.DataFrame:
    """Scale the data by centering each variable to have a mean of zero and scaling to have a standard deviation of one (i.e. z-score normalization).

    Parameters
    ----------
    df : pd.DataFrame
        Data to be scaled.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to scale values.
            - 0 or 'index': Scale values across rows (i.e., within each column).
            - 1 or 'columns': Scale values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Scaled data.
    """
    axis = check_axis_value(axis)
    method = "auto"
    logger.debug(f"Scaling data with {method} scaling along the specified axis...")
    df_scaled = df.sub(df.mean(axis=axis), axis=1 - axis).div(
        df.std(ddof=1, axis=axis), axis=1 - axis
    )
    logger.info(f"Scaled data with {method} scaling along the specified axis.")
    return df_scaled


def pareto_scale_data(
    df: pd.DataFrame, axis: Literal[0, "index", 1, "columns"] = 0
) -> pd.DataFrame:
    """Scale the data by centering each variable to have a mean of zero and scaling to have a standard deviation equal to the square root of the standard deviation of each variable (i.e. Pareto scaling).

    Parameters
    ----------
    df : pd.DataFrame
        Data to be scaled.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to scale values.
            - 0 or 'index': Scale values across rows (i.e., within each column).
            - 1 or 'columns': Scale values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Scaled data.
    """
    axis = check_axis_value(axis)
    method = "pareto"
    logger.debug(f"Scaling data with {method} scaling along the specified axis...")
    df_scaled = df.sub(df.mean(axis=axis), axis=1 - axis).div(
        np.sqrt(df.std(ddof=1, axis=axis)), axis=1 - axis
    )
    logger.info(f"Scaled data with {method} scaling along the specified axis.")
    return df_scaled


def range_scale_data(
    df: pd.DataFrame, method: str, axis: Literal[0, "index", 1, "columns"] = 0
) -> pd.DataFrame:
    """Scale the data by centering each variable to have a mean of zero and scaling to have a range of one (i.e. range scaling).

    Parameters
    ----------
    df : pd.DataFrame
        Data to be scaled.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to scale values.
            - 0 or 'index': Scale values across rows (i.e., within each column).
            - 1 or 'columns': Scale values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Scaled data.
    """
    axis = check_axis_value(axis)
    method = "range"
    logger.debug(f"Scaling data with {method} scaling along the specified axis...")
    df_scaled = df.sub(df.mean(axis=axis), axis=1 - axis).div(
        df.max(axis=axis) - df.min(axis=axis), axis=1 - axis
    )
    logger.info(f"Scaled data with {method} scaling along the specified axis.")
    return df_scaled


def scale_data(
    df: pd.DataFrame, method: str, axis: Literal[0, "index", 1, "columns"] = 0
) -> pd.DataFrame:
    """Apply a data scaling method to the data.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be scaled.
    method : str
        Data scaling method to apply. Must be one of the following:
            - "mean": Scale by centering to zero mean to zero while maintaining the original variance of each variable.
            - "auto": Scale by centering to zero mean and unit variance for each variable (i.e. z-score normalization).
            - "pareto": Scale by centering to zero mean and dividing by the square root of the standard deviation of each variable.
            - "range": Scale by centering to zero mean and dividing by the range of each variable.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to scale values.
            - 0 or 'index': Scale values across rows (i.e., within each column).
            - 1 or 'columns': Scale values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Scaled data.
    """
    logger.debug(f"Applying '{method}' data scaling...")
    data_scaling_functions = {
        "mean": mean_center_data,  # mean-centered only
        "auto": auto_scale_data,  # mean-centered and divided by the standard deviation of each variable
        "pareto": pareto_scale_data,  # mean-centered and divided by the square root of the standard deviation of each variable)
        "range": range_scale_data,  # mean-centered and divided by the range of each variable
    }
    scaling_function = check_method_value(method.lower(), data_scaling_functions)
    df_scaled = scaling_function(df, axis=axis)
    return df_scaled
