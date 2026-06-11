"""Submodule to handle data cleaning, including zero value replacement, identification and removal of duplicate features, and identification of features with constant values."""

from typing import Literal

from loguru import logger
import pandas as pd

from hk1_r12ter_analysis.util import check_axis_value


def replace_zero_values_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace zero values with NaN.

    Parameters
    ----------
    values : pd.DataFrame
        The input data containing zero values to be replaced.

    Returns
    -------
    pd.DataFrame
        The data with zero values replaced by NaN.

    """
    n_zeros = (df.values == 0).sum()
    if n_zeros == 0:
        logger.info("No zero values found in the data. No replacement needed.")
        return df

    logger.debug(f"Replacing {n_zeros} zero values with NaN values")
    df = df.mask(df == 0)
    logger.info(f"Replaced {n_zeros} zero values with NaN values")
    return df


def identify_constant_features(
    df: pd.DataFrame, tolerance: float = 1e-12, axis: Literal[0, "index", 1, "columns"] = 0
) -> list:
    """Identify features with constant values within a specified tolerance.

    Parameters
    ----------
    df : pd.DataFrame
        The input data containing features to be checked for constant values.
    tolerance : float, default=1e-12
        The tolerance within which values are considered constant.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).

    Returns
    -------
    list
        The list of the features that have constant values.

    """
    if tolerance is None or tolerance < 0:
        raise ValueError("Tolerance must be a non-negative value.")

    axis = check_axis_value(axis)
    series_constant = df.max(axis=axis) - df.min(axis=axis) <= tolerance
    constant_features = series_constant[series_constant].index.tolist()
    logger.info(
        f"Identified {len(constant_features)} features with constant values within a tolerance of {tolerance} along the specified axis."
    )
    all_nan_features = df.isna().all(axis=axis)
    if all_nan_features.any():
        all_nan_features = all_nan_features[all_nan_features].index.tolist()
        logger.info(
            f"Identified {len(all_nan_features)} features with all NaN values along the specified axis."
        )
        constant_features = list(set(constant_features) | set(all_nan_features))

    if constant_features:
        logger.warning(
            f"Identified {len(constant_features)} features with constant or all NaN values along the specified axis."
        )
    return constant_features


def remove_constant_features(
    df: pd.DataFrame, tolerance: float = 1e-12, axis: Literal[0, "index", 1, "columns"] = 0
) -> pd.DataFrame:
    """Remove features with constant values within a specified tolerance.

    Parameters
    ----------
    df : pd.DataFrame
        The input data containing features to be checked for constant values.
    tolerance : float, default=1e-12
        The tolerance within which values are considered constant.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).
    Returns
    -------
    pd.DataFrame
        The data with features that have constant values removed.

    """
    axis = check_axis_value(axis)
    constant_features = identify_constant_features(df, tolerance=tolerance, axis=axis)
    df = df.drop(constant_features, axis=1 - axis)
    logger.info(
        f"Removed {len(constant_features)} features with constant values within a tolerance of {tolerance}."
    )
    return df
