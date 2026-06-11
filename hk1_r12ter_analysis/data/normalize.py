from typing import Literal

from loguru import logger
import pandas as pd

from hk1_r12ter_analysis.util import check_axis_value, check_method_value


def normalize_by_sum(
    df: pd.DataFrame,
    axis: Literal[0, "index", 1, "columns"] = 0,
    reference: pd.Series | pd.DataFrame | int | float | None = None,
) -> pd.DataFrame:
    """Normalize the data by dividing each sample by the sum of its values.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be normalized.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to normalize values.
            - 0 or 'index': Normalize values across rows (i.e., within each column).
            - 1 or 'columns': Normalize values across columns (i.e., within each row).
    reference : pd.Series, pd.DataFrame, int, or float, default=None
        Reference values to multiply the normalized data by. Must have the same shape as the data along the axis of normalization.
        If an int or float is provided, the normalized data will be multiplied by this value.
        If a Series or DataFrame is provided, the values will be multiplied along the specified axis.

    Returns
    -------
    pd.DataFrame
        Normalized data.
    """
    axis = check_axis_value(axis)
    method = "sum"
    logger.debug(
        f"Normalizing data using '{method}' sample normalization along the specified axis..."
    )
    df = df.div(df.sum(axis=axis), axis=1 - axis)
    if reference is not None:
        logger.debug("Multiplying normalized data by reference...")
        df = df.mul(reference, axis=1 - axis)

    logger.info(f"Normalized data using '{method}' sample normalization along the specified axis.")
    return df


def normalize_by_median(
    df: pd.DataFrame,
    axis: Literal[0, "index", 1, "columns"] = 0,
    reference: pd.Series | pd.DataFrame | int | float | None = None,
) -> pd.DataFrame:
    """Normalize the data by dividing each sample by the median of its values.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be normalized.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to normalize values.
            - 0 or 'index': Normalize values across rows (i.e., within each column).
            - 1 or 'columns': Normalize values across columns (i.e., within each row).
    reference : pd.Series, pd.DataFrame, int, or float, default=None
        Reference values to multiply the normalized data by. Must have the same shape as the data along the axis of normalization.
        If an int or float is provided, the normalized data will be multiplied by this value.
        If a Series or DataFrame is provided, the values will be multiplied along the specified axis.

    Returns
    -------
    pd.DataFrame
        Normalized data.
    """
    axis = check_axis_value(axis)
    method = "median"
    logger.debug(
        f"Normalizing data using '{method}' sample normalization along the specified axis..."
    )
    df = df.div(df.median(axis=axis), axis=1 - axis)
    if reference is not None:
        logger.debug("Multiplying normalized data by reference...")
        df = df.mul(reference, axis=1 - axis)

    logger.info(f"Normalized data using '{method}' sample normalization along the specified axis.")
    return df


def normalize_data(
    df: pd.DataFrame,
    method: str,
    axis: Literal[0, "index", 1, "columns"] = 0,
    reference: pd.Series | pd.DataFrame | int | float | None = None,
) -> pd.DataFrame:
    """Normalize the data by applying a normalization method.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be normalized.
    method : str
        Normalization method to apply. Must be one of the following:
            - "sum": Normalize by dividing each sample by the sum of its values.
            - "median": Normalize by dividing each sample by the median of its values.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to normalize values.
            - 0 or 'index': Normalize values across rows (i.e., within each column).
            - 1 or 'columns': Normalize values across columns (i.e., within each row).
    reference : pd.Series, pd.DataFrame, int, or float, default=None
        Reference values to multiply the normalized data by. Must have the same shape as the data along the axis of normalization.
        If an int or float is provided, the normalized data will be multiplied by this value.
        If a Series or DataFrame is provided, the values will be multiplied along the specified axis.
        Required as a DataFrame or Series if the sample normalization method requires a reference (e.g. PQN).

    Returns
    -------
    pd.DataFrame
        Normalized data.
    """
    logger.debug(f"Applying '{method}' sample normalization...")
    data_normalization_functions = {
        # "factor": ,normalize_by_normalizing_factor # Sample-specific normalization by a normalization factor (i.e. weight, volume)
        "sum": normalize_by_sum,  # Normalization by sum
        "median": normalize_by_median,  # Normalization by median
        # "pqn": normalize_by_pqn, # Normalization by a reference sample (PQN)
        # "group_pqn": normalize_by_group_pqn, # Normalization by a pooled sample from group (group PQN)
        # "reference_feature": normalize_by_reference_feature, # Normalization by reference feature
        # "quantile": normalize_by_quantile, # Quantile normalization (suggested only for > 1000 features)
    }
    normalization_function = check_method_value(method.lower(), data_normalization_functions)
    df_normalized = normalization_function(df, axis=axis, reference=reference)
    return df_normalized
