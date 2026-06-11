from typing import Literal

from loguru import logger
import pandas as pd

from hk1_r12ter_analysis.util import check_axis_value

from .normalize import normalize_data
from .scale import scale_data
from .transform import transform_data


def sample_normalize_feature_scale_data(
    df: pd.DataFrame,
    normalization_method: str | None = None,
    transformation_method: str | None = None,
    scaling_method: str | None = None,
    axis: Literal[0, "index", 1, "columns"] = 0,
    reference: pd.Series | pd.DataFrame | int | float | None = None,
    add_constant: pd.Series | pd.DataFrame | int | float | None = None,
) -> pd.DataFrame:
    """Apply sample normalization, data transformation, and data scaling to the data in the specified order.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be normalized.
    normalization_method : str, default=None
        Sample-specific normalization method to apply. Must be one of "sum", "median". If None, no sample normalization will be applied.
    transformation_method : str, default=None
        Data transformation method to apply. Must be one of "log10", "log2", "sqrt", "cbrt". If None, no data transformation will be applied.
    scaling_method : str, default=None
        Data scaling method to apply. Must be one of "mean", "auto", "pareto", "range". If None, no data scaling will be applied.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to scale values.
            - 0 or 'index': Samples are rows and features are columns (i.e., normalize rows and transform/scale columns).
            - 1 or 'columns': Samples are columns and features are rows (i.e., normalize columns and transform/scale rows).
    reference : pd.Series, pd.DataFrame, int, or float, default=None
        Reference values to multiply the normalized data by. Must have the same shape as the data along the axis of normalization.
        If an int or float is provided, the normalized data will be multiplied by this value.
        If a Series or DataFrame is provided, the values will be multiplied along the specified axis.
        Required as a DataFrame or Series if the sample normalization method requires a reference (e.g. PQN).
    add_constant: pd.Series, pd.DataFrame, int, or float, default=None
        Value to add to data before applying the log transformation.
        Only applicable for transformations that require non-zero values (e.g. log transformations).
        If an int or float is provided, this value will be added to all values in the data.
        If a Series or DataFrame is provided, the values will be added to the data along the specified axis.
        If None, the value will be set to one-tenth of the minimum non-zero value along the specified axis.
        Useful for handling zero values in the data, as log transformations are undefined for zero values.

    Returns
    -------
    pd.DataFrame
        Normalized data.
    """
    # Sample normalization
    axis = check_axis_value(axis)
    if normalization_method is not None and normalization_method.lower() != "none":
        # Apply sample normalization along the opposite axis of transformation and scaling
        # (i.e. if transformation and scaling are applied along columns, sample normalization will be applied along rows, and vice versa)
        df = normalize_data(df, method=normalization_method, axis=1 - axis, reference=reference)
    else:
        logger.info("No sample normalization applied.")
    # Data transformation
    if transformation_method is not None and transformation_method.lower() != "none":
        df = transform_data(
            df,
            method=transformation_method,
            axis=axis,
            add_constant=add_constant,
        )
    else:
        logger.info("No data transformation applied.")
    # Data scaling
    if scaling_method is not None and scaling_method.lower() != "none":
        df = scale_data(df, method=scaling_method, axis=axis)
    else:
        logger.info("No feature scaling applied.")

    return df
