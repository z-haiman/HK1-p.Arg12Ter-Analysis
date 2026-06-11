from typing import Literal

from loguru import logger
import numpy as np
import pandas as pd

from hk1_r12ter_analysis.util import check_axis_value, check_method_value


def _numpy_transform(
    df: pd.DataFrame,
    method: str,
    axis: Literal[0, "index", 1, "columns"] = 0,
    add_constant: pd.Series | pd.DataFrame | int | float | None = None,
) -> pd.DataFrame:
    """Apply a data transformation method to the data using a specified function.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be transformed.
    method : str
        Data transformation method to apply. Must be one of the following:
            - "log10": Log transformation (base 10)
            - "log2": Log transformation (base 2)
            - "sqrt": Square root transformation (square root of data values)
            - "cbrt": Cube root transformation (cube root of data values)
            - "vsn": Variance stabilizing normalization (data-adaptive transformation)
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to transform values.
            - 0 or 'index': Transform values across rows (i.e., within each column).
            - 1 or 'columns': Transform values across columns (i.e., within each row).
    add_constant: pd.Series, pd.DataFrame, int, or float, default=None
        Value to add to data before applying the transformation.
        Only applicable for transformations that require non-zero values (e.g. log transformations).
        If an int or float is provided, this value will be added to all values in the data.
        If a Series or DataFrame is provided, the values will be added to the data along the specified axis.
        If None, the value will be set to one-tenth of the minimum non-zero value along the specified axis.
        Useful for handling zero values in the data, as log transformations are undefined for zero values.

    Warnings
    --------
    For data transformations, it is recommended to use the ``transform_data`` function instead of this function directly, as it provides a more user-friendly interface and handles the selection of the appropriate transformation function based on the specified method.

    Returns
    -------
    pd.DataFrame
        Transformed data.
    """
    axis = check_axis_value(axis)
    logger.debug(f"Transforming data with {method} transformation along the specified axis...")
    transformation_function = {
        "log10": np.log10,  # Log transformation (base 10)
        "log2": np.log2,  # Log transformation (base 2)
        "sqrt": np.sqrt,  # Square root transformation (square root of data values)
        "cbrt": np.cbrt,  # Cube root transformation (cube root of data values)
    }[method]

    # Handle zero values for log transformations
    if method.startswith("log"):
        if (df == 0).any(axis=None) and add_constant is None:
            logger.info("Zero values detected in the data before log transformation.")
            if add_constant is None:
                logger.debug(
                    "Calculating vector of constant values equal to one-tenth of the minimum non-zero value along the specified axis..."
                )
                add_constant = (
                    df[df != 0].abs().min() / 10 if axis == 0 else df.T[df.T != 0].abs().min() / 10
                )

        if add_constant is not None:
            logger.debug(
                f"Adding the specified constant value to the data before applying {method} transformation."
            )
            df = df.add(add_constant, axis=1 - axis)

    # Warn if negative values are present in the data for transformations that may produce NaN or infinite values (e.g. log transformations)
    if (df < 0).any(axis=None) and method != "cbrt":
        logger.warning(
            "Negative values detected in the data. The transformation may produce NaN or infinite values."
        )

    df_transformed = df.apply(transformation_function, axis=axis)
    logger.info(f"Transformed data with {method} transformation along the specified axis.")
    return df_transformed


def transform_log10(
    df: pd.DataFrame,
    axis: Literal[0, "index", 1, "columns"] = 0,
    add_constant: pd.Series | pd.DataFrame | int | float | None = None,
) -> pd.DataFrame:
    """Apply log transformation (base 10) to the data.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be transformed.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to transform values.
            - 0 or 'index': Transform values across rows (i.e., within each column).
            - 1 or 'columns': Transform values across columns (i.e., within each row).
    add_constant: pd.Series, pd.DataFrame, int, or float, default=None
        Value to add to data before applying the log transformation.
        If an int or float is provided, this value will be added to all values in the data.
        If a Series or DataFrame is provided, the values will be added to the data along the specified axis.
        If None, the value will be set to one-tenth of the minimum non-zero value along the specified axis.
        Useful for handling zero values in the data, as log transformations are undefined for zero values.

    Returns
    -------
    pd.DataFrame
        Transformed data.
    """
    return _numpy_transform(df, method="log10", axis=axis, add_constant=add_constant)


def transform_log2(
    df: pd.DataFrame,
    axis: Literal[0, "index", 1, "columns"] = 0,
    add_constant: pd.Series | pd.DataFrame | int | float | None = None,
) -> pd.DataFrame:
    """Apply log transformation (base 2) to the data.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be transformed.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to transform values.
            - 0 or 'index': Transform values across rows (i.e., within each column).
            - 1 or 'columns': Transform values across columns (i.e., within each row).
    add_constant: pd.Series, pd.DataFrame, int, or float, default=None
        Value to add to data before applying the log transformation.
        If an int or float is provided, this value will be added to all values in the data.
        If a Series or DataFrame is provided, the values will be added to the data along the specified axis.
        If None, the value will be set to one-tenth of the minimum non-zero value along the specified axis.
        Useful for handling zero values in the data, as log transformations are undefined for zero values.

    Returns
    -------
    pd.DataFrame
        Transformed data.
    """
    return _numpy_transform(df, method="log2", axis=axis, add_constant=add_constant)


def transform_sqrt(df: pd.DataFrame, axis: Literal[0, "index", 1, "columns"] = 0) -> pd.DataFrame:
    """Apply square root transformation to the data.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be transformed.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to transform values.
            - 0 or 'index': Transform values across rows (i.e., within each column).
            - 1 or 'columns': Transform values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Transformed data.
    """
    return _numpy_transform(df, method="sqrt", axis=axis, add_constant=None)


def transform_cbrt(df: pd.DataFrame, axis: Literal[0, "index", 1, "columns"] = 0) -> pd.DataFrame:
    """Apply cube root transformation to the data.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be transformed.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to transform values.
            - 0 or 'index': Transform values across rows (i.e., within each column).
            - 1 or 'columns': Transform values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Transformed data.
    """
    return _numpy_transform(df, method="cbrt", axis=axis, add_constant=None)


def transform_data(
    df: pd.DataFrame,
    method: str,
    axis: Literal[0, "index", 1, "columns"] = 0,
    add_constant: pd.Series | pd.DataFrame | int | float | None = None,
) -> pd.DataFrame:
    """Apply a data transformation method to the data.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be transformed.
    method : str
        Data transformation method to apply. Must be one of the following:
            - "log10": Log transformation (base 10)
            - "log2": Log transformation (base 2)
            - "sqrt": Square root transformation (square root of data values)
            - "cbrt": Cube root transformation (cube root of data values)
            - "vsn": Variance stabilizing normalization (data-adaptive transformation)
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to transform values.
            - 0 or 'index': Transform values across rows (i.e., within each column).
            - 1 or 'columns': Transform values across columns (i.e., within each row).
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
        Transformed data.
    """
    logger.debug(f"Applying '{method}' data transformation...")
    data_transformation_functions = {
        "log10": transform_log10,  # Log transformation (base 10)
        "log2": transform_log2,  # Log transformation (base 2)
        "sqrt": transform_sqrt,  # Square root transformation (square root of data values)
        "cbrt": transform_cbrt,  # Cube root transformation (cube root of data values)
        # "vsn": transform_vsn, # Variance stabilizing normalization (data-adaptive transformation)
    }
    transformation_function = check_method_value(method.lower(), data_transformation_functions)
    df_transformed = transformation_function(df, axis=axis, add_constant=add_constant)
    return df_transformed
