from typing import Literal

from loguru import logger
import pandas as pd

from hk1_r12ter_analysis.util import (
    check_axis_value,
    check_method_value,
    check_percentage_value,
)


def _log_filtering_results(
    n_vars_before: int,
    n_vars_after: int,
    filter_type: str,
    method: str,
    percent: float,
) -> None:
    """Log the results of the filtering process.

    Parameters
    ----------
    n_vars_before: int
        Number of variables before filtering.
    n_vars_after: int
        Number of variables after filtering.
    filter_type: str
        Type of filter applied (e.g. "quality", "repeatability", "variance", "abundance").
    method: str, None
        Method used for filtering. Must be one of the following:
        - "quality" filtering methods: "groupwise missingness", "global missingness"
        - "variance" filtering methods: "iqr", "sd", "mad", "rsd", "nprsd"
        - "abundance" filtering methods: "mean", "median"
    percent: float
        Percentage value or threshold used for filtering out variables.
    """
    logger.info(
        f"Filtered data using '{method}' low-{filter_type} filter at {percent:.0%} along the specified axis."
    )
    logger.info(f"{n_vars_before - n_vars_after}/{n_vars_before} variables removed.")
    logger.info(f"{n_vars_after}/{n_vars_before} variables remaining.")


def filter_low_quality_by_missingness(
    df: pd.DataFrame,
    percent: float = 0,
    use_groupwise_threshold: bool = False,
    group_key: str | None = None,
    axis: Literal[0, "index", 1, "columns"] = 0,
) -> pd.DataFrame:
    """Filter data by percentage of missing values (i.e., low-quality filter based on missingness).

    Parameters
    ----------
    df : pd.DataFrame
        Data to be filtered.
    percent : float, default=0
        Threshold for filtering out variables based on missingness. Must be between 0 and 1, inclusive.
        Variables with a percentage of missing values above this threshold will be filtered out.
        No filter is applied if the value is 0 (i.e., no variables filtered) or 1 (i.e., all variables filtered).
    use_groupwise_threshold : bool, default=False
        Whether to apply the missingness threshold separately for each group defined by 'group_key'.
        If True, variables that meet the threshold in at least one group will be retained.
        If False, the missingness threshold will be applied to the entire dataset without considering groupings.
    group_key : str, default=None
        Column name in 'df' that defines the groups for applying group-wise missingness threshold.
        Required if 'use_groupwise_threshold' is True.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).


    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    axis = check_axis_value(axis)
    percent = check_percentage_value(percent)
    method = ("groupwise" if use_groupwise_threshold else "global") + " missingness"
    filter_type = "quality"
    if percent == 0 or percent == 1:
        logger.info(f"No low-{filter_type} filter applied for value of {percent:.0%}.")
        return df

    logger.debug(
        f"Filtering data using '{method}' low-{filter_type} filter at '{percent:.0%}' along the specified axis..."
    )
    df_na = df.isna().T if axis == 1 else df.isna()
    if use_groupwise_threshold:
        # Check that `group_key` is provided when `use_groupwise_threshold` is True.
        if group_key is None:
            raise ValueError(
                "`group_key` must be provided when `use_groupwise_threshold` is True."
            )
        # Check for the presence of `group_key` in the index or columns of `df_na` and apply group-wise mean to calculate missingness.
        if group_key in df_na.index.names:
            df_na = df_na.groupby(level=group_key).mean()
        elif group_key in df_na.columns:
            df_na = df_na.groupby(by=group_key).mean()
        else:
            raise ValueError(f"`{group_key}` not found.")
        # Determine missingness for each variable based on whether the missingness threshold is met in at least one group.
        series_filter = (df_na < percent).any(axis=0)
    else:
        # Determine missingness for each variable based on whether the missingness threshold is met in the entire dataset.
        series_filter = df_na.mean(axis=0) < percent

    # Apply filter to the data and log the results.
    df_filtered = df.loc[:, series_filter] if axis == 0 else df.loc[series_filter, :]
    _log_filtering_results(
        n_vars_before=df.shape[1 - axis],
        n_vars_after=df_filtered.shape[1 - axis],
        filter_type=filter_type,
        method=method,
        percent=percent,
    )
    return df_filtered


# TODO: Implement low-quality filter functions for handling blanks.
# TODO Implement low-repeatability filter functions for handling QC samples.


# Low-variance filter functions
def get_percentage_by_n_vars(n_vars: int) -> float:
    """Return percentage of variables to be filtered based on the number of variables

    Parameters
    ----------
    n_vars: int
        Number of variables in the data.

    Returns
    -------
    float
        Percentage of variables to be filtered out.
    """
    logger.debug(
        "Determining percentage of variables to filter out based on number of variables in the data...."
    )
    if n_vars < 0:
        raise ValueError("'n_vars' must be a non-negative number'.")
    elif n_vars < 250:
        percentage = 0.05
    elif n_vars < 500:
        percentage = 0.1
    elif n_vars < 1000:
        percentage = 0.25
    else:
        percentage = 0.4

    logger.info(
        f"Number of variables in the data is {n_vars}. Setting percentage to filter out to {percentage:.0%}."
    )
    return percentage


def filter_low_variance_by_iqr(
    df: pd.DataFrame,
    percent: float | None = None,
    axis: Literal[0, "index", 1, "columns"] = 0,
) -> pd.DataFrame:
    """Filter data using the interquartile range (IQR).

    Parameters
    ----------
    df : pd.DataFrame
        Data to be filtered.
    percent : float, default=None
        Percentage of variables to filter out. Must be between 0 and 1, inclusive.
        No filter is applied if the value is 0 (i.e., no variables filtered) or 1 (i.e., all variables filtered).
        If None, the percentage to filter out will be determined based on the number of variables in the data.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    axis = check_axis_value(axis)
    if percent is None:
        percent = get_percentage_by_n_vars(df.shape[1 - axis])
    percent = check_percentage_value(percent)
    method = "iqr"
    filter_type = "variance"
    if percent == 0 or percent == 1:
        logger.info(f"No low-{filter_type} filter applied for value of {percent:.0%}.")
        return df

    # Get interquartile range filter, then apply filter to the data and log the results.
    logger.debug(
        f"Filtering data using '{method}' low-{filter_type} filter at '{percent:.0%}' along the specified axis..."
    )
    series_iqr = df.quantile(0.75, axis=axis) - df.quantile(0.25, axis=axis)
    series_filter = series_iqr > series_iqr.quantile(percent)
    df_filtered = df.loc[:, series_filter] if axis == 0 else df.loc[series_filter, :]
    _log_filtering_results(
        n_vars_before=df.shape[1 - axis],
        n_vars_after=df_filtered.shape[1 - axis],
        filter_type=filter_type,
        method=method,
        percent=percent,
    )
    return df_filtered


def filter_low_variance_by_sd(
    df: pd.DataFrame, percent: float | None = None, axis: Literal[0, "index", 1, "columns"] = 0
) -> pd.DataFrame:
    """Filter data using the standard deviation (SD).

    Parameters
    ----------
    df : pd.DataFrame
        Data to be filtered.
    percent : float, default=None
        Percentage of variables to filter out. Must be between 0 and 1, inclusive.
        No filter is applied if the value is 0 (i.e., no variables filtered) or 1 (i.e., all variables filtered).
        If None, the percentage to filter out will be determined based on the number of variables in the data.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    axis = check_axis_value(axis)
    if percent is None:
        percent = get_percentage_by_n_vars(df.shape[1 - axis])
    percent = check_percentage_value(percent)
    method = "sd"
    filter_type = "variance"
    if percent == 0 or percent == 1:
        logger.info(f"No low-{filter_type} filter applied for value of {percent:.0%}.")
        return df

    # Get standard deviation filter, then apply filter to the data and log the results.
    logger.debug(
        f"Filtering data using '{method}' low-{filter_type} filter at '{percent:.0%}' along the specified axis..."
    )
    series_sd = df.std(axis=axis)
    series_filter = series_sd > series_sd.quantile(percent)
    df_filtered = df.loc[:, series_filter] if axis == 0 else df.loc[series_filter, :]
    _log_filtering_results(
        n_vars_before=df.shape[1 - axis],
        n_vars_after=df_filtered.shape[1 - axis],
        filter_type=filter_type,
        method=method,
        percent=percent,
    )
    return df_filtered


def filter_low_variance_by_mad(
    df: pd.DataFrame, percent: float | None = None, axis: Literal[0, "index", 1, "columns"] = 0
) -> pd.DataFrame:
    """Filter data using the median absolute deviation (MAD).

    Parameters
    ----------
    df : pd.DataFrame
        Data to be filtered.
    percent : float, default=None
        Percentage of variables to filter out. Must be between 0 and 1, inclusive.
        No filter is applied if the value is 0 (i.e., no variables filtered) or 1 (i.e., all variables filtered).
        If None, the percentage to filter out will be determined based on the number of variables in the data.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    axis = check_axis_value(axis)
    if percent is None:
        percent = get_percentage_by_n_vars(df.shape[1 - axis])
    percent = check_percentage_value(percent)
    method = "mad"
    filter_type = "variance"
    if percent == 0 or percent == 1:
        logger.info(f"No low-{filter_type} filter applied for value of {percent:.0%}.")
        return df

    # Get median absolute deviation filter, then apply filter to the data and log the results.
    logger.debug(
        f"Filtering data using '{method}' low-{filter_type} filter at '{percent:.0%}' along the specified axis..."
    )
    series_mad = (df.subtract(df.median(axis=1 - axis), axis=axis)).abs().median(axis=axis)
    series_filter = series_mad > series_mad.quantile(percent)
    df_filtered = df.loc[:, series_filter] if axis == 0 else df.loc[series_filter, :]
    _log_filtering_results(
        n_vars_before=df.shape[1 - axis],
        n_vars_after=df_filtered.shape[1 - axis],
        filter_type=filter_type,
        method=method,
        percent=percent,
    )
    return df_filtered


def filter_low_variance_by_rsd(
    df: pd.DataFrame, percent: float | None = None, axis: Literal[0, "index", 1, "columns"] = 0
) -> pd.DataFrame:
    """Filter data using the relative standard deviation (RSD = SD/mean).

    Parameters
    ----------
    df : pd.DataFrame
        Data to be filtered.
    percent : float, default=None
        Percentage of variables to filter out. Must be between 0 and 1, inclusive.
        No filter is applied if the value is 0 (i.e., no variables filtered) or 1 (i.e., all variables filtered).
        If None, the percentage to filter out will be determined based on the number of variables in the data.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    axis = check_axis_value(axis)
    if percent is None:
        percent = get_percentage_by_n_vars(df.shape[1 - axis])
    percent = check_percentage_value(percent)
    method = "rsd"
    filter_type = "variance"
    if percent == 0 or percent == 1:
        logger.info(f"No low-{filter_type} filter applied for value of {percent:.0%}.")
        return df

    # Get relative standard deviation filter, then apply filter to the data and log the results.
    logger.debug(
        f"Filtering data using '{method}' low-{filter_type} filter at '{percent:.0%}' along the specified axis..."
    )
    series_rsd = df.std(axis=1 - axis).div(df.mean(axis=1 - axis))
    series_filter = series_rsd > series_rsd.quantile(percent)
    df_filtered = df.loc[:, series_filter] if axis == 0 else df.loc[series_filter, :]
    _log_filtering_results(
        n_vars_before=df.shape[1 - axis],
        n_vars_after=df_filtered.shape[1 - axis],
        filter_type=filter_type,
        method=method,
        percent=percent,
    )
    return df_filtered


def filter_low_variance_by_nprsd(
    df: pd.DataFrame, percent: float | None = None, axis: Literal[0, "index", 1, "columns"] = 0
) -> pd.DataFrame:
    """Filter data using the non-parametric relative standard deviation (NPRSD = MAD/median).

    Parameters
    ----------
    df : pd.DataFrame
        Data to be filtered.
    percent : float, default=None
        Percentage of variables to filter out. Must be between 0 and 1, inclusive.
        No filter is applied if the value is 0 (i.e., no variables filtered) or 1 (i.e., all variables filtered).
        If None, the percentage to filter out will be determined based on the number of variables in the data.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    axis = check_axis_value(axis)
    if percent is None:
        percent = get_percentage_by_n_vars(df.shape[1 - axis])
    percent = check_percentage_value(percent)
    method = "nprsd"
    filter_type = "variance"
    if percent == 0 or percent == 1:
        logger.info(f"No low-{filter_type} filter applied for value of {percent:.0%}.")
        return df

    # Get non-parametric relative standard deviation filter, then apply filter to the data and log the results.
    logger.debug(
        f"Filtering data using '{method}' low-{filter_type} filter at '{percent:.0%}' along the specified axis..."
    )
    series_nprsd = (
        (df.subtract(df.median(axis=1 - axis), axis=axis))
        .abs()
        .median(axis=axis)
        .div(df.median(axis=1 - axis))
    )
    series_filter = series_nprsd > series_nprsd.quantile(percent)
    df_filtered = df.loc[:, series_filter] if axis == 0 else df.loc[series_filter, :]
    _log_filtering_results(
        n_vars_before=df.shape[1 - axis],
        n_vars_after=df_filtered.shape[1 - axis],
        filter_type=filter_type,
        method=method,
        percent=percent,
    )
    return df_filtered


def filter_low_variance(
    df: pd.DataFrame,
    method: str,
    percent: float | None = None,
    axis: Literal[0, "index", 1, "columns"] = 0,
) -> pd.DataFrame:
    """Filter data by applying the specified low-variance method.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be filtered.
    method : str
        Method to use for low-variance filtering. Must be one of the following:
        - "iqr": Interquantile range (IQR)
        - "sd": Standard deviation (SD)
        - "mad": Median absolute deviation (MAD)
        - "rsd": Relative standard deviation (RSD = SD/mean)
        - "nprsd": Non-parametric relative standard deviation (MAD/median)
    percent : float, default=None
        Percentage of variables to filter out. Must be between 0 and 1, inclusive.
        No filter is applied if the value is 0 (i.e., no variables filtered) or 1 (i.e., all variables filtered).
        If None, the percentage to filter out will be determined based on the number of variables in the data.
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    logger.debug(f"Applying '{method}' low-variance filter...")
    low_variance_filter_functions = {
        "iqr": filter_low_variance_by_iqr,  # Interquantile range (IQR)
        "sd": filter_low_variance_by_sd,  # Standard deviation (SD)
        "mad": filter_low_variance_by_mad,  # Median absolute deviation (MAD)
        "rsd": filter_low_variance_by_rsd,  # Relative standard deviation (RSD = SD/mean)
        "nprsd": filter_low_variance_by_nprsd,  # Non-parametric relative standard deviation (MAD/median)
    }
    low_variance_filter_function = check_method_value(
        method.lower(), low_variance_filter_functions
    )
    df_filtered = low_variance_filter_function(df, percent=percent, axis=axis)
    return df_filtered


# Low-abundance filter functions
def filter_low_abundance_by_mean(
    df: pd.DataFrame, percent: float = 0, axis: Literal[0, "index", 1, "columns"] = 0
) -> pd.DataFrame:
    """Filter data using the mean abundance.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be filtered.
    percent : float, default=None
        Percentage of variables to filter out. Must be between 0 and 1, inclusive.
        No filter is applied if the value is 0 (i.e., no variables filtered) or 1 (i.e., all variables filtered).
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    axis = check_axis_value(axis)
    percent = check_percentage_value(percent)
    method = "mean"
    filter_type = "abundance"
    if percent == 0 or percent == 1:
        logger.info(f"No low-{filter_type} filter applied for value of {percent:.0%}.")
        return df

    # Get median abundance filter, then apply filter to the data and log the results.
    logger.debug(
        f"Filtering data using '{method}' low-{filter_type} filter at '{percent:.0%}' along the specified axis..."
    )
    series_mean = df.mean(axis=axis)
    series_filter = series_mean > series_mean.quantile(percent)
    df_filtered = df.loc[:, series_filter] if axis == 0 else df.loc[series_filter, :]
    _log_filtering_results(
        n_vars_before=df.shape[1 - axis],
        n_vars_after=df_filtered.shape[1 - axis],
        filter_type=filter_type,
        method=method,
        percent=percent,
    )
    return df_filtered


def filter_low_abundance_by_median(
    df: pd.DataFrame, percent: float = 0, axis: Literal[0, "index", 1, "columns"] = 0
) -> pd.DataFrame:
    """Filter data using the median abundance.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be filtered.
    percent : float, default=None
        Percentage of variables to filter out. Must be between 0 and 1, inclusive.
        No filter is applied if the value is 0 (i.e., no variables filtered) or 1 (i.e., all variables filtered).
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    axis = check_axis_value(axis)
    percent = check_percentage_value(percent)
    method = "median"
    filter_type = "abundance"
    if percent == 0 or percent == 1:
        logger.info(f"No low-{filter_type} filter applied for value of {percent:.0%}.")
        return df

    # Get median abundance filter, then apply filter to the data and log the results.
    logger.debug(
        f"Filtering data using '{method}' low-{filter_type} filter at '{percent:.0%}' along the specified axis..."
    )
    series_median = df.median(axis=axis)
    series_filter = series_median > series_median.quantile(percent)
    df_filtered = df.loc[:, series_filter] if axis == 0 else df.loc[series_filter, :]
    _log_filtering_results(
        n_vars_before=df.shape[1 - axis],
        n_vars_after=df_filtered.shape[1 - axis],
        filter_type=filter_type,
        method=method,
        percent=percent,
    )
    return df_filtered


def filter_low_abundance(
    df: pd.DataFrame,
    method: str,
    percent: float | None = None,
    axis: Literal[0, "index", 1, "columns"] = 0,
) -> pd.DataFrame:
    """Filter data by applying the specified low-abundance method.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be filtered.
    method : str
        Method to use for low-abundance filtering. Must be one of the following:
        - "mean": Mean abundance
        - "median": Median abundance
    percent : float, default=None
        Percentage of variables to filter out. Must be between 0 and 1, inclusive.
        No filter is applied if the value is 0 (i.e., no variables filtered) or 1 (i.e., all variables filtered).
    axis : {0 or 'index', 1 or 'columns'}, default=0
        The axis along which to check for constant values.
            - 0 or 'index': Check for constant values across rows (i.e., within each column).
            - 1 or 'columns': Check for constant values across columns (i.e., within each row).

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    logger.debug(f"Applying '{method}' low-abundance filter...")
    low_abundance_filter_functions = {
        "mean": filter_low_abundance_by_mean,  # Mean
        "median": filter_low_abundance_by_median,  # Median
    }
    low_abundance_filter_function = check_method_value(
        method.lower(), low_abundance_filter_functions
    )
    df_filtered = low_abundance_filter_function(df, percent=percent, axis=axis)
    return df_filtered
