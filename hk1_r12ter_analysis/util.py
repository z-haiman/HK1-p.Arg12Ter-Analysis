"""Submodule for various utility functions used across the HK1-R12X analysis package."""

import re
from typing import Callable, Iterable, Literal

import numpy as np


def floor_to_decimal(number: float, decimals: int = 0, scalar: float = 1.0) -> float:
    """Rounds a number down to a specified number of decimal places.

    Parameters
    ----------
    number : float
        The number to round down.
    decimals : int, optional
        The number of decimal places to round to (default is 0, which rounds to the nearest integer).
        Negative values will round to the left of the decimal point (e.g., -1 rounds to the nearest 10).
        Positive values will round to the right of the decimal point (e.g., 2 rounds to the nearest hundredth).
    scalar : float, optional
        An optional factor to multiply the number by before rounding and divide by after rounding (default is 1.0, which means no scaling).
        Ensures that the returned number is divisible by the scalar.

    Returns
    -------
    float
        The rounded number.
    """

    scalar = float(scalar)  # Ensure scalar is a float to avoid issues with integer division
    factor = 10**decimals
    return (np.floor((number / scalar) * factor) / factor) * scalar


def ceil_to_decimal(number: float, decimals: int = 0, scalar: float = 1.0) -> float:
    """Rounds a number up to a specified number of decimal places.

    Parameters
    ----------
    number : float
        The number to round up.
    decimals : int, optional
        The number of decimal places to round to (default is 0, which rounds to the nearest integer).
        Negative values will round to the left of the decimal point (e.g., -1 rounds to the nearest 10).
        Positive values will round to the right of the decimal point (e.g., 2 rounds to the nearest hundredth).
    scalar : float, optional
        An optional factor to multiply the number by before rounding and divide by after rounding (default is 1.0, which means no scaling).
        Ensures that the returned number is divisible by the scalar.

    Returns
    -------
    float
        The rounded number.
    """
    scalar = float(scalar)  # Ensure scalar is a float to avoid issues with integer division
    factor = 10**decimals
    return (np.ceil((number / scalar) * factor) / factor) * scalar


def check_axis_value(axis: Literal[0, "index", 1, "columns"]) -> Literal[0, 1]:
    """Check the value of the 'axis' variable and return it as an integer.

    Parameters
    ----------
    axis : int | str
        The axis value to check. Can be 0, 1, "index", or "columns".
    Returns
    -------
    int
        The axis value as an integer (0 for index, 1 for columns).
    """
    if axis in (0, "index"):
        return 0
    elif axis in (1, "columns"):
        return 1
    else:
        raise ValueError(
            f"Unrecognized value '{axis}' for 'axis'. Must be 0, 1, 'index', or 'columns'."
        )


def check_percentage_value(percentage: float) -> float:
    """Check whether a percentage value is valid (between 0 and 1, inclusive).

    Parameters
    ----------
    percentage : float
        The percentage value to check.

    Returns
    -------
    float
        The validated percentage value.
    """
    percentage = float(percentage)  # Ensure the value is a float
    if percentage < 0 or percentage > 1:
        raise ValueError(
            f"Percentage value must be between 0 and 1, inclusive. Received: {percentage}"
        )

    return percentage


def check_method_value(method: str, functions_dict: dict) -> Callable:
    """Check whether a method is allowed and return the corresponding function.

    Parameters
    ----------
    method : str
        The method name to check.
    functions_dict : dict
        A dictionary mapping method names to functions.

    Returns
    -------
    callable
        The function corresponding to the specified method.
    """
    # Ensure method is recognized
    try:
        return functions_dict[method]
    except KeyError:
        raise ValueError(
            f"Unrecognized method '{method}'. Must be one of the following: {list(functions_dict)}."
        )


def multiple_str_replace(text: str, replacement_dict: dict) -> str:
    """Replace multiple substrings in a text based on a replacement dictionary.

    Parameters
    ----------
    text : str
        The original text in which to perform replacements.
    replacement_dict : dict
        A dictionary where keys are substrings to be replaced and values are the corresponding replacements.
    Returns
    -------
    str
        The modified text with all replacements applied.

    """
    # Create a regular expression from the dictionary keys
    regex = re.compile("|".join(map(re.escape, replacement_dict.keys())))
    # For each match, look up the replacement in the dictionary
    return regex.sub(lambda match: replacement_dict[match.group(0)], text)


def ensure_iterable(obj) -> Iterable:
    """Ensure object is iterable.

    Parameters
    ----------
    obj : any
        The object to check.

    Returns
    -------
    iterable
        The object itself if it is already iterable (but not a string), or a list containing the object if it is not iterable.
    """
    try:
        iter(obj)  # Try to get an iterator
        return obj if not isinstance(obj, str) else [obj]
    except TypeError:
        # If a TypeError is raised, the object is not iterable, so wrap it in a list
        return [obj]
