from pathlib import Path
import shutil
from typing import Literal, Sequence

import pandas as pd

from .config import PROJ_ROOT, logger

FILE_SEP_DICT = {
    ".csv": ",",
    ".tsv": "\t",
}


def load_dataframe_from_file(
    file_path: str | Path,
    index_col: str | int | Sequence[str | int] | Literal[False] | None = None,
    header: int | Sequence[int] | Literal["infer"] = "infer",
    sep: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a DataFrame from a CSV file at the specified file path.

    Parameters
    ----------
    file_path : str, Path
        The path to the CSV file to be loaded.
    index_col : int, str, or None, default=None
        Column(s) to set as index (row labels). If None, no columns are used as the index.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if index_col is not None:
        if not isinstance(index_col, Sequence) or isinstance(index_col, str):
            index_col = [index_col]
        if header != "infer" and isinstance(index_col, Sequence):
            index_col = list(range(len(index_col)))

    if sep is None:
        sep = FILE_SEP_DICT.get(
            file_path.suffix, ","
        )  # Default to comma if extension is unrecognized

    try:
        logger.debug(f"Loading DataFrame from file at '{file_path.relative_to(PROJ_ROOT)}'...")
        df = pd.read_csv(file_path, index_col=index_col, sep=sep, header=header, **kwargs)
        logger.info(f"Loaded DataFrame successfully from '{file_path.relative_to(PROJ_ROOT)}'.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    except Exception as e:
        raise IOError(f"An error occurred while loading the DataFrame: '{e}'.")


def save_dataframe_to_file(
    df: pd.DataFrame, destination_path: str | Path, index: bool = False, sep: str | None = None
) -> None:
    """Save a DataFrame to a CSV file at the specified destination path.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be saved.
    destination_path : str, Path
        The path to the destination CSV file where the DataFrame should be saved.
    index : bool, default=False
        Whether to include the DataFrame index in the saved CSV file.
    """
    if isinstance(destination_path, str):
        destination_path = Path(destination_path)

    if sep is None:
        sep = FILE_SEP_DICT.get(
            destination_path.suffix, ","
        )  # Default to comma if extension is unrecognized

    try:
        logger.debug(f"Saving DataFrame to file at '{destination_path.relative_to(PROJ_ROOT)}'...")
        df.to_csv(destination_path, index=index, sep=sep)
        logger.info(
            f"Saved DataFrame successfully to '{destination_path.relative_to(PROJ_ROOT)}'."
        )
    except Exception as e:
        raise IOError(f"An error occurred while saving the DataFrame: '{e}'.")


def copy_file_to_destination(source_file: str | Path, destination_path: str | Path) -> None:
    """Copy a file to a specified directory.

    Parameters
    ----------
    source_file : str, Path
        The path to the source file to be copied.
    destination_path : str, Path
        The path to the destination directory where the file should be copied.
        If a different filename is desired, include it in the destination_path.
    """
    if isinstance(source_file, str):
        source_file = Path(source_file)
    if isinstance(destination_path, str):
        destination_path = Path(destination_path)

    if source_file.suffix == destination_path.suffix:
        try:
            # Copy the file
            logger.debug(
                f"Copying file '{source_file.relative_to(PROJ_ROOT)}' to '{destination_path.relative_to(PROJ_ROOT)}'..."
            )
            shutil.copy(source_file, destination_path)
            # Log success message
            logger.info(
                f"File '{source_file.relative_to(PROJ_ROOT)}' copied successfully to '{destination_path.relative_to(PROJ_ROOT)}'."
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"The source file '{source_file}' does not exist.")
    else:
        raise ValueError(
            f"Source file '{source_file}' and destination '{destination_path}' must have the same file extension."
        )


def make_filename(*args: str, sep: str = "-", filetype: str = "csv") -> str:
    filename = sep.join(args)
    return f"{filename}.{filetype}"
