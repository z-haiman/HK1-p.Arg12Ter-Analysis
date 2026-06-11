import re

from inmoose import limma
from loguru import logger
import numpy as np
import pandas as pd
from patsy import dmatrix


def fit_limma_model_for_genotype(data, genotype, control_group, covariates=None, contrasts=None):
    """TODO DOCSTRING."""
    if covariates:
        check_values = [x for x in covariates if x not in data.columns]
        if check_values:
            raise ValueError(f"covariates {check_values} not found in data.")
    else:
        covariates = set()
    if contrasts:
        check_values = set(
            [x for contrast in list(contrasts) for x in re.findall(r"Group\[(\S+)\]", contrast)]
        )
        check_values = set(data[genotype].dropna().astype(str).unique()).difference(check_values)
        if check_values:
            raise ValueError(f"{check_values} not found in group {genotype} values.")
    else:
        contrasts = []
    # Create Design Matrix (Formula: ~0 + Group + Covariate + Covariate + ...)
    # ~0+Group creates an ANOVA-style fit (intercept for each group)
    # Rename group to 'Group' to avoid issues with genotype IDs
    df_meta = data.loc[:, [genotype] + list(covariates)].rename({genotype: "Group"}, axis=1)
    df_vars = data.loc[:, ~data.columns.isin([genotype] + list(covariates))]

    for col, series in df_meta.items():
        logger.debug(f"For '{col}' there are '{series.isna().sum()}' samples with NaN values...")
    series_sample_na = df_meta.isna().any(axis=1)
    df_meta = df_meta[~series_sample_na].copy()
    logger.info(
        f"Removed '{series_sample_na.sum()}' samples due to NaN values, '{df_meta.shape[0]}' samples remain."
    )
    # Ensure samples in metadata and variables are aligned, then transpose so variable columns are samples.
    df_vars = df_vars.loc[df_meta.index]
    # Ensure variable values are numerical
    df_vars = df_vars.T.astype(float)

    # Format metadata
    df_meta["Group"] = pd.Categorical(df_meta["Group"])
    if control_group is None:
        # Use group with most values if control group is not provided.
        control_group = df_meta["Group"].value_counts()
        control_group = control_group[control_group == control_group.max()]
        if len(control_group) > 1:
            raise ValueError(
                f"Cannot determine control group from value counts, ties detected for the following: {control_group}"
            )
        control_group = control_group.idxmax()
    logger.info(f"Setting '{genotype} = {control_group}' as the control group.")
    df_meta["Group"] = df_meta["Group"].cat.reorder_categories(
        [control_group] + [c for c in df_meta["Group"].cat.categories if c != control_group]
    )

    formula = " + ".join(["Group"] + list(set(covariates)))
    logger.debug(f"Setting up design matrix with formula {formula}")

    design = dmatrix(f"~ 0 + {formula}", df_meta)
    # Fit linear model
    logger.debug("Fitting linear model...")
    fit = limma.lmFit(df_vars, design)
    logger.info("Fitted linear model.")
    # Create constrast matrix
    if contrasts:
        logger.debug("Creating contrast matrix...")
        contrast_matrix = limma.makeContrasts(
            list(contrasts), levels=design.design_info.column_names
        ).astype(int)
        fit = limma.contrasts_fit(fit, contrast_matrix)
        logger.info("Created contrast matrix.")
    # Perform eBayes
    logger.debug("Performing empirical bayes moderation...")
    fit_ebayes = limma.eBayes(fit)
    logger.info("Performed empirical bayes moderation.")
    # Create results table
    logger.debug("Creating results tables...")
    results_table = limma.topTable(fit_ebayes, coef=None, number=np.inf)
    if not contrasts:
        return results_table
    results_dict = {"Overall": results_table}
    results_dict.update(
        {
            contrast_label: limma.topTable(fit_ebayes, coef=contrast_label, number=np.inf)
            for contrast_label in contrasts
        }
    )

    logger.info("Created results tables.")
    return results_dict


def significance_for_covariate_adjustments(df_pvalues, alpha):
    """Label significance of p-values for a given alpha after covariate adjustment.

    Parameters
    ----------
    df_pvalues : pd.DataFrame
        DataFrame with columns for unadjusted and adjusted p-values at index 0 and 1 respectively.
    alpha : float
        Significance level for determining significance.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column indicating significance.
    """
    sig_col = "Significance"
    logger.debug(f"Comparing p-values to significance level of {alpha}...")
    is_significant = df_pvalues < alpha
    columns = df_pvalues.columns.tolist()
    df_pvalues[sig_col] = np.nan
    sig_unadj = is_significant[columns[0]]
    sig_adj = is_significant[columns[1]]
    # Assign significance values
    signifiance_dict = {
        "Significant": sig_unadj & sig_adj,
        "Sig. when adjusted": ~sig_unadj & sig_adj,
        "Non-sig. when adjusted": sig_unadj & ~sig_adj,
        "Non-significant": ~sig_unadj & ~sig_adj,
    }
    for sig_str, sig_series in signifiance_dict.items():
        df_pvalues.loc[sig_series, sig_col] = sig_str

    df_pvalues[sig_col] = pd.Categorical(df_pvalues[sig_col])
    df_pvalues[sig_col] = df_pvalues[sig_col].cat.reorder_categories(
        [s for s in list(signifiance_dict) if s in df_pvalues[sig_col].unique()], ordered=True
    )
    df_pvalues = df_pvalues.sort_index().sort_values(by=df_pvalues.columns[::-1].tolist())
    logger.info("Determined significance.")
    return df_pvalues
