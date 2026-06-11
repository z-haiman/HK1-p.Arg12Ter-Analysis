from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hk1_r12ter_analysis.util import ensure_iterable


def identify_independent_snps(
    df_r2_matrix: pd.DataFrame, list_of_snps: list[str], r2_threshold: float = 0.8
) -> list[str]:
    """Identify independent SNPs based on linkage disequilibrium correlations.

    Parameters
    ----------
    df_r2_matrix : pd.DataFrame
        A square DataFrame where both the index and columns are SNP identifiers, and the values are the r^2 correlation between the SNPs.
    list_of_snps : list of str
        A list of SNP identifiers to consider for independence.
    r2_threshold : float, optional
        The threshold for r^2 above which two SNPs are considered to be in linkage disequilibrium (default is 0.8). SNPs with r^2 above this threshold are not considered independent.

    Returns
    -------
    list of str
        A list of SNP identifiers that are considered independent based on the r^2 threshold.
    """
    keep_snps = []
    df_r2_matrix = df_r2_matrix.loc[list_of_snps, list_of_snps]
    for idx1, snp1 in enumerate(list_of_snps):
        is_independent = True
        # logger.debug(f"Determining if SNP {snp1} is independent from other kept SNPs...")
        for kept in keep_snps:
            idx2 = list_of_snps.index(kept)
            snp2 = list_of_snps[idx2]
            if snp1 == snp2 or idx1 == idx2:
                raise IndexError("Cannot compare identical SNPs.")
            # logger.debug(f"Checking correlation between {snp1} and {snp2}...")
            r2 = df_r2_matrix.loc[snp1, snp2]
            if r2 >= r2_threshold:
                # logger.info(f"Correlation between SNPs {snp1} and {snp2} is above threshold ({r2} >= {r2_threshold}), not considered independent.")
                is_independent = False
                break
            # else:
            # logger.info(f"Correlation between SNPs {snp1} and {snp2} is below threshold ({r2} < {r2_threshold}), considered independent.")
        if is_independent:
            # logger.info(f"SNPs {snp1} is considered indepenedent from other kept SNPs.")
            keep_snps += [snp1]

    return keep_snps


def compute_pairwise_LD_rogers_huff(
    dosage_matrix: np.ndarray | pd.DataFrame,
    rsids: list[str] | None = None,
    return_LD_coefficients: bool = True,
) -> pd.DataFrame:
    """Compute the linkage disequilibrium for SNPs using the Rogers and Huff method.

    Parameters
    ----------
    dosage_matrix : array-like
        Matrix of SNP dosage data (0, 1, 2) with samples as rows and SNPs as columns.
    rsids : list, default None
        List of SNPs corresponding to the columns of the matrix. If not provided and `dosage_matrix` is a DataFrame columns values will be used.
    return_LD_coefficients : bool, default True
        If True, include the LD coefficient (D) an the normalized LD coefficient (D') in the returned DataFrame.

    Returns
    -------
    ld_matrix : pd.DataFrame
        A DataFrame with a MultiIndex of SNP pairs, and columns corresponding to Pearson r and r2 values as well as D and D' values if requested.

    Notes
    -----
    The r and r2 values corresponds the Pearson correlation coefficient and its squared value. D and D' are then calculated accordingly.
    """
    if not isinstance(dosage_matrix, pd.DataFrame):
        dosage_matrix = pd.DataFrame(dosage_matrix)
    if rsids is None:
        rsids = list(dosage_matrix.columns)
    #  Calculate the r2 correlation matrix
    r_matrix = dosage_matrix.corr(method="pearson")
    r2_matrix = r_matrix**2
    matrices = [r_matrix, r2_matrix]
    cols = ["r", "r2"]
    if return_LD_coefficients:

        # Calculate the linkage disequilibrium coefficnients (D) values.
        df_var = dosage_matrix.var(axis=0, skipna=True, ddof=1)
        D_matrix = r_matrix * np.sqrt(np.outer(df_var, df_var)) / 4

        # Get the minor allele frequencies for each SNP
        series_maf = dosage_matrix.mean(axis=0, skipna=True) / 2
        # Calculate the normalized D' values.
        Dprime_matrix = D_matrix.div(
            np.select(
                condlist=[D_matrix > 0, D_matrix < 0],
                choicelist=[
                    np.minimum(
                        np.outer(series_maf, 1 - series_maf),
                        np.outer(series_maf, 1 - series_maf).T,
                    ),
                    np.minimum(
                        np.outer(series_maf, series_maf), np.outer(1 - series_maf, 1 - series_maf)
                    ),
                ],
                default=0,
            )
        ).replace({np.inf: 0, -np.inf: 0})
        matrices += [D_matrix, Dprime_matrix]
        cols += ["D", "D'"]

    # Combine result matriices into one.
    ld_matrix = pd.concat(
        [
            matrix.melt(ignore_index=False).set_index(["variable"], append=True)
            for matrix in matrices
        ],
        axis=1,
    )
    ld_matrix.columns = cols
    ld_matrix.index.names = ["SNP1", "SNP2"]
    # Reorder matrix to match order of SNPs
    ld_matrix = ld_matrix.loc[pd.IndexSlice[rsids, rsids], :]
    # Remove rows with identical SNPs being compared
    ld_matrix = ld_matrix.loc[
        ld_matrix.index.get_level_values(level=0) != ld_matrix.index.get_level_values(level=1)
    ].copy()
    return ld_matrix


def compute_pairwise_LD_expectation_maximization(
    dosage_matrix: np.ndarray | pd.DataFrame,
    rsids: list[str] | None = None,
    return_LD_coefficients: bool = True,
) -> pd.DataFrame:
    """Compute the linkage disequilibrium for SNPs using an expectation maximization algorithm.

    Parameters
    ----------
    dosage_matrix : array-like
        Matrix of SNP dosage data (0, 1, 2) with samples as rows and SNPs as columns.
    rsids : list, default None
        List of SNPs corresponding to the columns of the matrix. If not provided and `dosage_matrix` is a DataFrame columns values will be used.
    return_LD_coefficients : bool, default True
        If True, include the LD coefficient (D) an the normalized LD coefficient (D') in the returned DataFrame.

    Returns
    -------
    ld_matrix : pd.DataFrame
        A DataFrame with a MultiIndex of SNP pairs, and columns corresponding to Pearson r and r2 values as well as D and D' values if requested.
    """
    # def em_algorithm_two_snps(genotypes, max_iter=100, tol=1e-6):
    #     """
    #     EM Algorithm to estimate haplotype frequencies from unphased 2-SNP data.
    #     genotypes: Nx2 array, where each row is an individual, columns are SNPs (0,1,2)
    #     """
    #     # 1. Initialize haplotype frequencies (assume linkage equilibrium)
    #     # Haplotypes: 00, 01, 10, 11 (minor/major alleles)
    #     n = genotypes.shape[0]
    #     # Simple initialization: 0.25 each
    #     p = np.array([0.25, 0.25, 0.25, 0.25])

    #     # 2. EM Iterations
    #     for i in range(max_iter):
    #         old_p = p.copy()

    #         # Expectation: Estimate expected counts for ambiguous genotypes
    #         expected_counts = np.zeros(4)
    #         for g in genotypes:
    #             s1, s2 = g[0], g[1]

    #             # Cases based on unphased data
    #             if s1 == 1 and s2 == 1: # Double Heterozygote (Ambiguous)
    #                 # Probabilities of the two possible phases
    #                 prob_phase1 = 2 * p[0] * p[3] # 00/11
    #                 prob_phase2 = 2 * p[1] * p[2] # 01/10
    #                 total = prob_phase1 + prob_phase2

    #                 if total > 0:
    #                     expected_counts[0] += prob_phase1 / total
    #                     expected_counts[3] += prob_phase1 / total
    #                     expected_counts[1] += prob_phase2 / total
    #                     expected_counts[2] += prob_phase2 / total
    #             else:
    #                 # Unambiguous cases (e.g., 00, 02, 22, 10, etc.)
    #                 # Add 2 to the corresponding haplotype count
    #                 # Simplified representation here
    #                 if s1==0 and s2==0: expected_counts[0] += 2
    #                 elif s1==0 and s2==1: expected_counts[0] += 1; expected_counts[1] += 1
    #                 elif s1==0 and s2==2: expected_counts[1] += 2
    #                 elif s1==2 and s2==0:
    #                 elif s1==2 and s2==1:
    #                 elif s1==2 and s2==2:

    #                 # ... (add other cases for 10, 12, 20, 21, 22)
    #                 # A robust implementation maps s1, s2 combinations to haplotype pairs
    #                 pass # Placeholder for brevity, see standard genetic libraries

    #         # Maximization: Update haplotype frequencies
    #         p = expected_counts / (2 * n)

    #         # Check convergence
    #         if np.linalg.norm(p - old_p, ord=1) < tol:
    #             break

    #     return p # [p00, p01, p10, p11]

    # # Example Data: [SNP1, SNP2]
    # # 0: Hom Ref, 1: Het, 2: Hom Alt
    # data = np.array([
    #     [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2],
    # ])
    # haplotype_freqs = em_algorithm_two_snps(data)
    # haplotype_freqs
    raise NotImplementedError


def compute_pairwise_spearman_correlation_genomics(
    dosage_matrix: np.ndarray | pd.DataFrame, rsids: list[str] | None = None
) -> pd.DataFrame:
    """Compute the spearman correlation coefficient for each pair of SNPs.

    Parameters
    ----------
    dosage_matrix : array-like
        Matrix of SNP dosage data (0, 1, 2) with samples as rows and SNPs as columns.
    rsids : list, default None
        List of SNPs corresponding to the columns of the matrix. If not provided and `dosage_matrix` is a DataFrame columns values will be used.

    Returns
    -------
    spearman_matrix : pd.DataFrame
        A DataFrame with a MultiIndex of SNP pairs, and columns corresponding to Spearman rho values.
    """
    if not isinstance(dosage_matrix, pd.DataFrame):
        dosage_matrix = pd.DataFrame(dosage_matrix)
    if rsids is None:
        rsids = list(dosage_matrix.columns)
    #  Calculate the r2 correlation matrix
    spearman_matrix = dosage_matrix.corr(method="spearman")
    spearman_matrix = spearman_matrix.melt(ignore_index=False).set_index(["variable"], append=True)
    spearman_matrix.columns = ["Spearman rho"]
    spearman_matrix.index.names = ["SNP1", "SNP2"]
    # Reorder matrix to match order of SNPs
    spearman_matrix = spearman_matrix.loc[pd.IndexSlice[rsids, rsids], :]
    # Remove rows with identical SNPs being compared
    spearman_matrix = spearman_matrix.loc[
        spearman_matrix.index.get_level_values(level=0)
        != spearman_matrix.index.get_level_values(level=1)
    ].copy()
    return spearman_matrix


def compute_pairwise_linkage_disequilibrium(
    dosage_matrix: pd.DataFrame,
    method: Literal["RH", "EM"] = "RH",
    rsids: list[str] | None = None,
    return_LD_coefficients: bool = True,
) -> pd.DataFrame:
    """Compute the linkage disequilibrium for SNPs using dosage data i.e., 0, 1, 2)

    Parameters
    ----------
    dosage_matrix : array-like
        Matrix of SNP dosage data (0, 1, 2) with samples as rows and SNPs as columns.
    method : {'RH', 'EM'}, default 'RH'
        The method to utilize for computing pairwise linkage disequilibrium between two biallelic SNPs.
            * 'RH': Use the Rogers and Huff method to calculate linkage disquilibrium via covariance
            * 'EM': Use the Expectation Maximization algorithm to calculate linkage disquilibrium by estimating haplotype frequencies.
    rsids : list, default None
        List of SNPs corresponding to the columns of the matrix. If not provided and `dosage_matrix` is a DataFrame columns values will be used.
    return_LD_coefficients : bool, default True
        If True, include the LD coefficient (D) an the normalized LD coefficient (D') in the returned DataFrame.


    """
    if rsids is None:
        rsids = dosage_matrix.columns.tolist()
    if method == "RH":
        ld_matrix = compute_pairwise_LD_rogers_huff(
            dosage_matrix, rsids=rsids, return_LD_coefficients=return_LD_coefficients
        )
    elif method == "EM":
        ld_matrix = compute_pairwise_LD_expectation_maximization(
            dosage_matrix, rsids=rsids, return_LD_coefficients=return_LD_coefficients
        )
    else:
        raise ValueError(
            f"Unrecognized value for method: '{method}'. Must be one of the following: ['RH', 'EM']."
        )

    return ld_matrix


def validate_chromosome(chromosome: str | list[str] | None) -> list[str]:
    """
    Checks if input is a single chromosome or a list/array of chromosome.
    Returns: bool, list (normalized list of chromosome)
    """
    # Convert single input to list for unified processing
    if isinstance(chromosome, (str, int)):
        chromosome = [str(chromosome)]
    elif isinstance(chromosome, (list, np.ndarray)):
        chromosome = [str(c) for c in chromosome]
    else:
        return []

    # Regex or direct string manipulation to validate
    # Defines valid chromosome: 1-22, X, Y, M, or prefixed with 'chr'
    valid_numbers = {str(i) for i in range(1, 23)} | {"X", "Y", "M", "MT"}

    normalized = []
    for item in chromosome:
        # Strip 'chr' prefix to check core name
        core_name = item.lower().replace("chr", "").upper()

        # Additional check for 'MT' -> 'M' standardization if needed
        if core_name == "MT":
            core_name = "M"

        if core_name not in (valid_numbers | {"MT"}):
            return []
        normalized.append(core_name)

    chr_nums = set(normalized).difference(["X", "Y", "M"])
    chr_lets = set(normalized).intersection(["X", "Y", "M"])
    chr_nums = [str(x) for x in sorted([int(x) for x in chr_nums])]
    normalized = chr_nums + sorted(chr_lets)
    return normalized


# Create figure and subplots
def create_figure_for_LDheatmap_w_chromosome_mapping(
    chromosome: str | list[str] | None = None,
    chromosome_locx: Literal["top", "bottom", "t", "b"] = "top",
    chromosome_locy: Literal["left", "right", "l", "r"] = "left",
    chromosome_dim: float = 0.5,
    cbar_dim: float = 0.25,
    spacer: float = 0.1,
    figsize: tuple[float, float] = (10, 10),
) -> tuple[mpl.figure.Figure, list[mpl.axes.Axes]]:
    """Create a figure object with subplots to use for visualizing pairwise linkage disequilibrium with SNPs mapped to chromosomes.

    Parameters
    ----------
    chromosome : int, str, set
        The chromosome(s) where the SNPs are located on.
    chromosome_locx : {'top', t', 'bottom', 'b'}, default 'top'
        Whether to place the chromosome graphic above ot below the LD heatmap. The colorbar is placed on the opposite side.
    chromosome_locy : {'left', l', 'right', 'r'}, default 'left'
        Whether to place the chromosome graphic to the left or the right of the LD heatmap. THe colorbar is placed on the opposite side.
    chromosome_dim : float, default
        The dimension of the chromosome graphic. Corresponds to the height of the x-axis graphic and the width of the y-axis graphic.
    cbar_dim : float
        The dimension of the colorbar. Corresponds to the height of the x-axis colorbar and the width of the y-axis colorbar.
    spacer : float
        The amount of space between subplots. Passed to both the `hspace` and `wspace` arguments for the `matplotlib.gridspect.GridSpec` object
    figsize : tuple of floats, default (10, 10)
        The size of the figure to return. Ideally, the x and y dimensions are identical.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : list of matplotlib.axes.Axes objects
        The generated axes for the figure. Returned in the order of (ax_LDheatmap, ax_chr_x, ax_chr_y, ax_cbar_x, ax_cbar_y) where:
            * ax_LDheatmap : Axes object for the linkage disequilibrium heatmap.
            * ax_chr_x : list of Axes object for the chromosomes along the x-axis.
            * ax_chr_y : list of Axes object for the chromosomes along the y-axis.
            * ax_cbar_x : Axes object for the colorbar along the x-axis.
            * ax_cbar_y : Axes object for the colorbar along the y-axis.

    """
    for arg_name, arg_yalue, valid in zip(
        ["chromosome_locx", "chromosome_locy"],
        [chromosome_locx, chromosome_locy],
        [{"top", "bottom", "t", "b"}, {"left", "right", "l", "r"}],
    ):
        if arg_yalue not in valid:
            raise ValueError(f"`{arg_name}` must be one of the following: {valid}")

    for arg_name, arg_value in zip(
        ["chromosome_dim", "cbar_dim", "spacer"], [chromosome_dim, cbar_dim, spacer]
    ):
        if arg_value is not None and float(arg_value) < 0:
            raise ValueError(f"`{arg_name}` must be a non-negative numerical value")

    ndims = 9
    if len(chromosome) > 3:
        ndims += 2 * (len(chromosome) - 3)

    ratios = (
        [chromosome_dim]
        + [cbar_dim, chromosome_dim - spacer]  # 2 spacer - 1 spacer
        + [cbar_dim + chromosome_dim] * (ndims - 5)  # 5 axes minimum defined
        + [chromosome_dim - spacer, cbar_dim]  # 2 spacer - 1 spacer
    )
    fig = plt.figure(figsize=figsize)
    gs = mpl.gridspec.GridSpec(
        nrows=ndims,
        ncols=ndims,
        figure=fig,
        hspace=spacer,
        wspace=spacer,
        width_ratios=ratios if chromosome_locy[0] == "l" else ratios[::-1],
        height_ratios=ratios if chromosome_locx[0] == "t" else ratios[::-1],
    )

    # Create variables for indicies used to slice gridspec
    (gsr1, gsr2) = (2, ndims - 1) if chromosome_locx[0] == "t" else (1, ndims - 2)
    (gsc1, gsc2) = (2, ndims - 1) if chromosome_locy[0] == "l" else (1, ndims - 2)

    # Add the subplot for the heatmap
    ax_LDheatmap = fig.add_subplot(gs[gsr1:gsr2, gsc1:gsc2])
    # Add the subplot(s) for the chromosome(s)
    # Iterator used depends on number of chromosomes due to changes in plot dimensions
    if len(chromosome) <= 2:
        iter_chr_gs = np.arange(0, 3 * len(chromosome), 3)
    else:
        iter_chr_gs = np.arange(0, (len(chromosome) - 1) * 2 + 2, 2)

    ax_chromosome_x, ax_chromosome_y = ([], [])
    # TODO see if axes deletions are necessary or adding the subplots can just be avoided
    for i, j in zip(iter_chr_gs, iter_chr_gs[::-1]):
        ind = 0 if chromosome_locx[0] == "t" else ndims - 1
        slice1 = int(gsc1 - 1 if i == 0 else gsc1 + i)
        slice2 = int(gsc2 + 1 if j == 0 else gsc2 - j)
        ax_chromosome_x += [fig.add_subplot(gs[ind, slice1:slice2])]

        ind = 0 if chromosome_locy[0] == "l" else ndims - 1
        slice1 = int(gsr1 - 1 if i == 0 else gsr1 + i)
        slice2 = int(gsr2 + 1 if j == 0 else gsr2 - j)
        ax_chromosome_y += [fig.add_subplot(gs[slice1:slice2, ind])]

    # Add subplots for colorbars
    ax_cbar_x = fig.add_subplot(gs[0 if chromosome_locx[0] == "b" else ndims - 1, gsc1:gsc2])
    ax_cbar_y = fig.add_subplot(gs[gsr1:gsr2, 0 if chromosome_locy[0] == "r" else ndims - 1])
    # TODO see if axes deletions are necessary or adding the subplots can just be avoided
    if not chromosome:
        for ax_to_delete in ax_chromosome_x + ax_chromosome_y:
            fig.delaxes(ax_to_delete)
        ax_chromosome_x = []
        ax_chromosome_y = []

    axes = (ax_LDheatmap, ax_chromosome_x, ax_chromosome_y, ax_cbar_x, ax_cbar_y)
    return fig, axes


def format_matrices_for_LDheatmap(
    data: pd.DataFrame,
    value_type: str | list[str],
    ordered_snps: list[str],
    dropna: bool = True,
) -> list[pd.DataFrame]:
    """Format matrices for LD heatmap visualization.

    Parameters
    ----------
    data : pd.DataFrame
    value_type : str, list[str]
        Either a string or list of strings corresponding to the data column to use for matrix values. If a list is provided, the data values are assigned as [triu_values, tril_values] where triu_values corresponds to the values to be plotted in the upper triangle of the heatmap and tril_values corresponds to the values to be plotted in the lower triangle of the heatmap.
    ordered_snps : list[str]
    dropna : bool, default True
         Whether to drop rows and columns with all missing values. If True, rows and columns with all missing values are dropped from the matrices. If False, missing values are retained and will be plotted as the specified `nan_color` in the heatmap.

    Returns
    -------
    list[pd.DataFrame]
        A list of formatted matrices for LD heatmap visualization.
        If `value_type` is a single string, a list with one DataFrame is returned. If `value_type` is a list of two strings, a list with two DataFrames is returned where the first DataFrame corresponds to the values to be plotted in the upper triangle of the heatmap and the second DataFrame corresponds to the values to be plotted in the lower triangle of the heatmap.

    """
    snps = ["SNP1", "SNP2"]
    if data.index.names == snps:
        data = data.reset_index(drop=False)

    value_type = ensure_iterable(value_type)
    if set(value_type).difference(data.columns):
        raise ValueError(f"'{value_type}' not in data columns")
    matrices = data.loc[:, snps + value_type]
    matrices = [
        matrices.pivot(index="SNP1", columns="SNP2", values=values) for values in value_type
    ]
    matrices = [matrix.loc[ordered_snps, ordered_snps] for matrix in matrices]
    if dropna:
        matrices = [
            matrix.dropna(how="all", axis=0).dropna(how="all", axis=1) for matrix in matrices
        ]
    if len(matrices) == 2:
        assert matrices[0].shape == matrices[1].shape, "Matricies must be the same shape."
    return matrices


def create_LD_heatmap_w_chromosom_mapping(
    data: pd.DataFrame,
    value_type: str | list[str],
    ordered_positions: pd.Series,
    chromosome: str | list[str] | None,
    chromosome_locx: Literal["top", "bottom", "t", "b"] = "top",
    chromosome_locy: Literal["left", "right", "l", "r"] = "left",
    chromosome_position_rounding_factor: float = 1e4,
    figsize: tuple[float, float] = (10, 10),
    snp_color: str = "green",
    snp_linewidth: float = 0.5,
    connector_color: str = "grey",
    connector_linewidth: float | None = None,
    nan_color: str = "grey",
    diag_color: str = "black",
    cmap: list[str] = ["Reds", "Blues"],
    dropna: bool = True,
    label_fontproperties: dict | None = None,
    label_fontdict: dict | None = None,
    tick_fontdict: dict | None = None,
    grid_kwargs: dict | None = None,
):
    """Visualize pairwise linkage disequilibrium with SNPs mapped to chromosomes.

    Parameters
    ----------
    data : pandas.DataFrame

    value_type : str, list[str]
        Either a string or list of strings corresponding to the data column to use for matrix values.
        If a list is provided, the data values are assigned as [
    ordered_positions :
    chromosome : str, list[str], None
        Either a string or list of strings corresponding to the chromosomes to use. If None provided, no chromosome plots are made.
    chromosome_locx : {'top', 't', 'bottom', 'b'}
        The location of the horizontal chromosome graphic. Either above or below the heatmap.
        The colobar is placed on the opposite side.
    chromosome_locy : {'left', 'l', 'right', 'r'}
        The location of the vertical chromosome graphic. Either to the left or right the heatmap. The colobar is placed on the opposite side.
    chromosome_dim : float
        A float representing the relative dimension of the chromosome plot.
        Corresponds to the height for the chromosome above/below the heatmap, and corresponds to the width for the chromosome left/right of the heatmap.
    chromosome_loc_rounding_factor : float
        A float representing the factor to round chromosome positions to for visualization purposes. For example, a value of 1e4 rounds positions to the nearest 10kb.
    figsize : tuple of floats, default (10, 10)
        The size of the figure to return. Ideally, the x and y dimensions are identical.
    snp_color : str
        The color to use for the SNP position markers on the chromosome graphic.
    connector_color : str
        The color to use for the connector lines between the chromosome graphic and the heatmap.
    diag_color : str
        The color to use for the diagonal of the heatmap. By default, the diagonal is colored black to differentiate it from the rest of the heatmap since the diagonal corresponds to the correlation of SNPs with themselves.
    nan_color : str
        The color to use for missing values in the heatmap. By default, missing values are colored grey. This is particularly important to set if `dropna` is set to False since missing values will be plotted as the specified `nan_color` in the heatmap.
    cmap :  str, list of str, or list of matplotlib.colors.Colormap objects
        The colormap(s) to use for the heatmap. If a single string or colormap object is provided, the same colormap is used for the entire heatmap. If a list of two strings or colormap objects is provided, the first colormap is used for the upper triangle of the heatmap and the second colormap is used for the lower triangle of the heatmap.
    dropna : bool
        Whether to drop rows and columns with all missing values. If True, rows and columns with all missing values are dropped from the heatmap. If False, missing values are retained and will be plotted as the specified `nan_color` in the heatmap.
    label_fontproperties : dict, optional
        The font properties for the axis labels.
    label_fontdict : dict, optional
        The font dictionary for the axis labels.
    tick_fontdict : dict, optional
        The font dictionary for the tick labels.
    grid_kwargs : dict, optional
        The keyword arguments for the grid lines.
    """
    kwargs = dict(
        label_fontproperties=(
            label_fontproperties if label_fontproperties is not None else {"weight": "bold"}
        ),
        label_fontdict=label_fontdict if label_fontdict is not None else {"size": "large"},
        tick_fontdict=tick_fontdict if tick_fontdict is not None else {"size": "medium"},
        grid_kwargs=(
            grid_kwargs if grid_kwargs is not None else {"color": "xkcd:black", "linewidth": 0.1}
        ),
    )

    if chromosome:
        valid = validate_chromosome(chromosome)
        if not chromosome:
            raise ValueError(
                "Unrecognized value for chromosome. Must be a value between 1 and 22, or letter X, Y, M, or MT. Only valid prefix is 'chr'"
            )
        chromosome = valid
    else:
        chromosome = []
        chromosome_locx = "top" if chromosome_locx is None else chromosome_locx
        chromosome_locy = "left" if chromosome_locy is None else chromosome_locy

    matrices = format_matrices_for_LDheatmap(
        data,
        value_type=value_type,
        ordered_snps=ordered_positions.index,
        dropna=dropna,
    )
    if isinstance(cmap, mpl.colors.Colormap):
        cmap = [cmap]
    else:
        cmap = ensure_iterable(cmap)
    if connector_linewidth is None:
        connector_linewidth = snp_linewidth / 2
    fig, axes = create_figure_for_LDheatmap_w_chromosome_mapping(
        chromosome=chromosome,
        chromosome_locx=chromosome_locx,
        chromosome_locy=chromosome_locy,
        chromosome_dim=0.5,
        cbar_dim=0.25,
        spacer=0.1,
        figsize=figsize,
    )
    (ax_LDheatmap, ax_chromosome_x, ax_chromosome_y, ax_cbar_x, ax_cbar_y) = axes

    # Plot LD heatmap
    if len(matrices) == 2:
        triu_im, tril_im = matrices
        cmap_im = mpl.colormaps.get_cmap(cmap[0])
        cmap_im.set_bad(nan_color)
        triu_im = ax_LDheatmap.imshow(
            np.ma.masked_array(triu_im, mask=np.triu(np.ones_like(triu_im, dtype=bool))),
            cmap=cmap_im,
            vmin=0,
            vmax=1,
        )
        cmap_im = mpl.colormaps.get_cmap(cmap[1])
        tril_im = ax_LDheatmap.imshow(
            np.ma.masked_array(tril_im, mask=np.tril(np.ones_like(tril_im, dtype=bool))),
            cmap=cmap_im,
            vmin=0,
            vmax=1,
        )
    else:
        triu_im, tril_im = matrices[0], None
        cmap_im = mpl.colormaps.get_cmap(cmap[0])
        cmap_im.set_bad(nan_color)
        triu_im = ax_LDheatmap.imshow(
            np.ma.masked_array(triu_im),
            cmap=cmap_im,
            vmin=0,
            vmax=1,
        )
    diag_matrix = np.eye(*matrices[0].shape, dtype=bool)
    ax_LDheatmap.imshow(
        np.ma.masked_array(diag_matrix, mask=~diag_matrix),
        cmap=mpl.colors.ListedColormap([diag_color]),
    )
    ax_LDheatmap.set_xticks(np.arange(0, len(matrices[0]), 1))
    ax_LDheatmap.set_yticks(np.arange(0, len(matrices[0]), 1))
    # Explicitly set limits for axes, reverse ylims
    ax_LDheatmap.set_xlim(ax_LDheatmap.get_xlim())
    ax_LDheatmap.set_ylim(ax_LDheatmap.get_ylim()[::-1])
    # Set spine visibility
    for spine in ["top", "left", "bottom", "right"]:
        ax_LDheatmap.spines[spine].set_visible(True)

    # Set major and minor ticks. Major ticks are mapped to chromosomes, minor ticks are used for gridlines
    # Set xtick positions, but remove labels and tick marks
    ax_LDheatmap.set_xticks(ax_LDheatmap.get_xticks(), minor=False, labels=[])
    ax_LDheatmap.set_xticks(ax_LDheatmap.get_xticks() - 0.5, minor=True, labels=[])
    ax_LDheatmap.xaxis.set_tick_params(length=0, which="both")
    # Set ytick positions, but remove labels tick marks
    ax_LDheatmap.set_yticks(ax_LDheatmap.get_yticks(), minor=False, labels=[])
    ax_LDheatmap.set_yticks(ax_LDheatmap.get_yticks() - 0.5, minor=True, labels=[])
    ax_LDheatmap.yaxis.set_tick_params(length=0, which="both")
    # Set grid
    ax_LDheatmap.grid(which="minor", **kwargs.get("grid_kwargs"))

    # Set labels
    ax_LDheatmap.set_xlabel(
        "SNP1",
        fontdict=kwargs.get("label_fontdict"),
        fontproperties=kwargs.get("label_fontproperties"),
    )
    ax_LDheatmap.xaxis.set_label_position("top" if bool(chromosome_locx[0] == "t") else "bottom")
    ax_LDheatmap.set_ylabel(
        "SNP2",
        fontdict=kwargs.get("label_fontdict"),
        fontproperties=kwargs.get("label_fontproperties"),
    )
    ax_LDheatmap.yaxis.set_label_position("left" if bool(chromosome_locy[0] == "l") else "right")
    # try:
    #     ordered_positions = ordered_positions.groupby("#CHROM")["POS"].agg(list).to_dict()
    # except KeyError:
    #     ordered_positions = {None: ordered_positions["POS"].values}

    # Plot chomosome for x-axis
    if ax_chromosome_x and ax_chromosome_y:

        try:
            ordered_positions = ordered_positions.groupby("#CHROM")["POS"].agg(list).to_dict()
        except AttributeError:
            pass

        snp_color = ensure_iterable(snp_color)
        if len(snp_color) != len(ordered_positions):
            snp_color = snp_color * len(ordered_positions)

        connector_color = ensure_iterable(connector_color)
        if len(connector_color) != len(ordered_positions):
            connector_color = connector_color * len(ordered_positions)

        old_pos = 0
        for idx, (ax_chr_x, ax_chr_y, (chromosome, snp_positions)) in enumerate(
            zip(ax_chromosome_x, ax_chromosome_y, ordered_positions.items())
        ):
            chromosome_lims = (
                np.array(
                    [
                        np.floor(min(snp_positions) / chromosome_position_rounding_factor),
                        np.ceil(max(snp_positions) / chromosome_position_rounding_factor),
                    ]
                )
                * chromosome_position_rounding_factor
            )
            # X-axis
            ax_chr_x.vlines(
                x=snp_positions, ymin=0, ymax=1, color=snp_color[idx], linewidth=snp_linewidth
            )
            ax_chr_x.set_xticks(
                ax_chr_x.get_xticks(), labels=ax_chr_x.get_xticklabels(), fontdict=tick_fontdict
            )
            ax_chr_x.set_yticks(ax_chr_x.get_yticks(), labels=[])
            for spine in ["top", "left", "bottom", "right"]:
                ax_chr_x.spines[spine].set_linewidth(2)

            loc = bool(chromosome_locx[0] == "t")
            ax_chr_x.tick_params(
                axis="x",
                labeltop=loc,
                top=loc,
                labelbottom=not loc,
                bottom=not loc,
                rotation=-60 if loc else 60,
            )
            ax_chr_x.tick_params(
                axis="y", labelleft=False, left=False, labelright=False, right=False
            )
            for ticklabel in ax_chr_x.get_xticklabels():
                ticklabel.set_rotation_mode("anchor")
                ticklabel.set_horizontalalignment("right")
            ax_chr_x.set_facecolor("xkcd:light grey")
            ax_chr_x.set_xlabel(
                f"Chromosome {chromosome} Position", loc="center", fontdict=label_fontdict
            )
            ax_chr_x.xaxis.set_label_position("top" if loc else "bottom")
            ax_chr_x.xaxis.set_major_formatter(
                mpl.ticker.EngFormatter(unit="bp", places=3, sep=" ", useMathText=True)
            )
            ax_chr_x.xaxis.set_major_locator(
                mpl.ticker.FixedLocator(
                    np.linspace(*chromosome_lims, max(6 - len(ordered_positions), 2))
                )
            )
            ax_chr_x.set_xlim(chromosome_lims)
            ax_chr_x.set_ylim(0, 1)
            final_pos = old_pos + len(snp_positions)
            heatmap_ticks = ax_LDheatmap.get_yticks()[old_pos:final_pos]
            for xyA, xyB in zip(snp_positions, heatmap_ticks):
                xyA = (xyA, ax_chr_x.get_ylim()[0 if loc else -1])
                xyB = (xyB, ax_LDheatmap.get_ylim()[-1 if loc else 0])
                ax_LDheatmap.add_artist(
                    mpl.patches.ConnectionPatch(
                        xyA=xyA,
                        xyB=xyB,
                        coordsA="data",
                        coordsB="data",
                        axesA=ax_chr_x,
                        axesB=ax_LDheatmap,
                        color=connector_color[idx] if connector_color else snp_color[idx],
                        linestyle="-",
                        linewidth=connector_linewidth,
                    )
                )

            # Y-axis
            ax_chr_y.hlines(
                y=snp_positions, xmin=0, xmax=1, color=snp_color[idx], linewidth=snp_linewidth
            )
            ax_chr_y.set_yticks(
                ax_chr_y.get_yticks(), labels=ax_chr_y.get_yticklabels(), fontdict=tick_fontdict
            )
            ax_chr_y.set_xticks(ax_chr_y.get_xticks(), labels=[])
            for spine in ["top", "left", "bottom", "right"]:
                ax_chr_y.spines[spine].set_linewidth(2)

            loc = bool(chromosome_locy[0] == "l")
            ax_chr_y.tick_params(
                axis="y",
                labelleft=loc,
                left=loc,
                labelright=not loc,
                right=not loc,
                rotation=0,
            )
            ax_chr_y.tick_params(
                axis="x", labeltop=False, top=False, labelbottom=False, bottom=False
            )
            ax_chr_y.set_facecolor("xkcd:light grey")
            ax_chr_y.set_ylabel(
                f"Chromosome {chromosome} Position", loc="center", fontdict=label_fontdict
            )
            ax_chr_y.yaxis.set_label_position("left" if loc else "right")
            ax_chr_y.yaxis.set_major_formatter(
                mpl.ticker.EngFormatter(unit="bp", places=3, sep=" ", useMathText=True)
            )
            ax_chr_y.yaxis.set_major_locator(
                mpl.ticker.FixedLocator(
                    np.linspace(*chromosome_lims, max(6 - len(ordered_positions), 2))
                )
            )
            ax_chr_y.set_ylim(chromosome_lims)
            ax_chr_y.set_xlim(0, 1)

            final_pos = old_pos + len(snp_positions)
            heatmap_ticks = ax_LDheatmap.get_yticks()[old_pos:final_pos]
            for xyA, xyB in zip(snp_positions, heatmap_ticks):
                xyA = (ax_chr_y.get_xlim()[-1 if loc else 0], xyA)
                xyB = (ax_LDheatmap.get_xlim()[0 if loc else -1], xyB)
                ax_LDheatmap.add_artist(
                    mpl.patches.ConnectionPatch(
                        xyA=xyA,
                        xyB=xyB,
                        coordsA="data",
                        coordsB="data",
                        axesA=ax_chr_y,
                        axesB=ax_LDheatmap,
                        color=connector_color[idx] if connector_color else snp_color[idx],
                        linestyle="-",
                        linewidth=connector_linewidth,
                    )
                )
            old_pos += len(snp_positions)

        # X-axis Colorbar
        fig.colorbar(triu_im, ax_cbar_x, orientation="horizontal")
        loc = bool(chromosome_locx[0] == "t")
        ax_cbar_x.tick_params(
            axis="x",
            labeltop=not loc,
            top=not loc,
            labelbottom=loc,
            bottom=loc,
        )
        ax_cbar_x.set_title(
            value_type[0].replace("r2", "$r^{2}$"), y=-1.5 if loc else 1.5, fontdict=label_fontdict
        )

        fig.colorbar(tril_im, ax_cbar_y, orientation="vertical")
        loc = bool(chromosome_locy[0] == "l")
        ax_cbar_y.tick_params(
            axis="y",
            labelleft=not loc,
            left=not loc,
            labelright=loc,
            right=loc,
            rotation=0,
        )
        ax_cbar_y.set_title(value_type[1].replace("r2", "$r^{2}$"), fontdict=label_fontdict)

    return fig, axes
