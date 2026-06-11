# HK1 R12Ter analysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Analysis of omics data associated with the HK1 R12Ter single nucleotide variant.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         hk1_r12ter_analysis and configuration for tools like black
│
├── reports            <- Generated graphics, figures, downloaded data and analysis from external tools, etc. to be used for Figures
│   ├── Figure_1       <- For Figure 1
│   ├── Figure_2       <- For Figure 1
│   ├── Figure_3       <- For Figure 1
│   ├── Figure_4       <- For Figure 1
│   ├── Figure_5       <- For Figure 1
│   ├── Figure_6       <- For Figure 1
│   ├── Figure_S1      <- For Figure 1
│   └── Final          <- Final versions of the figures as .PNG files
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8, pre-commit, and other project tools
│
└── hk1_r12ter_analysis   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes hk1_r12ter_analysis a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── linear_model.py     <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
## Notebook Organization

Notebooks are organized as the following:

* Notebooks that start with "1" are responsible for data processing and computations for analysis. 
    * Notebooks that start with "1.0" are related to analysis of the REDS RBC-Omics data
    * Notebooks that start with "1.1" are related to analysis of the HK1-Arg12Ter data.
    * Notebooks that start with "1.2" are related to processing data from the **Supplemental_Data_S1.xlsx** file.
* Notebooks that start with "2" are responsible for data visualization and generation of figure panels. 
    * Notebooks that start with "2.0" are related to analysis of the REDS RBC-Omics data
    * Notebooks that start with "2.1" are related to analysis of the HK1-Arg12Ter data.
* Notebooks that have suffixes "a" and "b" after the number are near identical but act on different datasets (i.e., the REDS Index and REDS Recall cohorts, respectively.)

### Warning
The following notebooks will not work without the initial raw data. 
* 1.01-ZBH-REDS-Data-CleanFormat.ipynb
* 1.02-ZBH-REDS-Data-Filter.ipynb
* 1.03-ZBH-REDS-Data-Normalize.ipynb
* 1.04-ZBH-REDS-Data-Combine.ipynb

In order to preserve the privacy and integrity surrounding and follow regulations surroudning the use of REDS RBC-Omics data, the publically available data found in the tables in the **Supplemental_Data_S1.xlsx** file may be used instead.

To get the data formattd for analysis...
1. Place the **Supplemental_Data_S1.xlsx** file into the */data/raw/* directory.
2. Run the provided notebook: "1.21-ZBH-Supplemental-Data-To-Processed. This will skip the need to run the previously listed notebooks.
3. Continue to run the notebooks in order, skipping the previously listed notebooks
