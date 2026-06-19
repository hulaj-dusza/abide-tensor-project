# fmri-project

# Tensor Regression for Neuroimaging

[Polska wersja / Polish version](README_PL.md)

## Overview

This repository contains research experiments on tensor regression and tensor-based classification for neuroimaging data.

The project investigates whether low-rank tensor models can identify spatial patterns in brain imaging data while using substantially fewer parameters than conventional voxel-wise models.

The current work focuses on classification of participants with autism spectrum disorder (ASD) and control participants using neuroimaging derivatives from the ABIDE dataset.

## Research objective

The main objective is to evaluate whether high-dimensional neuroimaging data can be represented and analysed efficiently using tensor-based machine learning models.

The project addresses the following research questions:

* Can low-rank tensor regression distinguish ASD participants from control participants?
* How do CP tensor models compare with conventional voxel-wise logistic regression?
* What is the influence of tensor rank, regularization strength, and preprocessing choices on classification performance?

## Data

The project uses data derived from the ABIDE Preprocessed dataset.

The analysed data include different neuroimaging representations, such as:

* four-dimensional functional MRI data,
* three-dimensional Regional Homogeneity (ReHo) maps,

The classification task is:

```text
ASD = 1
CONTROL = 0
```

Raw neuroimaging data, NIfTI files, and subject-level NumPy arrays are not included in this repository. They are stored locally and excluded through `.gitignore`.

## Methods

The main methodological focus is tensor regression, particularly CP-based logistic regression.

For a participant-specific tensor:

[
\mathcal{X}_i \in \mathbb{R}^{p_1 \times p_2 \times p_3},
]

the model predicts the probability of ASD using:

[
P(y_i = 1 \mid \mathcal{X}_i)
=============================

\sigma\left(
\beta_0 +
\langle \mathcal{X}_i, \mathcal{B} \rangle
\right),
]

where (\mathcal{B}) is a coefficient tensor and (\sigma) is the logistic sigmoid function.

To reduce the number of parameters, the coefficient tensor is approximated with a CP decomposition:

[
\mathcal{B}
\approx
\sum_{r=1}^{R}
\lambda_r
\mathbf{a}_r \circ
\mathbf{b}_r \circ
\mathbf{c}_r.
]

The project includes experiments with:

* CP-logistic regression,
* alternating block-coordinate optimization,
* CP-ALS-inspired optimization procedures,
* gradient-based tensor optimization,
* voxel-wise logistic regression baselines,
* stratified cross-validation,
* comparison of tensor ranks and regularization parameters.

## Main branches

The repository is organized into branches representing different stages of development and experiments.

| Branch                          | Purpose                                              |
| ------------------------------- | ---------------------------------------------------- |
| `main`                          | General project overview and stable documentation    |
| `feature/build-dataset`         | Dataset construction and preparation                 |
| `feature/cp-4d-logistic`        | CP-logistic regression for 4D fMRI data              |
| `feature/cp-als-logistic`       | CP-ALS-inspired logistic regression experiments      |
| `feature/cv-comparison`         | Cross-validation and model comparison                |
| `feature/benchmark-flat-logreg` | Conventional voxel-wise logistic regression baseline |
| `feature/cp-3d-als-logistic`    | 3D ReHo tensor classification experiments  |

The available branches may evolve as the project develops.

## Typical workflow

1. Obtain and prepare neuroimaging data locally.
2. Construct participant-level tensors and an index CSV file.
3. Configure local paths and model hyperparameters in the selected script.
4. Run the experiment.
5. Save fold-level and summary results to CSV files.
6. Compare results across tensor ranks, preprocessing variants, and baseline models.

## Requirements

The project uses Python and scientific computing libraries, including:

```text
numpy
pandas
scikit-learn
```

Some experiments may require additional libraries depending on the implementation.

Install the core dependencies with:

```bash
pip install numpy pandas scikit-learn
```

## Reproducibility notes

The scripts contain local file paths that must be updated before execution on another computer.

For reproducibility, experiments should document:

* the data derivative and preprocessing strategy,
* tensor shape,
* participant selection criteria,
* model type,
* CP rank,
* regularization parameter,
* optimization settings,
* cross-validation scheme,
* random seed,
* evaluation metrics.

## Evaluation metrics

The main metrics used in the project include:

* ROC-AUC,
* accuracy,
* sensitivity for ASD,
* specificity for control participants,
* balanced accuracy,
* runtime.

The results should be interpreted as exploratory research outcomes rather than clinical diagnostic performance.

## Current limitations

* Hyperparameter selection is not yet fully nested within cross-validation.
* Site-related variation is not explicitly harmonized in all experiments.
* Demographic and clinical covariates are not currently included in every model.
* The dataset is heterogeneous across acquisition sites and preprocessing choices.
* Results should be compared across multiple configurations before drawing conclusions.

## Future work

* Systematic comparison of CP ranks and regularization strengths.
* Comparison of ReHo, fALFF, and 4D fMRI representations.
* Extension to Tucker-based tensor regression.
* Reconstruction and visualization of learned coefficient tensors.
* Inclusion of demographic, clinical, and motion-related covariates.
* Site harmonization and more rigorous nested cross-validation.
* Comparison with additional machine learning baselines.

## Author

Project developed as part of a Master's thesis in data analysis, with a focus on tensor methods and neuroimaging data analysis.
