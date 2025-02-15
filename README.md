# Uncertainty Estimates for Short-Term Grid Load Forecasting: Post-hoc vs. Natively Probabilistic Approaches

This is the repo of my master's thesis at the University of Tübingen.

The commit tagged `submission_commit` represents the last commit made before the submission deadline.


## Trained Models and Prediction Arrays

The trained models as well as the prediction arrays are stored seperately due to their size:
https://drive.google.com/drive/folders/1hHt2N6MbeeUNSSszukw9e3XsDiC9IVA7?usp=sharing

The directory contains two folders: `arrays` and `models`. For both file types, the most important infix to watch out
for is `_n210432_`. The number after the `n` specifies the number of training data points, and that infix correponds the
entire training data set. (I have used smaller sets for testing.)
The directory structure is as follows:
- `arrays`: contains prediction and error arrays.
  - prediction arrays - the models' point and uncertainty estimates over the data. filename contains `y_pred`
    (point predictions), `y_std`, or `y_quantiles` (STD and quantile predictions, respectively)
  - error arrays - for the given model, they store the specified error metric for each data point. filename contains
      `ae`, `crps`, or `ssr`.
- `models`: contains the trained models. As far as I can tell, for all models, the sorting the models alphabetically and
  picking the last model corresponds to the most current one, i.e. the one used to generate the thesis' results.


## Overview of the Repo Structure

### In brief

- model source code: `src_*` directories
- pipeline: [uq_comparison_pipeline_abc.py](uq_comparison_pipeline_abc.py) (base class) and [uq_comparison_pipeline.py](uq_comparison_pipeline.py) (instantiation)
- metrics, plots, run configs: [comparison_storage](comparison_storage)
- data preprocessing + preprocessed data: [data](data)
- 1-week-long example UQ plots: [make_partial_uq_plots.py](make_partial_uq_plots.py)
- all other plots: `plot_*.ipynb` notebooks


### Everything

- [comparison_storage](comparison_storage) container for outputs, such as plots and metrics
  - [metrics](comparison_storage/metrics) - the models' metrics. contains a single subdirectory for "historical" reasons.
  - [plots](comparison_storage/plots) - the plots used in the thesis, plus some old ones
  - [run_settings](comparison_storage/run_settings) - the hyperparameter settings of each model, as best as I could determine them + some additional notes
- [data](data): contains the notebook for preprocessing the data, as well as the preprocessed data.
- [data_Energy_Germany](data_Energy_Germany) - The original data, which has been uploaded to the repo for
  more convenient verifying. The source of the data is the
  [Bundesnetzagentur | SMARD.de](https://www.smard.de/home/downloadcenter/download-marktdaten/).
- [env_files](env_files) - somewhat chaotic list of the packages used.
- [helpers](helpers) - various helpers
- `src_*` directories - contain the base, native, and post-hoc models' source code. HGBR and QHGBR are the base and
  native tree models, respectively
- [temp](temp) - temp files, removed from top-level view. probably need to be moved to top level to execute without errors
- [deploy.sh](deploy.sh) - the deployment script
- [make_partial_uq_plots.py](make_partial_uq_plots.py) - produces the 1-week-long example UQ plots
- [metrics_comparison_test.csv](metrics_comparison_test.csv), [metrics_comparison_test_skill.csv](metrics_comparison_test_skill.csv),
  and [resource_usage.csv](resource_usage.csv) - tables featured in the thesis. they are used by some of the plots scripts 
- top level notebooks named `plot_*.ipynb` - produce the remaining plots featured in the thesis. note that you need to change some notebooks' flags (like `PLOT_EXAMPLE=False`) to produce all plots used in my thesis. 
  (the notebooks shouldn't really be on the top level, but moving them down would break the paths and require code changes, so the clean-up will have to wait until after the review.)
  - [plot_calibration.ipynb](plot_calibration.ipynb) - plots the calibration curve
  - [plot_coverage.ipynb](plot_coverage.ipynb) - plots the coverage curve
  - [plot_data_distr.ipynb](plot_data_distr.ipynb) - plots the data distribution, both the train/test split and the yearly distribution
  - [plot_data_example.ipynb](plot_data_example.ipynb) - plots the introductory plot showing the lagged features used
  - [plot_error_metrics_distr.ipynb](plot_error_metrics_distr.ipynb) - plots the error distributions (CRPS, AE, SSR), both the full plots and the excerpts
  - [plot_error_metrics_lines.ipynb](plot_error_metrics_lines.ipynb) - plots the metrics lineplot showing the general performance
  - [plot_qq.ipynb](plot_qq.ipynb) - plots introductory QQ plot
  - [plot_resource_usage.ipynb](plot_resource_usage.ipynb) - plots color-coded resource usage 
  - [plot_resource_usage_vs_metrics.ipynb](plot_resource_usage_vs_metrics.ipynb) - plots resource usage vs. metrics plots
  - [plot_skill.ipynb](plot_skill.ipynb) - plots skill scores
  - [plot_stds_distr.ipynb](plot_stds_distr.ipynb) - plots the predicted STD distributions
- [settings.py](settings.py)`settings.py` - the central configurations file for all models
- [settings_update.py](settings_update.py) - settings-related convenience functions
- [store_error_arrs.py](store_error_arrs.py) - helper to compute and store the pointwise error arrays, which lie in the Drive
- [temp_compute_metrics.py](temp_compute_metrics.py) (BAD NAMING) - computes the metrics in [comparison_storage/metrics](comparison_storage/metrics)
- [uq_comparison_pipeline_abc.py](uq_comparison_pipeline_abc.py) - pipeline abstract base class. takes care of the general workflow
- [uq_comparison_pipeline.py](uq_comparison_pipeline.py) - instantiated pipeline. implements specifics: data loading, model training, metrics computations.
