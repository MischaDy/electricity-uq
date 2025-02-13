# Uncertainty Estimates for Short-Term Grid Load Forecasting: Post-hoc vs. Natively Probabilistic Approaches

This is the repo of my master's thesis at the University of Tübingen.

The trained models as well as the prediction arrays are stored seperately due to their size:
https://drive.google.com/drive/folders/1hHt2N6MbeeUNSSszukw9e3XsDiC9IVA7?usp=sharing


## Overview of the repo structure

[//]: # (todo: goal-level view rather than repo-level view? )

- `comparison_storage`: container for outputs, such as plots and metrics
- `data`: contains the notebook for preprocessing the data, as well as the preprocessed data. The original data has not 
          been uploaded to the repo, but is needed to get the results. It is expected to be in a directory called
          `data_Energy_Germany`.
- top level notebooks named `plot_*.ipynb`: produce almost all plots shown in the thesis
  - TODO
- `make_metrics_hist_plots.py`: TODO - needed?
- `make_partial_uq_plots.py`: generates the one-week example plots showing the true data as well as the model's
                              prediction and uncertainty
- `settings.py`: contains the configurations for all 
