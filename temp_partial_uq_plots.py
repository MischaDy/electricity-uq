import numpy as np
from matplotlib import pyplot as plt

from helpers.io_helper import IO_Helper
from helpers.misc_helpers import get_data


data = get_data(
    filepath=settings.DATA_FILEPATH,
    train_years=settings.TRAIN_YEARS,
    val_years=settings.VAL_YEARS,
    test_years=settings.TEST_YEARS,
    do_standardize_data=True,
)
X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = data

io_helper = IO_Helper('comparison_storage')

gp_pred = io_helper.load_array(filename='native_gpytorch_y_pred_n210432_it200.npy')
gp_quantiles = io_helper.load_array(filename='native_gpytorch_y_quantiles_n210432_it200.npy')
gp_std = io_helper.load_array(filename='native_gpytorch_y_std_n210432_it200.npy')

mvnn_pred = io_helper.load_array(filename='native_mvnn_y_pred_n210432_it100_nh2_hs50.npy')
mvnn_quantiles = io_helper.load_array(filename='native_mvnn_y_quantiles_n210432_it100_nh2_hs50.npy')
mvnn_std = io_helper.load_array(filename='native_mvnn_y_std_n210432_it100_nh2_hs50.npy')

qr_pred = io_helper.load_array(filename='native_quantile_regression_nn_y_pred_n210432_it100_nh2_hs20.npy')
qr_quantiles = io_helper.load_array(filename='native_quantile_regression_nn_y_quantiles_n210432_it100_nh2_hs20.npy')
qr_std = io_helper.load_array(filename='native_quantile_regression_nn_y_std_n210432_it100_nh2_hs20.npy')

cp_arr_names = [['posthoc_conformal_prediction_base_model_hgbr_y_pred_n210432_it5.npy',
                 'posthoc_conformal_prediction_base_model_hgbr_y_quantiles_n210432_it5.npy',
                 'posthoc_conformal_prediction_base_model_hgbr_y_std_n210432_it5.npy'],
                ['posthoc_conformal_prediction_base_model_linreg_y_pred_n210432_it5.npy',
                 'posthoc_conformal_prediction_base_model_linreg_y_quantiles_n210432_it5.npy',
                 'posthoc_conformal_prediction_base_model_linreg_y_std_n210432_it5.npy'],
                ['posthoc_conformal_prediction_base_model_nn_y_pred_n210432_it5.npy',
                 'posthoc_conformal_prediction_base_model_nn_y_quantiles_n210432_it5.npy',
                 'posthoc_conformal_prediction_base_model_nn_y_std_n210432_it5.npy']]
cp_arrs = [[io_helper.load_array(filename=filename) for filename in arr_name_set] for arr_name_set in cp_arr_names]

la_arr_names = [
'posthoc_laplace_approximation_base_model_nn_y_pred_n210432_it100.npy',
    'posthoc_laplace_approximation_base_model_nn_y_quantiles_n210432_it100.npy',
    'posthoc_laplace_approximation_base_model_nn_y_std_n210432_it100.npy',
]
la_arrs = [io_helper.load_array(filename=filename) for filename in la_arr_names]

drawing_quantiles = y_quantiles is not None
if drawing_quantiles:
    ci_low, ci_high = (
        y_quantiles[:, 0],
        y_quantiles[:, -1],
    )
    drawn_quantile = round(max(quantiles) - min(quantiles), 2)
else:
    ci_low, ci_high = y_pred - n_stds * y_std, y_pred + n_stds * y_std

n_samples_to_plot = 1600  # about 2 weeks
n_samples_train, n_samples_test = X_train.shape[0], X_test.shape[0]
if n_samples_train < n_samples_to_plot or n_samples_test < n_samples_to_plot:
    logging.info(f'not enough train ({n_samples_train}) and/or test ({n_samples_test}) samples for'
                 f' partial plots (must be >= {n_samples_to_plot} each) - skipping.')
    partial_plots = False

    logging.info('plotting partial plots...')
    y_pred_train = y_pred[:n_samples_to_plot]
    start_test = y_train.shape[0] + y_val.shape[0]
    y_pred_test = y_pred[start_test: start_test + n_samples_to_plot]


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))

start_train, end_train = 0, X_train.shape[0]
start_val, end_val = end_train, end_train + X_val.shape[0]
start_test, end_test = end_val, end_val + X_test.shape[0]

x_plot_train = np.arange(start_train, end_train)
x_plot_val = np.arange(start_val, end_val)
x_plot_test = np.arange(start_test, end_test)

ax.plot(x_plot_train, y_train, label='train data', color="black", linestyle='dashed')
ax.plot(x_plot_val, y_val, label='val data', color="purple", linestyle='dashed')
ax.plot(x_plot_test, y_test, label='test data', color="blue", linestyle='dashed')

x_plot_full = self._get_x_plot_full(X_train, X_val, X_test)
ax.plot(x_plot_full, y_pred, label="point prediction", color="green")
# noinspection PyUnboundLocalVariable
label = rf'{100 * drawn_quantile}% CI' if drawing_quantiles else f'{n_stds} std'
ax.fill_between(
    x_plot_full.ravel(),
    ci_low,
    ci_high,
    color="green",
    alpha=0.2,
    label=label,
)
ax.legend()
ax.set_xlabel("data")
ax.set_ylabel("target")
ax.set_title(method_name)
if save_plot:
    self.io_helper.save_plot(method_name=method_name)
if show_plots:
    plt.show(block=True)
plt.close(fig)
