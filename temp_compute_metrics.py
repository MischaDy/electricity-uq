import settings
import settings_update
from helpers import misc_helpers
from helpers.uq_arr_helpers import get_uq_method_to_arrs_dict
from uq_comparison_pipeline import UQ_Comparison_Pipeline


RUN_SIZE = 'full'


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = load_data()

    uq_method_to_arrs_dict = get_uq_method_to_arrs_dict()
    arrs = list(uq_method_to_arrs_dict.values())[0]
    y_pred, y_quantiles, y_std = arrs
    metrics_det = UQ_Comparison_Pipeline.compute_metrics_det(y_pred, y)
    metrics_uq = UQ_Comparison_Pipeline.compute_metrics_uq(y_pred, y_quantiles, y_std, y, settings.QUANTILES)


def load_data():
    settings.RUN_SIZE = RUN_SIZE
    settings_update.update_run_size_setup()
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y = misc_helpers.get_data(
        filepath=settings.DATA_FILEPATH,
        train_years=settings.TRAIN_YEARS,
        val_years=settings.VAL_YEARS,
        test_years=settings.TEST_YEARS,
        n_points_per_group=settings.N_POINTS_PER_GROUP,
        do_standardize_data=True,
    )
    y_train, y_val, y_test, y = map(scaler_y.inverse_transform, [y_train, y_val, y_test, y])
    return X_train, y_train, X_val, y_val, X_test, y_test, X, y, scaler_y


if __name__ == '__main__':
    main()
