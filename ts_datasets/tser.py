"""
The TSER Monash-UEA-UCR archive gives us we have N-many sequences of M in length e.g. (N, M, n_features).
The number of columns or features (``n_features``) vary in the supplied ``df``, based on the dataset.
- With one exceptiont, all datasets have the same number of time steps per feature.

NOTE: If you're doing deep learning, this would be a great dataset for experimentation with the embedding of a sequence.

The Time Series Extrinsic Regression problem is to predict a single value from the entire sequence,
rather than having single for each time step (such as shown in e.g. Google's Machine Learning Crash Course).

The *only* label (and therefore, signal) is for the whole sequence
- It may or may not be from the final time step of the sequence e.g. total kwH consumed by a household's appliance usage
- It may be a separate value e.g. approximate BPM given 12s of 3 different vital signs

so one might be able to derive values for each time step - but that is not the explicit goal.

We believe this better represents the real world, where we don't have labels at every time step for e.g. time-to-event prediction
but will experience some event and work backwards to find some descriptive labels with e.g. subject-matter experts.

This contrasts with e.g. industry giant Amazon's perspective in their Gluon Time Series library, where they are
forecasting outputs per time step within the same domain rather than mapping to a new domain for their predictions.
"""

from pathlib import Path

from pandas import Series, concat, DataFrame
import torch as pt

from .tsextrinsic_regression_data_loader import load_from_tsfile_to_dataframe, regression_datasets as REGRESSION_DS_NAMES

def dataset_rows_to_train_val_tensors(df: DataFrame,
                                      targets_array: Series,
                                      *,
                                      validation_portion=0.2):
    """Convert the dataframe and targets from the TSER load function into PyTorch tensors.
    
    Args:
        df:
            the sequence-dataframe, where each cell is a series of length ``M`` (sans targets)
            and each row is ``n_features`` columns wide.
        targets_array:
            the value to predict from the multivariate series input of shape ``[time: M, feaure: n_features]``.
            One element in this array corresponds to the value to regress from an entire sequence of shape ``[M, n_features]``.
        validation_portion:
            the portion of the data (time series and targets) to consider for the validation set, in interval [0, 1]
    """
    dfs = [concat(row_tup, axis='columns') for _, *row_tup in df.itertuples("SeriesColumns")]

    val_len = round(len(dfs) * validation_portion) or 1
    tens = [pt.tensor(d.values) for d in dfs]
    tens = pt.stack(tens)
    tens_targets = pt.tensor(targets_array)
    train_val_tensors = ((tens[:-val_len], tens_targets[:-val_len]), (tens[-val_len:], tens_targets[-val_len:]))
    return train_val_tensors


def process_regression_dataset(dataset_archive_root: Path,
                               unzip_root: Path,
                               dataset_dir_name: str = "AppliancesEnergy"):
    """Use write train-val and test sets to ``dataset_archive_root / 'trainable-tensors'``

    The function uses ``load_from_tsfile_to_dataframe(...)`` and ``dataset_rows_to_train_val_tensors(...)`` in
    proper sequence. Feel free to adapt.

    Example usage:

        >>> dataset_archive_root = Path("/c/datasets/timeseries-forecasting")
        >>> unzip_root           = Path("/c/datasets/timeseries-forecasting/unzipped")
        >>> process_regression_dataset(dataset_archive_root, unzip_root) # writes to disk.

    Args:
        dataset_archive_root:
            Root directory for all time series dataset work.
        unzip_root:
            Directory where all the time series data is unzipped *to*.
        dataset_dir_name:
            Name of the particular dataset that we want to convert into Tensors.
            Defaults to "AppliancesEnergy".
    """
    assert dataset_dir_name in REGRESSION_DS_NAMES, "Dataset name must be a directory in the extracted archive"
    if dataset_dir_name == "PPGDalia":
        # NOTE: just downsample or 2-step rolling average the PPG, as it's 64Hz sampled and the other signals are 32Hz
        raise ValueError(
            "'PPGDalia' dataset is current unsupported; its data alignment is non-standard")

    trainables_output_dir = dataset_archive_root / "trainable-tensors"

    mvts_data_root = unzip_root / "Monash_UEA_UCR_Regression_Archive"

    mvts_data_root.mkdir(parents=True, exist_ok=True)
    trainables_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = mvts_data_root / dataset_dir_name

    # mvts: Multivariate Time Series sequence
    mvts_train_df, mvts_train_targets = load_from_tsfile_to_dataframe(dataset_dir /
                                                                      f"{dataset_dir_name}_TRAIN.ts")
    mvts_test_df, mvts_test_targets = load_from_tsfile_to_dataframe(dataset_dir /
                                                                    f"{dataset_dir_name}_TEST.ts")

    processed_train_val = dataset_rows_to_train_val_tensors(mvts_train_df, mvts_train_targets)

    # train
    pt.save(processed_train_val, f"{trainables_output_dir}/{dataset_dir_name}_TRAIN.pt")

    # test
    stacked_tensors = pt.stack([
        pt.tensor(concat(row_tup, axis='columns').values)
        for _, *row_tup in mvts_test_df.itertuples()
    ])
    targets = pt.tensor(mvts_test_targets)

    pt.save((stacked_tensors, targets), f"{trainables_output_dir}/{dataset_dir_name}_TEST.pt")
