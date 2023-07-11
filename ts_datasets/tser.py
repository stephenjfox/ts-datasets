from pathlib import Path

from pandas import Series, concat, DataFrame
import torch as pt

from .tsextrinsic_regression_data_loader import load_from_tsfile_to_dataframe, regression_datasets as REGRESSION_DS_NAMES


def dataset_rows_to_train_val_tensors(df: DataFrame,
                                      targets_array: Series,
                                      *,
                                      validation_portion=0.2):
    """Convert the dataframe and targets from the TSER load function into PyTorch tensors.
    
    The TSER Monash-UEA-UCR archive gives us we have N-many sequences of M in length e.g. (N, M, *).
    The number of features (``n_features``) vary in the supplied ``df``.
    NOTE: this would be a great time to experiment with the embedding of a sequence.
    
    The problem is to predict a single value from the entire sequence, rather than having single for each time step.
    The *only* signal is from the final time step, so one might be able to derive values for each time step - but that is
    not the explicit goal.
    We beliieve this better represents the real world, where we don't have labels at every time step for e.g. time-to-event prediction
    but will experience some event and work backwards to find some descriptive labels with e.g. subject-matter experts.
    
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
    out = ((tens[:-val_len], tens_targets[:-val_len]), (tens[-val_len:], tens_targets[-val_len:]))
    return out


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

    DEBUG = True

    trainables_output_dir = dataset_archive_root / "trainable-tensors"

    # FIXME(stephen): it should probably be segregated by file format.
    mvts_data_root = unzip_root / "Monash_UEA_UCR_Regression_Archive"

    mvts_data_root.mkdir(parents=True, exist_ok=True)
    trainables_output_dir.mkdir(parents=True, exist_ok=True)

    if DEBUG:
        print("Files and Folders in the data root directory:\n", list(mvts_data_root.iterdir()))

    dataset_dir = mvts_data_root / dataset_dir_name

    # mvts: Multivariate Time Series sequence
    mvts_train_df, mvts_train_targets = load_from_tsfile_to_dataframe(
        dataset_dir / f"{dataset_dir_name}_TRAIN.ts")
    mvts_test_df, mvts_test_targets = load_from_tsfile_to_dataframe(dataset_dir /
                                                                    f"{dataset_dir_name}_TEST.ts")

    processed_train_val = dataset_rows_to_train_val_tensors(mvts_train_df, mvts_train_targets)
    if DEBUG:
        print("df_{train,test}.shape:", mvts_train_df.shape, mvts_test_df.shape)
        if mvts_train_df.shape[-1] > 1:
            print("df.iloc[0, [0, 1]]:", mvts_train_df.iloc[0, 0], mvts_train_df.iloc[0, 1])
        else:
            print("df.iloc[0, 0]:", mvts_train_df.iloc[0, 0])
        print("df.iloc[0, -1]:", mvts_train_df.iloc[0, -1])
        print("Output tensor sizes (train, val):\n", [
            (feat_tens.size(), target_tens.size()) for feat_tens, target_tens in processed_train_val
        ])

    # train
    pt.save(processed_train_val, f"{trainables_output_dir}/{dataset_dir_name}_TRAIN.pt")

    # test
    stacked_tensors_and_targets = (pt.stack([
        pt.tensor(concat(row_tup, axis='columns').values)
        for _, *row_tup in mvts_test_df.itertuples()
    ]), pt.tensor(mvts_test_targets))

    pt.save(stacked_tensors_and_targets, f"{trainables_output_dir}/{dataset_dir_name}_TEST.pt")
