from pathlib import Path

import pandas as pd

from ts_datasets.monash_tsforecasting_data_loader import convert_tsf_to_dataframe
from ts_datasets.tser import process_regression_dataset


def handle_australian_electricity(data: pd.DataFrame):
    # Series value is a PandasArray, instance, because it's jagged. Why they didn't just pack a Series, idk
    truncation_length = min([len(array) for array in data['series_value']])
    print(f"{truncation_length = }")
    df = pd.DataFrame(
        {row['state']: row['series_value'][:truncation_length] for _, row in data.iterrows()})
    correlation = df.corr()
    print(correlation)


def work_monash(
        monash_root="/Users/stephenfox/dev/Datasets/Monash-Time-Series-Datasets-in-TSF-format"):
    australian_electricity_path = Path(monash_root, "australian_electricity_demand_dataset.tsf")
    (loaded_data, frequency, forecast_horizon, contain_missing_values,
     contain_equal_length) = convert_tsf_to_dataframe(australian_electricity_path)

    print(f"{frequency = }")
    print(f"{forecast_horizon = }")
    print(f"{contain_missing_values = }")
    print(f"{contain_equal_length = }")
    print(f"{loaded_data}")

    # handle_australian_electricity(loaded_data)


def work_tse(tse_root="/Users/stephenfox/dev/Datasets/TST-Comparison/"):
    tse_root = Path(tse_root)
    process_regression_dataset(tse_root, unzip_root=tse_root, dataset_dir_name="IEEEPPG")


def main():
    work_tse()
    # work_monash()


if __name__ == '__main__':
    main()