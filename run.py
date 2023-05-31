import json
from pathlib import Path

import pandas as pd

from monash_tsforecasting_data_loader import convert_tsf_to_dataframe
from tsextrinsic_regression_data_loader import load_from_tsfile_to_dataframe


def handle_australian_electricity(data: pd.DataFrame):
    truncation_length = min([len(array) for array in data['series_value']])
    print(f"{truncation_length = }")
    df = pd.DataFrame({row['state']: row['series_value'][:truncation_length] for _, row in data.iterrows()})
    correlation = df.corr()
    print(correlation)

def main():
    monash_root = "/Users/stephenfox/dev/Datasets/Monash-Time-Series-Datasets-in-TSF-format"
    tse_root = "/Users/stephenfox/dev/Datasets/TST-Comparison/Monash_UEA_UCR_Regression_Archive"

    australian_electricity_path = Path(monash_root, "australian_electricity_demand_dataset.tsf")
    (loaded_data, frequency, forecast_horizon, contain_missing_values,
     contain_equal_length) = convert_tsf_to_dataframe(australian_electricity_path)

    print(f"{frequency = }")
    print(f"{forecast_horizon = }")
    print(f"{contain_missing_values = }")
    print(f"{contain_equal_length = }")
    print(f"{loaded_data}")

    # handle_australian_electricity(loaded_data)





if __name__ == '__main__':
    main()