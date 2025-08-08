import pandas as pd
from pathlib import Path
import warnings

# useful paths
IOTs_folder = Path("/home/rita/Documents/Tesi/Projects/reformat_EXIOBASE/example/output/reformat")


energy_sectors=["ENERGY"]

import warnings
import pandas as pd

def test_energy_C_Ap_zero(IOT: pd.DataFrame, energy_sectors: list[str], filename: str):
    cap_cols = IOT.columns.get_level_values('Subcategory') == 'C_Ap'

    for sector in energy_sectors:
        row = ('CI_imp', sector)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            values = IOT.loc[row, cap_cols]

        if not (values == 0).all().all():
            nonzero = values[values != 0]
            raise AssertionError(
                f"File '{filename}': Non-zero values found for row {row} in C_Ap columns:\n{nonzero}"
            )
    print(f"Test passed for file '{filename}' and sectors {energy_sectors}. ")

def test_tax_AP_Imp_Dom_zero(IOT: pd.DataFrame, energy_sectors: list[str], filename: str):
    # Columns to check: ('Imp', 'ENERGY')
    col = ('Imp', 'ENERGY')

    # Rows to check
    rows = [('Tax', 'AP Imp'), ('Tax', 'AP Dom')]

    for row in rows:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            value = IOT.loc[row, col]

        if value.squeeze() != 0:
            raise AssertionError(
                f"File '{filename}': Non-zero value found at row {row} and column {col}: {value}"
            )
    print(f"Test passed for file '{filename}' and rows {rows} in column {col}.")

for IOT_file in IOTs_folder.glob("*.csv"):
    IOT = pd.read_csv(
        IOT_file,
        header=[0, 1],
        index_col=[0, 1])
    test_energy_C_Ap_zero(IOT, energy_sectors, IOT_file.name)
    test_tax_AP_Imp_Dom_zero(IOT, energy_sectors, IOT_file.name)
