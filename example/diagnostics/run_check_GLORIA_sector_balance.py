from pathlib import Path

import pandas as pd

from reformat_exiobase.diagnostics.check_GLORIA_sector_balance import compute_sector_accounting_balance
from reformat_exiobase.diagnostics.extract_GLORIA_raw_flows import DEFAULT_FOLDER_NAMES

current_file = Path(__file__).resolve()
example_path = current_file.parent.parent

output_dir = example_path / "data" / "temp"

balance = compute_sector_accounting_balance(
    output_dir=str(output_dir),
    folders=DEFAULT_FOLDER_NAMES.values(),
)

pd.set_option("display.width", 200)
print(balance.to_string(index=False))
