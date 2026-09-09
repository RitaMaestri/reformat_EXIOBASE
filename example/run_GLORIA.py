from reformat_exiobase.aggregate_GLORIA import aggregate_GLORIA
from reformat_exiobase.reformat_EXIOBASE import reformat_GLORIA
from pathlib import Path

current_file = Path(__file__).resolve()
current_path = current_file.parent

output_path = current_path / "output"
input_path = current_path / "data"
aggregation_path = output_path / "aggregation_GLORIA"
reformat_path = output_path / "reformat_GLORIA"
aggregation_path.mkdir(parents=True, exist_ok=True)
reformat_path.mkdir(parents=True, exist_ok=True)

mappings_path = current_path / "mappings"
reg_map_file = mappings_path / "map_regions_GLORIA.csv"
sec_map_file = mappings_path / "map_sectors_GLORIA.csv"

year = 2020

aggregate_GLORIA(
    reg_map_path=str(reg_map_file),
    sec_map_path=str(sec_map_file),
    output_path=str(aggregation_path),
    input_path=str(input_path),
    year=year,
)

reformat_GLORIA(aggregation_folder=str(aggregation_path), reformat_folder=str(reformat_path))
