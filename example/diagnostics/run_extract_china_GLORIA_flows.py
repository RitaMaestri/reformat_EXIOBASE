from pathlib import Path

from reformat_exiobase.diagnostics.extract_GLORIA_raw_flows import extract_china_target_sector_flows

current_file = Path(__file__).resolve()
example_path = current_file.parent.parent

input_path = example_path / "data"
output_dir = example_path / "data" / "temp"
output_dir.mkdir(parents=True, exist_ok=True)

year = 2020

extract_china_target_sector_flows(
    path=str(input_path),
    output_dir=str(output_dir),
    year=year,
    region_acronym="CHN",
)
