The reformat_EXIOBASE package provides two parallel pipelines, one for EXIOBASE and one for GLORIA, each function in its own module:

### EXIOBASE

download_EXIOBASE: Downloads EXIOBASE from Zenodo.

⚠️ Reformatting is only supported for versions 3.9.4 and later.

aggregate_EXIOBASE: Aggregates the EXIOBASE database based on region and sector mappings provided by the user.

reformat_EXIOBASE: Produces N CSV files—one for each region—containing a restructured version of EXIOBASE data in KLEM format.

### GLORIA

aggregate_GLORIA: Parses a GLORIA MRIO release and aggregates it based on region and sector mappings provided by the user, mirroring aggregate_EXIOBASE. Parsing is done internally with parse_gloria_lowmem, a low-memory parser that reads GLORIA's large transaction files in chunks instead of loading them fully into memory.

reformat_GLORIA: Produces the same SCAF/GTAP-style regional CSV output as reformat_EXIOBASE, adapted for GLORIA's basic-price system (no energy reallocation step, and a different consumption-tax allocation approach—see the function's docstring in reformat_IOT.py for details).

## Installation

>pip install git+https://github.com/RitaMaestri/reformat_EXIOBASE.git

## Installing a specific version

To pin the install to a specific tagged release instead of the latest commit on main, append `@<tag>` to the URL, e.g. to install v0.2.0:

>pip install git+https://github.com/RitaMaestri/reformat_EXIOBASE.git@v0.2.0

Available tags/releases can be found on the [GitHub tags page](https://github.com/RitaMaestri/reformat_EXIOBASE/tags).

## Installation of a new version

You need to uninstall the old version and reinstall the new.

>pip uninstall reformat-EXIOBASE -y

>pip install git+https://github.com/RitaMaestri/reformat_EXIOBASE.git@v0.2.0 --no-cache-dir

To instead track the latest main branch rather than a specific tag, omit the `@<tag>` suffix:

>pip install git+https://github.com/RitaMaestri/reformat_EXIOBASE.git --no-cache-dir


## Example usage
Example scripts are provided in example/run_EXIOBASE.py and example/run_GLORIA.py.

In example/run_EXIOBASE.py, the user must define:

-The output folder for each of the three steps.

-The paths to region and sector mapping files for aggregation.

-The year, system (ixi or pxp), and EXIOBASE version to download.

In example/run_GLORIA.py, the user must define:

-The path to the folder containing the downloaded GLORIA MRIO data.

-The output folders for the aggregation and reformatting steps.

-The paths to region and sector mapping files for aggregation.

-The year to parse.
