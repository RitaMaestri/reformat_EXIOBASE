The reformat_EXIOBASE package provides three main functions, each in its own module:

download_EXIOBASE: Downloads EXIOBASE from Zenodo.

⚠️ Reformatting is only supported for versions 3.9.4 and later.

aggregate_EXIOBASE: Aggregates the EXIOBASE database based on region and sector mappings provided by the user.

reformat_EXIOBASE: Produces N CSV files—one for each region—containing a restructured version of EXIOBASE data in KLEM format.

## Installation

>pip install git+https://github.com/RitaMaestri/reformat_EXIOBASE.git

## Installation of a new version

You need to uninstall the old version and reinstall the new.

>pip uninstall reformat-EXIOBASE -y

>pip install git+https://github.com/RitaMaestri/reformat_EXIOBASE.git --no-cache-dir


## Example usage
An example script is provided in example/run.py.

In the use file, the user must define:

-The output folder for each of the three steps.

-The paths to region and sector mapping files for aggregation.

-The year, system (ixi or pxp), and EXIOBASE version to download.

