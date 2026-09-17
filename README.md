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

## Output format

`reformat_EXIOBASE` and `reformat_GLORIA` each write one CSV per region (e.g. `CHN.csv`) into the reformat output folder. Every file is a single SCAF/GTAP-style regional table with a `(Category, Subcategory)` MultiIndex on both rows and columns, built by the same shared `build_regional_IOTs` — only how the tax rows (block 5 below) are computed differs between the two pipelines (see `reformat_IOT.py` docstrings).

### Rows — resources / cost side, top to bottom

1. `CI_imp` / `CI_dom` (`∑` each) / `CI ∑∑` — intermediate inputs purchased by each sector, at basic price, split into imported and domestic. Together with the matching column block below, this is the square intermediate-consumption (IC) block.
2. `VA` `∑` — value added (labor, capital, …).
3. `M` `∑` — imports, at basic price.
4. `Tax Prod` `∑` — production taxes.
5. `Tax_imp` / `Tax_dom` (`∑` each) / `Tax ∑∑` — taxes on products purchased as intermediate inputs. Unlike every other block, the sectoral rows here (one per sector) are indexed row = buyer, column = product purchased (e.g. `Tax_imp/Chemical` at column `Equipment` = tax Chemical paid buying imported Equipment). Only the `∑` row restores the table's usual "column = sector j" convention: it holds buyer j's own total tax paid across all products, at column j — this is the buyer-side figure `check_unbalance`/`Ressource_tot` use, and `Tax ∑∑ = Tax_imp ∑ + Tax_dom ∑`. This convention allows to compute the output valued at basic prices by summing over the columns, in a way that is consistent with the national accounting convention:
output valued at basic prices = gross value added at basic prices plus intermediate consumption valued at *purchasers’* prices, i.e. tax on intermediate consumptions included (see https://ec.europa.eu/eurostat/documents/3859598/5925693/KS-02-13-269-EN.PDF.pdf/ , paragraph 9.31 for reference)
6. `Tax FBCF Imp/Dom`, `Tax HSLD Imp/Dom`, `Tax AP Imp/Dom` (`∑` each) — final-demand sales taxes on goods (imported or domestic) bought for investment (FBCF), household consumption (HSLD), and government consumption (AP).
7. `Tax Total_TLSP` — total taxes less subsidies on products; a separate, product-side total, independent of blocks 5-6.
8. `Ressource Ressource_tot` — total resources per sector; the row-sum balance check.

### Columns — uses side, left to right

1. `Imp` / `Dom` (`∑` each) / `∑∑` — intermediate demand by buying sector, at basic price (the other side of the square IC block).
2. `Imp`/`Dom` `C_HSLD` `∑` — household final consumption, at purchaser price (basic price + tax).
3. `Imp`/`Dom` `C_Ap` `∑` — government final consumption, at purchaser price.
4. `Imp`/`Dom` `FBCF` `∑` — investment (gross fixed capital formation), at purchaser price.
5. `EXP` / `EXP_TRANSP` `∑` — exports, at basic price (not grossed up by tax).
6. `∑∑∑` — grand total per row; the column-sum balance check.

`check_unbalance`/`check_unbalance_final_format` verify that, for every sector, row block 8's total matches column block 6's total.
