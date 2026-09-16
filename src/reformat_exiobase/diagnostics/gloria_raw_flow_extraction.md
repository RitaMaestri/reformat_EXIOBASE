# Extracting raw GLORIA flows for a region's sectors

`src/reformat_exiobase/extract_GLORIA_raw_flows.py` pulls raw GLORIA cell
values (not the pymrio-aggregated coefficients produced by the main
`aggregate_GLORIA`/`reformat_GLORIA` pipeline) for one region's sectors, so
they can be inspected directly or cross-checked against the aggregated
output. It currently powers `example/run_extract_china_GLORIA_flows.py`,
which extracts China's "Basic inorganic chemicals", "Machinery and
equipment" and "Electrical equipment" sectors into
`example/data/temp/{Chemical,Equipment,Electrical_equipment}/`.

## Why raw extraction, not the normal pipeline

`aggregate_GLORIA` reshapes everything into pymrio's technical-coefficient
matrix (`A`) and lumps unmapped sectors into a single `Composite` bucket.
That's the right format for the reformatting pipeline, but it's not directly
comparable to the raw source data cell-by-cell. This extraction instead
reads the untouched transaction values straight out of GLORIA's raw CSVs,
to validate that the pipeline's sector-level numbers are correct.

## The raw files have no labels — how positions are inferred

GLORIA's raw `T` (transactions), `Y` (final demand) and `V` (value added)
files are plain numeric CSVs: no header row, no index column. A cell's
meaning — which region, which sector, Industry or Product — is implied
entirely by its position in the grid.

That position is fixed by the *order* of rows in
`GLORIA_ReadMe_0<version>.xlsx`'s `Regions`, `Sectors` and `Value added and
final demand` sheets. Those sheets are sorted ascending by `Lfd_Nr`, so a
region's or sector's 0-based row position in the sheet is also its 0-based
position in the raw file. Concretely, `T`'s and `V`'s row/column axis is:

```
itertools.product(regions, ["Industry", "Product"], sectors)
```

region-major, then Industry/Product, then sector-minor — 164 regions × 2 ×
120 sectors = 39,360 rows/columns. `Y`'s columns and `V`'s rows are
`itertools.product(regions, categories)` (6 categories each). This is the
same convention `parse_GLORIA.parse_gloria_lowmem` already relies on to
build its labels, so it isn't a new assumption introduced by this module.

From this, a region + sector's row/column offsets are pure arithmetic
(`compute_positions()` in the module) — no raw MRIO file needs to be opened
to compute them:

```
industry_row(region, sector) = region_pos * 240 + sector_pos
product_row(region, sector)  = region_pos * 240 + 120 + sector_pos
va_row(region, category)     = region_pos * 6 + category_pos
y_col(region, category)      = region_pos * 6 + category_pos
```

This arithmetic was cross-checked once against GLORIA's own `Sequential
region-sector labels` sheet, which spells out the full 39,360-row label list
as ground truth (e.g. row 7920 = `"China (CHN) Growing wheat industry"` =
`33 * 240`, confirming region order, sector order and the Industry/Product
split boundary all line up with the arithmetic above). It also reproduces
`example/data/temp/china_non_composite_sectors_balance_check.csv`'s totals
exactly (see Verification below), which is a second, independent check on
the same offsets.

## Reading multi-gigabyte files without loading them whole

`T` is ~6.3GB per markup (basic price / taxes / subsidies) and is never
loaded in full. Two read patterns cover every extraction:

- **Row-span read** (`_read_row_span`, domestic + export): the target
  region's target-sector rows are few and clustered together, but every
  column may be needed. `pandas.read_csv(..., skiprows=<span start>,
  nrows=<span size>)` reads just that row span — cheap, since skipped rows
  aren't tokenized.
- **Narrow-column read** (`_read_narrow_columns`, imports): imports need the
  target sectors' rows from *every other* region, scattered every 240 rows
  across the whole file — but only the target region's own ~120/6 columns.
  `pandas.read_csv(..., usecols=<target region's columns>, chunksize=1000)`
  streams the whole file while keeping only those columns, bounding memory
  to `n_rows × len(columns)` regardless of file size. (Gotcha: with
  `header=None`, `usecols` returns columns in ascending file order
  regardless of the order requested, so the code sorts and relabels
  explicitly afterward.)

Each of the three valuations (Markup001 basic price, Markup004 taxes on
products, Markup005 subsidies on products) is read once with these
patterns, for all target sectors together — so each markup's `T`/`Y` files
are scanned only once regardless of how many sectors are requested. The
combined result is only split into one output subfolder per sector at the
final write step.

## From raw cells to output rows

Read blocks are melted into long format (`_to_long`): one row per seller ×
buyer cell, labeled with region/sector names rather than positions. For the
tax/subsidy files, a `Net_taxes_less_subsidies` row is added per cell
(Markup004 + Markup005 — addition, not subtraction, since GLORIA already
stores subsidies as negative values; same convention as
`parse_GLORIA.compute_net_sales_taxes`), so the raw components and their net
are all inspectable side by side.

Value added is read only from the basic-price `V` file's own native rows
(Compensation of employees D.1, Taxes/Subsidies on production D.29/D.39, Net
operating surplus B.2n, Net mixed income B.3n, Consumption of fixed capital
K.1) — deliberately **not** `parse_gloria_lowmem`'s derived "Taxes less
subsidies on products purchased: Total" row, since that aggregates trade
data across every buyer region and would make VA trade-dependent.

## Output layout

```
example/data/temp/
├── Chemical/
│   ├── domestic_basic_price.csv
│   ├── imports_basic_price.csv
│   ├── exports_basic_price.csv
│   ├── domestic_tax_subsidy.csv
│   ├── imports_tax_subsidy.csv
│   ├── exports_tax_subsidy.csv
│   └── value_added_basic_price.csv
├── Equipment/          (same 7 files)
└── Electrical_equipment/  (same 7 files)
```

Flow files: `seller_region_acronym, seller_region_name, seller_sector,
seller_system, buyer_type, buyer_region_acronym, buyer_region_name,
buyer_label, value_type, value, unit`.

VA file: `region_acronym, region_name, sector, system, va_category,
value_type, value, unit`.

## Verification

Every number this module produces for China's 3 target sectors reproduces
`example/data/temp/china_non_composite_sectors_balance_check.csv` (an
existing, independently-derived balance check) exactly:

- Domestic and export intermediate-demand totals match the balance check's
  `domestic_intermediate_output` / `intermediate_exports` columns for all 3
  sectors.
- Value added, summed over Compensation of employees + Net operating
  surplus + Net mixed income + Consumption of fixed capital + production
  taxes/subsidies (D.29 + D.39), matches `gva_L_K + production_tax_D29_D39`
  for all 3 sectors.
