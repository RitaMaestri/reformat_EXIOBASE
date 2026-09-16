"""
Extract raw GLORIA cell values (not pymrio-derived coefficients) for one
region's sectors, split into domestic/import/export and intermediate/final
consumers, in both basic prices and product tax/subsidy flows.

Unlike parse_GLORIA.parse_gloria_lowmem, this never runs pymrio's IO
construction -- it reads exactly the row/column positions needed straight
out of the raw T/Y/V CSVs (using the same region/sector ordering convention
verified in parse_GLORIA.py) and writes them out as labeled, long-format
CSVs for direct inspection. See docs/gloria_raw_flow_extraction.md for the
full methodology (how positions are inferred from unlabeled raw CSVs, the
two memory-safe read patterns, and how the output was verified).
"""

import glob
import os

import numpy as np
import pandas as pd

from ..parse_GLORIA import _first_match

UNIT = "'000 US$"

DEFAULT_TARGET_SECTORS = (
    "Basic inorganic chemicals",
    "Machinery and equipment",
    "Electrical equipment",
)

DEFAULT_FOLDER_NAMES = {
    "Basic inorganic chemicals": "Chemical",
    "Machinery and equipment": "Equipment",
    "Electrical equipment": "Electrical_equipment",
}


def _resolve_gloria_paths(path, year, version):
    path = os.path.abspath(os.path.normpath(str(path)))
    mrio_path = glob.glob(os.path.join(path, f"GLORIA_MRIOs_{version}_{year}*"))[0]
    readme_path = glob.glob(os.path.join(path, f"GLORIA_ReadMe_0{version}.xlsx"))[0]
    return mrio_path, readme_path


def _markup_subfolder(markup):
    return {
        "001": "Basic prices",
        "004": "Taxes on product",
        "005": "Subsidies on Products",
    }[markup]


def _ty_paths(mrio_path, year, version, markup):
    subfolder = os.path.join(mrio_path, _markup_subfolder(markup))
    ext = f"Markup{markup}(full)"
    t_path = _first_match(subfolder, f"_120secMother_AllCountries_002_T-Results_{year}_0{version}_{ext}.csv")
    y_path = _first_match(subfolder, f"_120secMother_AllCountries_002_Y-Results_{year}_0{version}_{ext}.csv")
    return t_path, y_path


def _va_path(mrio_path, year, version):
    subfolder = os.path.join(mrio_path, _markup_subfolder("001"))
    return _first_match(subfolder, f"_120secMother_AllCountries_002_V-Results_{year}_0{version}_Markup001(full).csv")


def load_gloria_labels(readme_path):
    """Ordered region/sector/category lists, matching the raw files' fixed row/column order.

    Verified in parse_GLORIA.py and independently cross-checked against the
    'Sequential region-sector labels' sheet: sheet row order == raw file
    position order, so 0-based sheet-row index doubles as file position.
    """
    regions = pd.read_excel(readme_path, sheet_name="Regions")
    sectors = pd.read_excel(readme_path, sheet_name="Sectors")["Sector_names"].to_list()
    va_fd = pd.read_excel(readme_path, sheet_name="Value added and final demand")
    return {
        "region_acronyms": regions["Region_acronyms"].to_list(),
        "region_names": regions["Region_names"].to_list(),
        "sectors": sectors,
        "va_cats": va_fd["Value_added_names"].to_list(),
        "fd_cats": va_fd["Final_demand_names"].to_list(),
    }


def compute_positions(labels, region_acronym, target_sectors):
    """All row/column offsets needed to extract `region_acronym`'s `target_sectors`.

    T/V column axis and T row axis share the same (region, system, sector)
    layout, region-major then Industry/Product then sector-minor -- see
    parse_GLORIA.parse_gloria_lowmem's `itertools.product(regions, system,
    sectors)` labels_T construction.
    """
    region_acronyms = labels["region_acronyms"]
    sectors = labels["sectors"]
    n_regions = len(region_acronyms)
    n_sectors = len(sectors)
    n_va = len(labels["va_cats"])
    n_fd = len(labels["fd_cats"])
    region_block = 2 * n_sectors

    region_pos = region_acronyms.index(region_acronym)
    sector_pos = {s: sectors.index(s) for s in target_sectors}

    product_row = {s: region_pos * region_block + n_sectors + sector_pos[s] for s in target_sectors}
    industry_col_for_sector = {s: region_pos * region_block + sector_pos[s] for s in target_sectors}

    own_industry_cols = list(range(region_pos * region_block, region_pos * region_block + n_sectors))
    own_y_cols = list(range(region_pos * n_fd, region_pos * n_fd + n_fd))
    own_va_rows = list(range(region_pos * n_va, region_pos * n_va + n_va))

    other_region_positions = [r for r in range(n_regions) if r != region_pos]

    other_industry_cols = []
    other_y_cols = []
    for r in other_region_positions:
        other_industry_cols.extend(range(r * region_block, r * region_block + n_sectors))
        other_y_cols.extend(range(r * n_fd, r * n_fd + n_fd))

    # (region_pos, sector) -> product row, for every OTHER region's target sectors
    other_target_product_rows = []
    for r in other_region_positions:
        for s in target_sectors:
            other_target_product_rows.append((r, s, r * region_block + n_sectors + sector_pos[s]))

    return dict(
        region_pos=region_pos,
        product_row=product_row,
        industry_col_for_sector=industry_col_for_sector,
        own_industry_cols=own_industry_cols,
        own_y_cols=own_y_cols,
        own_va_rows=own_va_rows,
        other_industry_cols=other_industry_cols,
        other_y_cols=other_y_cols,
        other_target_product_rows=other_target_product_rows,
    )


def _industry_col_label(col, labels, region_block, n_sectors):
    region_pos, sector_pos = divmod(col, region_block)
    return labels["region_acronyms"][region_pos], labels["region_names"][region_pos], labels["sectors"][sector_pos]


def _y_col_label(col, labels, n_fd):
    region_pos, fd_pos = divmod(col, n_fd)
    return labels["region_acronyms"][region_pos], labels["region_names"][region_pos], labels["fd_cats"][fd_pos]


def _buyer_frame(col_positions, labels, kind):
    """Build the buyer-side label columns for a list of column positions."""
    n_sectors = len(labels["sectors"])
    region_block = 2 * n_sectors
    n_fd = len(labels["fd_cats"])
    if kind == "industry":
        rows = [_industry_col_label(c, labels, region_block, n_sectors) for c in col_positions]
        buyer_type = "Intermediate_Industry"
    else:
        rows = [_y_col_label(c, labels, n_fd) for c in col_positions]
        buyer_type = "FinalDemand"
    df = pd.DataFrame(rows, columns=["buyer_region_acronym", "buyer_region_name", "buyer_label"])
    df.insert(0, "buyer_type", buyer_type)
    return df


def _to_long(values, seller_df, buyer_df, value_type):
    """Cartesian-melt a (n_sellers x n_buyers) numeric array against its row/col label frames."""
    n_s, n_b = values.shape
    seller_rep = seller_df.loc[seller_df.index.repeat(n_b)].reset_index(drop=True)
    buyer_rep = pd.concat([buyer_df] * n_s, ignore_index=True)
    out = pd.concat([seller_rep, buyer_rep], axis=1)
    out["value_type"] = value_type
    out["value"] = np.asarray(values).reshape(-1)
    out["unit"] = UNIT
    return out


def _read_row_span(csv_path, row_positions):
    """Pattern A: read the minimal contiguous span covering `row_positions`, all columns."""
    span_start, span_end = min(row_positions), max(row_positions)
    block = pd.read_csv(csv_path, header=None, skiprows=span_start, nrows=span_end - span_start + 1, dtype=np.float64)
    return block.iloc[[r - span_start for r in row_positions]].reset_index(drop=True)


def _read_narrow_columns(csv_path, col_positions, chunksize):
    """Pattern B: stream the whole file, keeping only the requested columns."""
    col_sorted = sorted(col_positions)
    chunks = [chunk for chunk in pd.read_csv(csv_path, header=None, usecols=col_sorted, chunksize=chunksize, dtype=np.float64)]
    full = pd.concat(chunks, ignore_index=True)
    full.columns = col_sorted
    return full


def _seller_frame(sectors_in_row_order, region_acronym, region_name):
    return pd.DataFrame(
        {
            "seller_region_acronym": region_acronym,
            "seller_region_name": region_name,
            "seller_sector": list(sectors_in_row_order),
            "seller_system": "Product",
        }
    )


def extract_flows_for_valuation(t_path, y_path, labels, positions, region_acronym, target_sectors, chunksize, value_type):
    """Domestic/import/export long-format DataFrames for one T/Y valuation (one markup)."""
    region_pos = positions["region_pos"]
    region_name = labels["region_names"][region_pos]

    # --- domestic + export: China's target-sector rows, all columns (Pattern A) ---
    t_rows = _read_row_span(t_path, [positions["product_row"][s] for s in target_sectors])
    y_rows = _read_row_span(y_path, [positions["product_row"][s] for s in target_sectors])
    seller_own = _seller_frame(target_sectors, region_acronym, region_name)

    domestic_t = _to_long(
        t_rows[positions["own_industry_cols"]].to_numpy(),
        seller_own,
        _buyer_frame(positions["own_industry_cols"], labels, "industry"),
        value_type,
    )
    domestic_y = _to_long(
        y_rows[positions["own_y_cols"]].to_numpy(),
        seller_own,
        _buyer_frame(positions["own_y_cols"], labels, "y"),
        value_type,
    )
    export_t = _to_long(
        t_rows[positions["other_industry_cols"]].to_numpy(),
        seller_own,
        _buyer_frame(positions["other_industry_cols"], labels, "industry"),
        value_type,
    )
    export_y = _to_long(
        y_rows[positions["other_y_cols"]].to_numpy(),
        seller_own,
        _buyer_frame(positions["other_y_cols"], labels, "y"),
        value_type,
    )

    # --- imports: every other region's target-sector rows, China's columns only (Pattern B) ---
    t_narrow = _read_narrow_columns(t_path, positions["own_industry_cols"], chunksize)
    y_narrow = _read_narrow_columns(y_path, positions["own_y_cols"], chunksize)

    import_rows = positions["other_target_product_rows"]
    import_row_idx = [row for _, _, row in import_rows]
    seller_import = pd.DataFrame(
        {
            "seller_region_acronym": [labels["region_acronyms"][r] for r, _, _ in import_rows],
            "seller_region_name": [labels["region_names"][r] for r, _, _ in import_rows],
            "seller_sector": [s for _, s, _ in import_rows],
            "seller_system": "Product",
        }
    )

    import_t = _to_long(
        t_narrow.iloc[import_row_idx].to_numpy(),
        seller_import,
        _buyer_frame(positions["own_industry_cols"], labels, "industry"),
        value_type,
    )
    import_y = _to_long(
        y_narrow.iloc[import_row_idx].to_numpy(),
        seller_import,
        _buyer_frame(positions["own_y_cols"], labels, "y"),
        value_type,
    )

    domestic = pd.concat([domestic_t, domestic_y], ignore_index=True)
    imports = pd.concat([import_t, import_y], ignore_index=True)
    exports = pd.concat([export_t, export_y], ignore_index=True)
    return domestic, imports, exports


def extract_value_added(v_path, labels, positions, region_acronym, target_sectors):
    region_pos = positions["region_pos"]
    region_name = labels["region_names"][region_pos]
    cols = [positions["industry_col_for_sector"][s] for s in target_sectors]

    block = pd.read_csv(v_path, header=None, skiprows=min(positions["own_va_rows"]), nrows=len(positions["own_va_rows"]), usecols=cols, dtype=np.float64)
    block = block[cols]  # enforce requested column order (usecols does not preserve it)
    block.index = labels["va_cats"]
    block.columns = target_sectors

    long_rows = []
    for sector in target_sectors:
        for va_cat in labels["va_cats"]:
            long_rows.append(
                {
                    "region_acronym": region_acronym,
                    "region_name": region_name,
                    "sector": sector,
                    "system": "Industry",
                    "va_category": va_cat,
                    "value_type": "Markup001_basic_price",
                    "value": block.loc[va_cat, sector],
                    "unit": UNIT,
                }
            )
    return pd.DataFrame(long_rows)


def _net_tax_subsidy(taxes_df, subsidies_df):
    net = taxes_df.copy()
    net["value"] = taxes_df["value"].to_numpy() + subsidies_df["value"].to_numpy()
    net["value_type"] = "Net_taxes_less_subsidies"
    return net


def _all_product_rows(labels, region_acronym):
    """Every (region, sector) Product row in T -- the full universe of possible
    input suppliers, not just the target sectors."""
    n_sectors = len(labels["sectors"])
    region_block = 2 * n_sectors
    rows = [
        (acr, labels["region_names"][r], sec, r * region_block + n_sectors + sec_idx)
        for r, acr in enumerate(labels["region_acronyms"])
        for sec_idx, sec in enumerate(labels["sectors"])
    ]
    df = pd.DataFrame(rows, columns=["seller_region_acronym", "seller_region_name", "seller_sector", "row"])
    df["seller_system"] = "Product"
    df["is_domestic"] = df["seller_region_acronym"] == region_acronym
    return df


def extract_inputs_bought_for_valuation(t_path, labels, positions, region_acronym, target_sectors, chunksize, value_type):
    """Domestic/imported long-format DataFrames of inputs BOUGHT by each target
    sector (buyer fixed to that sector's own Industry column; seller varies
    across all 120 sectors and all 164 regions), for one T valuation."""
    region_pos = positions["region_pos"]
    region_name = labels["region_names"][region_pos]
    buyer_cols = [positions["industry_col_for_sector"][s] for s in target_sectors]

    narrow = _read_narrow_columns(t_path, buyer_cols, chunksize)
    all_rows = _all_product_rows(labels, region_acronym)
    values = narrow.iloc[all_rows["row"].to_list()][buyer_cols].to_numpy()
    seller_cols = all_rows[["seller_region_acronym", "seller_region_name", "seller_sector", "seller_system"]]
    is_domestic = all_rows["is_domestic"].to_numpy()

    domestic_parts, imported_parts = [], []
    for i, sector in enumerate(target_sectors):
        out = seller_cols.copy()
        out["buyer_region_acronym"] = region_acronym
        out["buyer_region_name"] = region_name
        out["buyer_sector"] = sector
        out["buyer_system"] = "Industry"
        out["value_type"] = value_type
        out["value"] = values[:, i]
        out["unit"] = UNIT
        domestic_parts.append(out[is_domestic])
        imported_parts.append(out[~is_domestic])

    domestic = pd.concat(domestic_parts, ignore_index=True)
    imported = pd.concat(imported_parts, ignore_index=True)
    return domestic, imported


def extract_china_target_sector_flows(
    path,
    output_dir,
    year=2020,
    version=59,
    region_acronym="CHN",
    target_sectors=DEFAULT_TARGET_SECTORS,
    folder_names=None,
    chunksize=1000,
):
    """Extract raw GLORIA basic-price and tax/subsidy flows for `region_acronym`'s
    `target_sectors`, and write one subfolder per sector under `output_dir`,
    each containing 7 labeled CSVs (domestic/import/export x basic_price/
    tax_subsidy, plus value_added).
    """
    folder_names = folder_names or DEFAULT_FOLDER_NAMES

    mrio_path, readme_path = _resolve_gloria_paths(path, year, version)
    labels = load_gloria_labels(readme_path)
    positions = compute_positions(labels, region_acronym, target_sectors)

    print("Extracting basic-price flows (Markup001)...")
    t001, y001 = _ty_paths(mrio_path, year, version, "001")
    domestic_bp, imports_bp, exports_bp = extract_flows_for_valuation(
        t001, y001, labels, positions, region_acronym, target_sectors, chunksize, "Markup001_basic_price"
    )

    print("Extracting taxes-on-products flows (Markup004)...")
    t004, y004 = _ty_paths(mrio_path, year, version, "004")
    domestic_tax, imports_tax, exports_tax = extract_flows_for_valuation(
        t004, y004, labels, positions, region_acronym, target_sectors, chunksize, "Markup004_taxes_on_products"
    )

    print("Extracting subsidies-on-products flows (Markup005)...")
    t005, y005 = _ty_paths(mrio_path, year, version, "005")
    domestic_sub, imports_sub, exports_sub = extract_flows_for_valuation(
        t005, y005, labels, positions, region_acronym, target_sectors, chunksize, "Markup005_subsidies_on_products"
    )

    domestic_ts = pd.concat(
        [domestic_tax, domestic_sub, _net_tax_subsidy(domestic_tax, domestic_sub)], ignore_index=True
    )
    imports_ts = pd.concat(
        [imports_tax, imports_sub, _net_tax_subsidy(imports_tax, imports_sub)], ignore_index=True
    )
    exports_ts = pd.concat(
        [exports_tax, exports_sub, _net_tax_subsidy(exports_tax, exports_sub)], ignore_index=True
    )

    print("Extracting value added (Markup001)...")
    v001 = _va_path(mrio_path, year, version)
    value_added = extract_value_added(v001, labels, positions, region_acronym, target_sectors)

    print("Extracting inputs bought (Markup001/004/005)...")
    inputs_dom_bp, inputs_imp_bp = extract_inputs_bought_for_valuation(
        t001, labels, positions, region_acronym, target_sectors, chunksize, "Markup001_basic_price"
    )
    inputs_dom_tax, inputs_imp_tax = extract_inputs_bought_for_valuation(
        t004, labels, positions, region_acronym, target_sectors, chunksize, "Markup004_taxes_on_products"
    )
    inputs_dom_sub, inputs_imp_sub = extract_inputs_bought_for_valuation(
        t005, labels, positions, region_acronym, target_sectors, chunksize, "Markup005_subsidies_on_products"
    )
    inputs_dom_ts = pd.concat(
        [inputs_dom_tax, inputs_dom_sub, _net_tax_subsidy(inputs_dom_tax, inputs_dom_sub)], ignore_index=True
    )
    inputs_imp_ts = pd.concat(
        [inputs_imp_tax, inputs_imp_sub, _net_tax_subsidy(inputs_imp_tax, inputs_imp_sub)], ignore_index=True
    )

    print("Writing per-sector output files...")
    for sector in target_sectors:
        folder = folder_names.get(sector, sector.replace(" ", "_").replace(",", ""))
        sector_dir = os.path.join(str(output_dir), folder)
        os.makedirs(sector_dir, exist_ok=True)

        # sales-side files: this sector is the seller, so filter on seller_sector
        sales_outputs = {
            "domestic_basic_price.csv": domestic_bp,
            "imports_basic_price.csv": imports_bp,
            "exports_basic_price.csv": exports_bp,
            "domestic_tax_subsidy.csv": domestic_ts,
            "imports_tax_subsidy.csv": imports_ts,
            "exports_tax_subsidy.csv": exports_ts,
        }
        for filename, df in sales_outputs.items():
            df[df["seller_sector"] == sector].to_csv(os.path.join(sector_dir, filename), index=False)

        # cost-side files: this sector is the buyer, so filter on buyer_sector
        cost_outputs = {
            "inputs_domestic_basic_price.csv": inputs_dom_bp,
            "inputs_imported_basic_price.csv": inputs_imp_bp,
            "inputs_domestic_tax_subsidy.csv": inputs_dom_ts,
            "inputs_imported_tax_subsidy.csv": inputs_imp_ts,
        }
        for filename, df in cost_outputs.items():
            df[df["buyer_sector"] == sector].to_csv(os.path.join(sector_dir, filename), index=False)

        value_added[value_added["sector"] == sector].to_csv(
            os.path.join(sector_dir, "value_added_basic_price.csv"), index=False
        )

    print(f"Done. Wrote {len(target_sectors)} sector folders under {output_dir}")
