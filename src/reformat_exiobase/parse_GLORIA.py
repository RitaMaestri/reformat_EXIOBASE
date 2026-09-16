"""
Created on Sun Aug 31 2026

@author: rita
"""

import glob
import os
import zipfile
from collections import namedtuple

import numpy as np
import pandas as pd
import pymrio
import pymrio.tools.ioparser as _ioparser
from pymrio.tools.ioparser import IDX_NAMES, MRIOMetaData

# reuse pymrio's own IO-construction math (A/VA/Q formulas) so this stays
# numerically identical to pymrio.parse_gloria for the parts we don't change
_construct_IO = getattr(_ioparser, "__construct_IO")


def _first_match(directory, suffix):
    """Glob one file, raising a clear error instead of an IndexError on no match."""
    matches = glob.glob(os.path.join(directory, "*" + suffix))
    if not matches:
        raise FileNotFoundError(f"No file matching *{suffix} found in {directory}")
    return matches[0]


def compute_net_production_taxes(va):
    """Net taxes on production (region x sector), from GLORIA's own VA rows.

    Subsidies on production D.39 are already stored as negative values in GLORIA's
    raw data (verified against the source CSV), so netting against Taxes on
    production D.29 is addition, not subtraction.
    """
    return va.loc["Taxes on production D.29"] + va.loc["Subsidies on production D.39"]


def _stream_use_block_sum(csv_paths, is_industry, product_pos, industry_pos, product_index, industry_index, chunksize, label):
    """Extract and sum the product-rows x industry-columns corner from one or more T-shaped csvs, streamed in chunks.

    T004/T005 (taxes/subsidies on products) share T's exact row/column layout, so
    this is the same slicing parse_gloria_lowmem already does inline for T itself to
    build data["U"] -- factored out here so it can run again, unchanged, for T004 and
    T005 without duplicating that loop. Accumulates every csv_path into one array
    instead of building a separate full-size matrix per file and adding them
    afterwards -- with each of these matrices already ~3GB at GLORIA's full
    resolution, materializing T004's and T005's matrices at the same time (on top
    of the V/U matrices already resident from the main T loop) is enough extra
    memory pressure to risk exhausting available RAM.
    """
    arr = np.zeros((len(product_pos), len(industry_pos)), dtype=np.float64)
    for csv_path in csv_paths:
        cursor = 0
        row = 0
        file_label = f"{label} ({os.path.basename(csv_path)})"
        for chunk in pd.read_csv(csv_path, header=None, chunksize=chunksize, dtype=np.float64):
            n = chunk.shape[0]
            chunk_is_industry = is_industry[cursor : cursor + n]
            prod_rows = chunk.to_numpy()[~chunk_is_industry][:, industry_pos]
            arr[row : row + prod_rows.shape[0], :] += prod_rows
            row += prod_rows.shape[0]
            cursor += n
            print(f"  {file_label}: {cursor} rows", end="\r")
        print(f"  {file_label}: {cursor} rows -- done")
    return pd.DataFrame(arr, index=product_index, columns=industry_index)


def _load_y_tax_block(csv_path, is_industry, product_index, y_columns):
    """Load a Y004/Y005-shaped csv (same layout as Y) and keep only its product rows."""
    values = pd.read_csv(csv_path, header=None, dtype=np.float64).to_numpy()[~is_industry]
    return pd.DataFrame(values, index=product_index, columns=y_columns)


def compute_net_sales_tax_matrices(mrio_path, year, version, is_industry, product_pos, industry_pos,
                                    product_index, industry_index, y_columns, chunksize):
    """Net taxes less subsidies on products purchased, at full seller x buyer resolution.

    GLORIA has no ready-made row for this (unlike production taxes). It's derived from
    the tax/subsidy markup matrices: Markup004 = taxes on products, Markup005 =
    subsidies on products, each published separately for the T (intermediate
    transactions) and Y (final demand) matrices.

    Unlike a single collapsed total (the previous version of this function), this
    keeps the full seller-row x buyer-column resolution -- shaped exactly like
    data["U"]/data["Y"] -- so downstream code can attribute tax correctly per buying
    sector and per domestic/imported source instead of applying one uniform rate to
    every buyer (see reformat_lib.compute_real_consumption_taxes). It rides through
    aggregate_GLORIA's region/sector aggregation the same way Z/Y do, since it shares
    their exact row/column labeling.

    Subsidies (Markup005) are already stored as negative values in the raw files
    (verified against the source CSVs, same convention used elsewhere in this
    module), so netting against taxes (Markup004) is addition, not subtraction.
    """
    t004_path = _first_match(
        os.path.join(mrio_path, "Taxes on product"),
        f"_120secMother_AllCountries_002_T-Results_{str(year)}_0{str(version)}_Markup004(full).csv",
    )
    t005_path = _first_match(
        os.path.join(mrio_path, "Subsidies on Products"),
        f"_120secMother_AllCountries_002_T-Results_{str(year)}_0{str(version)}_Markup005(full).csv",
    )
    y004_path = _first_match(
        os.path.join(mrio_path, "Taxes on product"),
        f"_120secMother_AllCountries_002_Y-Results_{str(year)}_0{str(version)}_Markup004(full).csv",
    )
    y005_path = _first_match(
        os.path.join(mrio_path, "Subsidies on Products"),
        f"_120secMother_AllCountries_002_Y-Results_{str(year)}_0{str(version)}_Markup005(full).csv",
    )

    U_tax_net = _stream_use_block_sum(
        [t004_path, t005_path], is_industry, product_pos, industry_pos, product_index, industry_index, chunksize,
        "T004/T005 matrix",
    )
    Y_tax_net = (
        _load_y_tax_block(y004_path, is_industry, product_index, y_columns)
        + _load_y_tax_block(y005_path, is_industry, product_index, y_columns)
    )
    return U_tax_net, Y_tax_net


def _collapse_to_buyer_region_total(tax_block):
    """Collapse a seller-row x buyer-column tax matrix to one total per (buyer region, product sector).

    Sums away the seller region (keeping the product's own sector) and the buyer's
    own sector/category (keeping only the buyer's region) -- the same buyer-region x
    product-sector total the old compute_net_sales_taxes produced, just derived from
    the already-built full-resolution matrix instead of re-reading the source csvs.
    """
    by_product_sector = tax_block.groupby(level="sector").sum()
    return by_product_sector.T.groupby(level="region").sum().T


def parse_gloria_lowmem(path, year, version=59, price="bp", country_names="gloria", construct="B", chunksize=1000):
    """Parse the GLORIA database without ever loading the full T matrix into memory.

    Equivalent to pymrio.parse_gloria, but reads the (huge, e.g. 39360x39360)
    raw transactions CSV in row chunks and extracts the Supply (V) and Use (U)
    sub-matrices directly, instead of loading T whole and slicing V/U out of it.
    This works because GLORIA's T is structurally a block matrix
    T = [[0, V], [U, 0]] over the Industry/Product row and column split, so V
    and U (each about a quarter of T's cells) contain everything T has to offer.

    Only supports the extracted-folder GLORIA_MRIOs layout (not a .zip), since
    that's the large-file case this function exists for.
    """
    if country_names[0].lower() == "g":
        country_col = "Region_acronyms"
    elif country_names[0].lower() == "f":
        country_col = "Region_names"
    else:
        raise ValueError("Parameter country_names must be gloria or full")

    path = os.path.abspath(os.path.normpath(str(path)))

    if price == "bp":
        extension = "Markup001(full)"
    elif price == "pp":
        extension = "Markup005(full)"
    else:
        raise ValueError("price should be bp or pp")

    version_readme = version

    gloria_mrio_files = {
        "T": f"_120secMother_AllCountries_002_T-Results_{str(year)}_0{str(version)}_{extension}.csv",
        "Y": f"_120secMother_AllCountries_002_Y-Results_{str(year)}_0{str(version)}_{extension}.csv",
        "VA": f"_120secMother_AllCountries_002_V-Results_{str(year)}_0{str(version)}_Markup001(full).csv",
    }
    gloria_satellite_files = {
        "Q": f"_120secMother_AllCountries_002_TQ-Results_{str(year)}_0{str(version)}_Markup001(full).csv",
        "QY": f"_120secMother_AllCountries_002_YQ-Results_{str(year)}_0{str(version)}_Markup001(full).csv",
    }

    header = namedtuple("header", "index columns index_names, column_names")
    gloria_header_spec = {
        "Q": header(index="labels_Q", columns="labels_T", index_names=IDX_NAMES["F_row_cat_unit"], column_names=IDX_NAMES["T_col"]),
        "QY": header(index="labels_Q", columns="labels_Y", index_names=IDX_NAMES["F_row_cat_unit"], column_names=IDX_NAMES["Y_col2"]),
        "VA": header(index="labels_VA", columns="labels_T", index_names=IDX_NAMES["VA_row_region"], column_names=IDX_NAMES["T_col"]),
        "Y": header(index="labels_T", columns="labels_Y", index_names=IDX_NAMES["T_row"], column_names=IDX_NAMES["Y_col2"]),
    }

    mrio_path = glob.glob(os.path.join(path, f"GLORIA_MRIOs_{str(version)}_{str(year)}*"))[0]
    if os.path.splitext(mrio_path)[1] == ".zip":
        raise ValueError("parse_gloria_lowmem only supports an extracted GLORIA_MRIOs folder, not a .zip")

    meta_rec = MRIOMetaData(location=mrio_path)
    data = {}

    print("Loading Y and VA...")
    data["Y"] = pd.read_csv(glob.glob(os.path.join(mrio_path, "*" + gloria_mrio_files["Y"]))[0], header=None)
    data["VA"] = pd.read_csv(glob.glob(os.path.join(mrio_path, "*" + gloria_mrio_files["VA"]))[0], header=None)

    print("Loading satellite accounts...")
    satellite_path = glob.glob(os.path.join(path, f"GLORIA_SatelliteAccounts_0{str(version)}_{str(year)}*"))[0]
    if os.path.splitext(satellite_path)[1] == ".zip":
        zip_file = zipfile.ZipFile(satellite_path)
        for key, filename in gloria_satellite_files.items():
            file = [fn for fn in zip_file.namelist() if fn.endswith(filename)][0]
            data[key] = pd.read_csv(zip_file.open(file), header=None)
        zip_file.close()
    else:
        for key, filename in gloria_satellite_files.items():
            data[key] = pd.read_csv(glob.glob(os.path.join(satellite_path, "*" + filename))[0], header=None)

    print("Loading labels from ReadMe...")
    gloria_meta_path = glob.glob(os.path.join(path, f"GLORIA_ReadMe_0{str(version_readme)}.xlsx"))[0]
    regions = pd.read_excel(gloria_meta_path, sheet_name="Regions")[country_col]
    sectors = pd.read_excel(gloria_meta_path, sheet_name="Sectors")["Sector_names"]
    va_fd_sheet = pd.read_excel(gloria_meta_path, sheet_name="Value added and final demand")
    fd_cats = va_fd_sheet["Final_demand_names"].to_list()
    va_cats = va_fd_sheet["Value_added_names"].to_list()
    satellite_cats = pd.read_excel(gloria_meta_path, sheet_name="Satellites")

    system = ["Industry", "Product"]
    import itertools

    labels_T = pd.DataFrame(itertools.product(regions, system, sectors))
    labels_Y = pd.DataFrame(itertools.product(regions, fd_cats))
    labels_VA = pd.DataFrame(itertools.product(regions, va_cats))
    labels_Q = satellite_cats[["Sat_indicator", "Sat_head_indicator", "Sat_unit"]]
    labels = {"labels_T": labels_T, "labels_Y": labels_Y, "labels_VA": labels_VA, "labels_Q": labels_Q}

    for key in gloria_header_spec:
        data[key].columns = labels[gloria_header_spec[key].columns].set_index(list(labels[gloria_header_spec[key].columns])).index
        data[key].columns.names = gloria_header_spec[key].column_names
        data[key].index = labels[gloria_header_spec[key].index].set_index(list(labels[gloria_header_spec[key].index])).index
        data[key].index.names = gloria_header_spec[key].index_names

    print("Streaming T matrix to extract V (supply) and U (use)...")
    t_path = glob.glob(os.path.join(mrio_path, "*" + gloria_mrio_files["T"]))[0]

    full_index = pd.MultiIndex.from_frame(labels_T, names=IDX_NAMES["T_row"])
    is_industry = full_index.get_level_values("system") == "Industry"
    is_product = ~is_industry
    industry_pos = np.where(is_industry)[0]
    product_pos = np.where(is_product)[0]

    industry_index = full_index[industry_pos].droplevel("system")
    product_index = full_index[product_pos].droplevel("system")

    n_total = len(full_index)
    V_arr = np.empty((len(industry_pos), len(product_pos)), dtype=np.float64)
    U_arr = np.empty((len(product_pos), len(industry_pos)), dtype=np.float64)

    cursor = 0
    v_row = 0
    u_row = 0
    for chunk in pd.read_csv(t_path, header=None, chunksize=chunksize, dtype=np.float64):
        n = chunk.shape[0]
        chunk_is_industry = is_industry[cursor : cursor + n]
        values = chunk.to_numpy()

        ind_rows = values[chunk_is_industry][:, product_pos]
        prod_rows = values[~chunk_is_industry][:, industry_pos]

        V_arr[v_row : v_row + ind_rows.shape[0], :] = ind_rows
        U_arr[u_row : u_row + prod_rows.shape[0], :] = prod_rows
        v_row += ind_rows.shape[0]
        u_row += prod_rows.shape[0]
        cursor += n
        print(f"  T matrix: {cursor}/{n_total} rows", end="\r")
    print(f"  T matrix: {cursor}/{n_total} rows -- done")

    data["V"] = pd.DataFrame(V_arr, index=industry_index, columns=product_index)
    data["U"] = pd.DataFrame(U_arr, index=product_index, columns=industry_index)

    print("Streaming tax/subsidy-on-products matrices for net sales tax...")
    data["U_tax_net"], data["Y_tax_net"] = compute_net_sales_tax_matrices(
        mrio_path, year, version, is_industry, product_pos, industry_pos,
        product_index, industry_index, data["Y"].columns, chunksize,
    )

    print("Checking for empty countries...")
    row_sum = data["V"].groupby(level="region").sum().sum(axis=1).add(
        data["U"].groupby(level="region").sum().sum(axis=1), fill_value=0
    )
    column_sum = data["V"].T.groupby(level="region").sum().sum(axis=1).add(
        data["U"].T.groupby(level="region").sum().sum(axis=1), fill_value=0
    )
    empty_countries = row_sum[(row_sum == 0) & (column_sum == 0)].index.to_list()

    for key in ("V", "U", "Y", "VA", "Q", "QY", "U_tax_net", "Y_tax_net"):
        if "region" in data[key].columns.names:
            if empty_countries:
                meta_rec._add_modify(f"Remove empty countries ({empty_countries}) columns from {key}")
            data[key] = data[key].drop(empty_countries, axis=1, level=0)
        if "region" in data[key].index.names:
            if empty_countries:
                meta_rec._add_modify(f"Remove empty countries ({empty_countries}) row from {key}")
            data[key] = data[key].drop(empty_countries, axis=0, level=0)

    if empty_countries:
        print(f"  Removed empty countries: {empty_countries}")
    else:
        print("  None found.")

    U_tax_net = data.pop("U_tax_net")
    Y_tax_net = data.pop("Y_tax_net")
    net_sales_tax = _collapse_to_buyer_region_total(U_tax_net).add(
        _collapse_to_buyer_region_total(Y_tax_net), fill_value=0.0
    ).T.stack()
    net_sales_tax.index.names = ["region", "sector"]

    # Remove 0s in value added, final demand and satellites (redundant Industry/Product level)
    data["VA"] = data["VA"].loc[:, data["VA"].columns.get_level_values(1) == "Industry"]
    data["VA"].columns = data["VA"].columns.droplevel(1)

    # VA's row index carries (region, inputtype), but the raw data is block-diagonal
    # by region (a region's VA is nonzero only in that region's own industry columns),
    # so the row-region is redundant with the column-region and can be dropped by
    # summing -- no values actually mix, since only one region contributes a nonzero
    # term per column. This also makes VA's row index a plain category index like any
    # other pymrio extension, so pymrio's own aggregate() aggregates it correctly.
    data["VA"] = data["VA"].groupby(data["VA"].index.get_level_values("inputtype")).sum()
    data["VA"].index.name = "inputtype"

    data["VA"].loc["Other net taxes on production"] = compute_net_production_taxes(data["VA"])
    data["VA"].loc["Taxes less subsidies on products purchased: Total"] = net_sales_tax

    data["Y"] = data["Y"].loc[data["Y"].index.get_level_values(1) == "Product", :]
    data["Y"].index = data["Y"].index.droplevel(1)

    data["Q"] = data["Q"].loc[:, data["Q"].columns.get_level_values(1) == "Industry"]
    data["Q"].columns = data["Q"].columns.droplevel(1)

    print("Constructing IO matrices...")
    gloria_data = _construct_IO(data, construct=construct)

    A_unit = pd.DataFrame(
        data=["‘000 US$"] * len(gloria_data["A"].index),
        index=gloria_data["A"].index,
        columns=["unit"],
    )
    VA_unit = pd.DataFrame(
        data=["‘000 US$"] * len(gloria_data["VA"].index),
        index=gloria_data["VA"].index,
        columns=["unit"],
    )

    Q_unit = pd.DataFrame(gloria_data["Q"].index.get_level_values(2))
    gloria_data["Q"].index = gloria_data["Q"].index.droplevel(2)
    gloria_data["QY"].index = gloria_data["QY"].index.droplevel(2)
    Q_unit.columns = IDX_NAMES["unit"]
    Q_unit.index = gloria_data["Q"].index

    gloria = pymrio.IOSystem(
        A=gloria_data["A"],
        Y=gloria_data["Y"],
        unit=A_unit,
        Q={
            "name": "Q",
            "unit": Q_unit,
            "F": gloria_data["Q"],
            "F_Y": gloria_data["QY"],
        },
        VA={
            "name": "VA",
            "F": gloria_data["VA"],
            "unit": VA_unit,
        },
        meta=meta_rec,
    )

    # Full-resolution net tax matrices, stashed as extra attributes rather than
    # folded into VA.F -- aggregate_GLORIA reindexes these to gloria.Z/gloria.Y's
    # order right before calling .aggregate(), which picks them up automatically
    # (same seller-row x buyer-column labeling as Z/Y) without any change to
    # pymrio's own aggregation code. See reformat_lib.compute_real_consumption_taxes
    # for how the aggregated result gets used.
    gloria.VA.tax_on_intermediate = U_tax_net
    gloria.VA.tax_on_final_demand = Y_tax_net

    print("Parsing complete.")
    return gloria
