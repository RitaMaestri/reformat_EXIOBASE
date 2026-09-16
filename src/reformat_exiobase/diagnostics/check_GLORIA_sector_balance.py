"""
Sanity-check the per-sector flows extracted by extract_GLORIA_raw_flows.py
against a basic supply-use accounting identity:

    VA + intermediate inputs (basic price) + net taxes/subsidies on inputs + imports
        == intermediate demand + final demand + exports

Both sides are built purely from the extracted CSVs, so a small residual is
expected (rounding, close-to-zero cleanup) while a large one flags a bug in
the extraction.
"""

import os

import pandas as pd


def _sum_column(path, column="value"):
    return pd.read_csv(path)[column].sum()


def _sum_filtered(paths, column, value, sum_column="value"):
    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    return combined.loc[combined[column] == value, sum_column].sum()


def compute_sector_accounting_balance(output_dir, folders):
    """Compute the LHS/RHS accounting balance for each sector folder.

    Parameters
    ----------
    output_dir : str or os.PathLike
        Directory containing one subfolder per sector, as produced by
        extract_china_target_sector_flows.
    folders : iterable of str
        Sector folder names under output_dir (e.g. DEFAULT_FOLDER_NAMES.values()).

    Returns
    -------
    pandas.DataFrame
        One row per sector with VA, intermediate_inputs_basic,
        net_taxes_less_subsidies_on_inputs, imports, LHS, intermediate_demand,
        final_demand, exports, RHS, residual, relative_residual.
    """
    rows = []
    for folder in folders:
        sector_dir = os.path.join(output_dir, folder)

        va = _sum_column(os.path.join(sector_dir, "value_added_basic_price.csv"))

        intermediate_inputs_basic = _sum_column(
            os.path.join(sector_dir, "inputs_domestic_basic_price.csv")
        ) + _sum_column(os.path.join(sector_dir, "inputs_imported_basic_price.csv"))

        net_taxes_less_subsidies_on_inputs = _sum_filtered(
            [
                os.path.join(sector_dir, "inputs_domestic_tax_subsidy.csv"),
                os.path.join(sector_dir, "inputs_imported_tax_subsidy.csv"),
            ],
            column="value_type",
            value="Net_taxes_less_subsidies",
        )

        imports = _sum_column(os.path.join(sector_dir, "imports_basic_price.csv"))

        lhs = va + intermediate_inputs_basic + net_taxes_less_subsidies_on_inputs + imports

        demand_paths = [
            os.path.join(sector_dir, "domestic_basic_price.csv"),
            os.path.join(sector_dir, "imports_basic_price.csv"),
        ]
        intermediate_demand = _sum_filtered(
            demand_paths, column="buyer_type", value="Intermediate_Industry"
        )
        final_demand = _sum_filtered(demand_paths, column="buyer_type", value="FinalDemand")

        exports = _sum_column(os.path.join(sector_dir, "exports_basic_price.csv"))

        rhs = intermediate_demand + final_demand + exports

        residual = lhs - rhs
        relative_residual = residual / rhs

        rows.append(
            {
                "sector": folder,
                "VA": va,
                "intermediate_inputs_basic": intermediate_inputs_basic,
                "net_taxes_less_subsidies_on_inputs": net_taxes_less_subsidies_on_inputs,
                "imports": imports,
                "LHS": lhs,
                "intermediate_demand": intermediate_demand,
                "final_demand": final_demand,
                "exports": exports,
                "RHS": rhs,
                "residual": residual,
                "relative_residual": relative_residual,
            }
        )

    return pd.DataFrame(rows)
