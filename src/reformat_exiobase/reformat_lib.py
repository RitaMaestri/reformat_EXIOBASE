"""
Created on Thu Mar 13 14:20:45 2025

@author: rita
"""

import json
import warnings
import pandas as pd
import numpy as np
from . import mappings
import importlib.resources as pkg_resources
from scipy.optimize import least_squares
from typing import List


########################################
########## AGGREGATING DATA ############
########################################


def EXIOBASE_name(SCAF_name, map_final_demand):
    return map_final_demand.loc[map_final_demand['SCAF'] == SCAF_name, 'EXIOBASE_name'].values

def exempt_from_taxes(map_final_demand):
    return map_final_demand.loc[map_final_demand['exempt_cons_taxes'], 'EXIOBASE_name'].values

def final_demand_agents(map_final_demand):
    return map_final_demand["SCAF"].loc[map_final_demand["EXIOBASE_file"] == "Y"].unique()


def reorder_io_columns(df, desired_sector_order):
    order_map = {sector: i for i, sector in enumerate(desired_sector_order)}
    new_cols = sorted(
        df.columns,
        key=lambda x: (x[0], order_map.get(x[1], float('inf')))
    )
    return df[new_cols]


def reorder_io_rows(df, desired_sector_order):
    order_map = {sector: i for i, sector in enumerate(desired_sector_order)}
    new_idx = sorted(
        df.index,
        key=lambda x: (x[0], order_map.get(x[1], float('inf')))
    )
    return df.loc[new_idx]


def reorder_io_matrix(df, desired_sector_order):
    df = reorder_io_rows(df, desired_sector_order)
    df = reorder_io_columns(df, desired_sector_order)
    return df


def reallocate_G_I_energy_to_C(Y, energy_sectors):
    """
    For each column region in Y, move 'Final consumption expenditure by government',
    'Gross fixed capital formation', 'Changes in inventories', and 'Changes in valuables'
    for the given energy sectors into 'Final consumption expenditure by households',
    then set those cells to zero.

    Parameters
    ----------
    Y : pd.DataFrame
        EXIOBASE final demand matrix with a MultiIndex on rows (region, sector)
        and MultiIndex columns (region, category).
    energy_sectors : list of str
        List of sector names (e.g. ["ENERGY"]) to apply the reallocation to.

    Returns
    -------
    pd.DataFrame
        Modified copy of Y.
    """
    Y = Y.copy()

    cols_to_reallocate = [
        'Final consumption expenditure by government',
        'Gross fixed capital formation',
        'Changes in inventories',
        'Changes in valuables',
    ]

    row_mask = Y.index.get_level_values('sector').isin(energy_sectors)

    for col_region in Y.columns.get_level_values('region').unique():
        households_col = (col_region, 'Final consumption expenditure by households')

        for cat in cols_to_reallocate:
            col = (col_region, cat)
            Y.loc[row_mask, households_col] += Y.loc[row_mask, col]
            Y.loc[row_mask, col] = 0

    return Y




def compute_intermediate_domestic_demand(Z):
    regions = Z.columns.get_level_values(0).unique()
    sectors = Z.index.get_level_values(1).unique()

    intermediate_dom = pd.DataFrame(0.0, index=sectors, columns=Z.columns)

    for r in regions:
        for s in sectors:
            # Filter: rows where region == r and sector == s, columns where region == r
            intermediate_dom.loc[s, intermediate_dom.columns.get_level_values(0) == r] = Z.loc[
                (Z.index.get_level_values(0) == r) & (Z.index.get_level_values(1) == s),
                Z.columns.get_level_values(0) == r
            ].to_numpy()

    return intermediate_dom

def compute_intermediate_imports(Z):
    regions = Z.columns.get_level_values(0).unique()
    sectors = Z.index.get_level_values(1).unique()

    intermediate_imp = pd.DataFrame(0.0, index=sectors, columns=Z.columns)

    for r in regions:
        for s in sectors:
            # Rows: region != r and sector == s; Columns: region == r
            intermediate_imp.loc[s, intermediate_imp.columns.get_level_values(0) == r] = Z.loc[
                (Z.index.get_level_values(0) != r) & (Z.index.get_level_values(1) == s),
                Z.columns.get_level_values(0) == r
            ].sum().to_numpy()

    return intermediate_imp


def aggregate_final_demand_agents(Y, map_final_demand):
    # final demand with SCAF categories C,G,I; bilateral trade
    final_demand_agents_SCAF = final_demand_agents(map_final_demand)

    regions = Y.columns.get_level_values("region").unique()

    # Create MultiIndex for columns: (region, category)
    region_index = np.repeat(regions, len(final_demand_agents_SCAF))
    category_index = np.tile(final_demand_agents_SCAF, len(regions))
    zip_columns = list(zip(region_index, category_index))
    final_demand_columns = pd.MultiIndex.from_tuples(zip_columns, names=["region", "category"])

    # Initialize empty DataFrame for SCAF final demand
    final_demand_aggregated_agents = pd.DataFrame(0.0, index=Y.index, columns=final_demand_columns)

    # Fill in the SCAF final demand by summing over mapped EXIOBASE categories
    for r in regions:
        for c in final_demand_agents_SCAF:
            EXIOBASE_categories = EXIOBASE_name(c, map_final_demand)  # Map SCAF category to EXIOBASE ones

            final_demand_aggregated_agents.loc[:, (r, c)] = Y.loc[
                :,
                (r, EXIOBASE_categories)
            ].sum(axis=1)

    return final_demand_aggregated_agents


def compute_final_demand_domestic(Y, map_final_demand):
    fd_aggregated_agents = aggregate_final_demand_agents(Y, map_final_demand)
    # Extract region and category levels from the column MultiIndex
    regions = fd_aggregated_agents.columns.get_level_values("region").unique()
    categories = fd_aggregated_agents.columns.get_level_values("category").unique()

    # Create output DataFrame with same index but single-level columns (categories only)
    fd_dom = pd.DataFrame(
        0.0,
        index=fd_aggregated_agents.index,
        columns=categories
    )

    for region in regions:
        # Filter: rows and columns that belong to the current region
        is_region_row = fd_aggregated_agents.index.get_level_values("region") == region
        is_region_col = fd_aggregated_agents.columns.get_level_values("region") == region

        # Extract the sub-DataFrame for this region and drop the 'region' level from columns
        regional_data = fd_aggregated_agents.loc[is_region_row, is_region_col]
        regional_data.columns = regional_data.columns.droplevel("region")

        # Assign the region-specific data into the corresponding rows of the result
        fd_dom.loc[is_region_row, :] = regional_data

    return fd_dom


def compute_final_demand_imported(Y, map_final_demand):
    fd_aggregated_agents = aggregate_final_demand_agents(Y, map_final_demand)
    # Extract levels from MultiIndex
    regions = fd_aggregated_agents.columns.get_level_values("region").unique()
    categories = fd_aggregated_agents.columns.get_level_values("category").unique()
    sectors = fd_aggregated_agents.index.get_level_values("sector").unique()

    # Initialize the output DataFrame
    fd_imp = pd.DataFrame(
        0.0,
        index=fd_aggregated_agents.index,
        columns=categories,
        dtype=np.float64
    )
    # Loop over each region, category, and sector
    for region in regions:
        for category in categories:
            for sector in sectors:
                # Mask for all rows with a different region and the current sector
                row_mask = (
                    (fd_aggregated_agents.index.get_level_values("region") != region) &
                    (fd_aggregated_agents.index.get_level_values("sector") == sector)
                )

                # Mask for columns matching current region and category
                col_mask = (
                    (fd_aggregated_agents.columns.get_level_values("region") == region) &
                    (fd_aggregated_agents.columns.get_level_values("category") == category)
                )

                # Sum over foreign demand and assign to output
                value = fd_aggregated_agents.loc[row_mask, col_mask].sum().item()
                fd_imp.loc[(region, sector), category] = value

    return fd_imp

def set_to_zero_gvt_energy(df: pd.DataFrame, energy_sectors: List[str]) -> pd.DataFrame:
    """
    For every region in the MultiIndex DataFrame, for each given sector,
    take the value from column 'G', add it to column 'C',
    and set the original 'G' value to zero.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame with index levels [region, sector]
        and columns including 'C' and 'G'.
    sectors : list of str
        List of sector names for which the operation is performed.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame.
    """
    df = df.copy()

    for region in df.index.get_level_values(0).unique():
        for sector in energy_sectors:
            if (region, sector) in df.index:
                g_val = df.loc[(region, sector), 'G']
                df.loc[(region, sector), 'C'] += g_val
                df.loc[(region, sector), 'G'] = 0

    return df

def concatenate_total_demand(fd_dom, fd_imp, intermediate_dom, intermediate_imp):

    # Extract shared regions and categories
    regions = fd_dom.index.get_level_values("region").unique()
    sectors = fd_dom.index.get_level_values("sector").unique()
    final_demand_agents = fd_dom.columns

    # Build MultiIndex for columns: ("imp"/"dom", category)
    imp_dom_index = (
        ["imp"] * len(sectors) + ["dom"] * len(sectors) +
        ["imp", "dom"] * len(final_demand_agents))

    category_index = (
        list(sectors) * 2 + list(np.repeat(final_demand_agents, 2)))

    column_tuples = list(zip(imp_dom_index, category_index))
    final_columns = pd.MultiIndex.from_tuples(column_tuples, names=["imp_dom", "category"])

    # Initialize output DataFrame
    fd = pd.DataFrame(0.0, index=fd_dom.index, columns=final_columns)

    for r in regions:
        for s in sectors:
            fd.loc[r, ("imp", s)] = intermediate_imp.loc[:, (r, s)].to_numpy()
            fd.loc[r, ("dom", s)] = intermediate_dom.loc[:, (r, s)].to_numpy()
        for c in final_demand_agents:
            fd.loc[r, ("imp", c)] = fd_imp.loc[r, c].to_numpy()
            fd.loc[r, ("dom", c)] = fd_dom.loc[r, c].to_numpy()
    return fd



def disaggregate_tax(tax_rates,Z,Y, map_final_demand):

    tot_dem = pd.concat([Z, Y], axis=1)
    tot_dem.columns = tot_dem.columns.set_names('category', level=1)


    tax_rates_df = pd.DataFrame(0, index=tot_dem.index, columns=tot_dem.columns, dtype=np.float64)

    is_exempt_from_taxes = exempt_from_taxes(map_final_demand)

    row_sector_labels = tax_rates_df.index.get_level_values('sector')
    col_region_labels = tax_rates_df.columns.get_level_values('region')

    for (r_ts, s_ts), tax_val in tax_rates.items():

        row_mask = (row_sector_labels == s_ts)

        col_mask = (col_region_labels == r_ts) & \
        (~tax_rates_df.columns.get_level_values(1).isin(is_exempt_from_taxes))

        tax_rates_df.loc[row_mask, col_mask] = tax_val

    tax_df = tax_rates_df * tot_dem

    return tax_df


def adjust_tax_rates(Z: pd.DataFrame, Y: pd.DataFrame, F: pd.DataFrame, map_final_demand) -> pd.Series:

    #the taxes on consumption in the file F.txt for region R and sector S include
    #taxes that are paid abroad for the consumption of the good S produced in R.
    # we extract:
    # -import and export net of taxes.
    # -the tax that is paid on good S consumed in region R and of all origin.
    # the resulting national account is balanced

    def reallocate_tax(tax_values, tax_index, Z, Y, F, map_final_demand):
        tax_rates = pd.Series(tax_values, index=tax_index)
        tax_df = disaggregate_tax(tax_rates, Z, Y, map_final_demand)
        computed_TLSP = tax_df.sum(axis=1)
        discrepancy = F.loc[EXIOBASE_name("Consumption_taxes", map_final_demand)] - computed_TLSP
        return discrepancy.values[0]

    tax_guess = pd.Series(0.01, index=Z.columns)

    result = least_squares(reallocate_tax, tax_guess.values, args=(tax_guess.index, Z, Y, F, map_final_demand), verbose=2)

    return pd.Series(result.x, index=Z.columns)



def compute_imports(net_flows: pd.DataFrame) -> pd.DataFrame:
    regions = net_flows.columns.get_level_values("region").unique()
    sectors = net_flows.index.get_level_values("sector").unique()

    M_columns = pd.MultiIndex.from_product([regions, sectors], names=["region", "sector"])
    M = pd.DataFrame(0.0, index=["M"], columns=M_columns)

    for r in regions:
        for s in sectors:
            row_mask = (net_flows.index.get_level_values("region") != r) & \
                    (net_flows.index.get_level_values("sector") == s)
            col_mask = net_flows.columns.get_level_values("region") == r
            M.loc["M", (r, s)] = net_flows.loc[row_mask, col_mask].sum().sum()

    return M


def compute_exports(net_flows: pd.DataFrame) -> pd.DataFrame:
    regions = net_flows.columns.get_level_values("region").unique()

    X = pd.DataFrame(0.0, index=net_flows.index, columns=["X"])
    for r in regions:
        row_mask = net_flows.index.get_level_values("region") == r
        col_mask = net_flows.columns.get_level_values("region") != r
        X.loc[row_mask] = net_flows.loc[row_mask, col_mask].sum(axis=1).values.reshape(-1, 1)

    return X

######################################
######### FORMATTING DATA ############
######################################

def expand_All_sectors(map_GTAP_format, sectors, cat_col=0, subcat_col=1):
        """
        Expands 'All_sectors' occurrences in the given DataFrame
        by replacing them with specific sectors.

        Args:
        df (pd.DataFrame): DataFrame containing category and subcategory columns.
        cat_col (int): Index of the category column.
        subcat_col (int): Index of the subcategory column.

        Returns:
        pd.DataFrame: Expanded DataFrame with new sector rows.
        """
        expanded_rows = []

        for _, row in map_GTAP_format.iterrows():
            category, subcategory = row[cat_col], row[subcat_col]

            if subcategory == "All_sectors":
                for sector in sectors:
                    new_row = row.copy()
                    new_row[subcat_col] = sector  # Replace with sector name
                    expanded_rows.append(new_row)
            else:
                expanded_rows.append(row)

        return pd.DataFrame(expanded_rows, columns=map_GTAP_format.columns)



def fill_reformat_df_row_wise(reformat_df, row_start, allocation_df, col_start, col_end):
    if isinstance(allocation_df, pd.Series):
        allocation_df = pd.DataFrame(
            [allocation_df.values], columns=allocation_df.index)

    # Handle 1D numpy arrays
    elif isinstance(allocation_df, (np.ndarray, list)) and np.ndim(allocation_df) == 1:
        if len(allocation_df) != col_end:
            raise ValueError(
                "1D input must have exactly col_end elements.")
        allocation_df = pd.DataFrame([allocation_df])

    # Check if it's now a valid 2D shape with 10 columns
    if not isinstance(allocation_df, pd.DataFrame) or allocation_df.shape[1] != col_end:
        raise ValueError(
            "allocation_df must be a DataFrame (or convertible) with exactly col_end columns.")

    num_rows = allocation_df.shape[0]
    row_end = row_start + num_rows

    # Check if it fits within the original_df
    if row_end > reformat_df.shape[0]:
        raise ValueError(
            "Allocation exceeds the number of rows in original_df.")

    # Fill in the block
    reformat_df.iloc[row_start:row_end,
                        col_start:col_end] = allocation_df.values

    # Return the updated row_start
    return row_end


def fill_reformat_df_columnwise(reformat_df, col_start, allocation_df, row_start, row_end):

    # Convert Series (1D) to 1-column DataFrame
    if isinstance(allocation_df, pd.Series):
        allocation_df = pd.DataFrame({0: allocation_df.values})

    # Convert 1D list/array to 1-column DataFrame
    elif isinstance(allocation_df, (np.ndarray, list)) and np.ndim(allocation_df) == 1:
        if len(allocation_df) != row_end - row_start:
            raise ValueError(
                "1D input must match the number of rows to fill.")
        allocation_df = pd.DataFrame({0: allocation_df})

    # Now check if it's a 2D DataFrame with correct number of rows
    num_cols = allocation_df.shape[1]
    if not isinstance(allocation_df, pd.DataFrame) or allocation_df.shape[0] != (row_end - row_start):
        raise ValueError(
            "allocation_df must have exactly the number of rows as row_end - row_start.")

    col_end = col_start + num_cols

    # Bounds check
    if col_end > reformat_df.shape[1]:
        raise ValueError(
            "Allocation exceeds the number of columns in reformat_df.")

    # Fill in the block column-wise
    reformat_df.iloc[row_start:row_end,
                        col_start:col_end] = allocation_df.values

    return col_end


def _report_unbalance(per_region, error_threshold, warning_threshold):
    """
    Shared threshold/reporting logic for both `check_unbalance` (pre-build,
    computed from the raw components) and `check_unbalance_final_format`
    (post-build, read off the assembled GTAP-shaped table) -- factored out
    so the two share one implementation of "how big is too big" instead of
    duplicating it.

    Parameters
    ----------
    per_region : dict
        {region: (sector_labels, cost_array, use_array)}, one entry per
        region. `cost_array`/`use_array` are 1D arrays of total production
        cost / total use per sector (same order as `sector_labels`); the
        unbalance is measured relative to each sector's own scale (mean of
        its cost and use) so it is comparable across sectors of very
        different size.

    Raises a ValueError if the largest relative unbalance (across all
    regions/sectors) exceeds `error_threshold` (default 1%). Otherwise,
    issues a UserWarning listing every (region, sector) pair whose relative
    unbalance exceeds `warning_threshold` (default 1e-4).

    Returns
    -------
    dict
        {region: max relative unbalance in that region}.
    """
    max_unbalance = 0
    unbalance_by_region = {}
    flagged = []  # (relative_unbalance, region, sector)

    for r, (sector_labels, cost_arr, use_arr) in per_region.items():
        abs_unbalance = np.abs(use_arr - cost_arr)
        scale = (np.abs(cost_arr) + np.abs(use_arr)) / 2
        relative_unbalance = np.divide(
            abs_unbalance, scale, out=np.zeros_like(abs_unbalance), where=scale != 0
        )

        max_unbalance_r = relative_unbalance.max() if len(relative_unbalance) else 0.0
        unbalance_by_region[r] = max_unbalance_r
        if max_unbalance_r > max_unbalance:
            max_unbalance = max_unbalance_r

        for sector, rel in zip(sector_labels, relative_unbalance):
            if rel > warning_threshold:
                flagged.append((rel, r, sector))

    print("Max unbalance across regions:", max_unbalance)

    if flagged:
        flagged.sort(reverse=True)
        details = "; ".join(f"{r}/{s}: {rel:.4%}" for rel, r, s in flagged)
        if max_unbalance > error_threshold:
            raise ValueError(
                f"Max unbalance ({max_unbalance:.4%}) exceeds the {error_threshold:.2%} error "
                f"threshold. Offending region/sector pairs (unbalance > {warning_threshold:.4%}): {details}"
            )
        warnings.warn(
            f"Unbalance exceeds {warning_threshold:.4%} for the following region/sector pairs: {details}"
        )

    return unbalance_by_region


def _region_cost_and_use(r, intermediate_dom, intermediate_imp, L, K, R, M, production_taxes,
                          imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand, X):
    """
    Per-(region, sector) total production cost vs. total use, computed
    directly from the pre-build components -- mirrors exactly what
    `check_unbalance_final_format` would read off the '∑'-labeled rows/
    columns of the table `build_regional_IOTs` assembles from these same
    inputs (verified empirically against `build_regional_IOTs`).
    """
    cost = (
        intermediate_imp[r].sum(axis=0)
        + intermediate_dom[r].sum(axis=0)
        + pd.concat([L, K, R]).sum(axis=0)[r]
        + M[r].sum(axis=0)
        + production_taxes[r].sum(axis=0)
        + imp_intermediate_cons_tax.loc[r].sum(axis=0)
        + dom_intermediate_cons_tax.loc[r].sum(axis=0)
        + cons_taxes["imp"]["I"][r].sum(axis=0) + cons_taxes["dom"]["I"][r].sum(axis=0)
        + cons_taxes["imp"]["C"][r].sum(axis=0) + cons_taxes["dom"]["C"][r].sum(axis=0)
        + cons_taxes["imp"]["G"][r].sum(axis=0) + cons_taxes["dom"]["G"][r].sum(axis=0)
    )
    sub_demand = total_demand.loc[r]
    use = sub_demand.sum(axis=1) + X.loc[r]["X"]
    return cost.reindex(sub_demand.index), use.reindex(sub_demand.index), sub_demand


def _zero_negligible_values_by_region(df, threshold=1e-8):
    """
    For each region, zero out values whose absolute size is negligible
    relative to that region's own scale (< threshold * that region's max
    absolute value). Cleans up floating-point noise -- e.g. residue from the
    adjust_tax_rates least-squares fit -- before it's picked up by
    check_unbalance. 'region' lives on the row axis for some callers and the
    column axis for others (see zero_out_regional_noise), so detect which
    and slice there.
    """
    df = df.copy()

    if "region" in (df.index.names or []):
        region_level = df.index.get_level_values("region")
        for r in region_level.unique():
            mask = region_level == r
            block = df.loc[mask]
            max_val = block.abs().to_numpy().max() if block.size else 0.0
            df.loc[mask] = block.mask(block.abs() < threshold * max_val, 0.0)
    elif "region" in (df.columns.names or []):
        region_level = df.columns.get_level_values("region")
        for r in region_level.unique():
            mask = region_level == r
            block = df.loc[:, mask]
            max_val = block.abs().to_numpy().max() if block.size else 0.0
            df.loc[:, mask] = block.mask(block.abs() < threshold * max_val, 0.0)
    else:
        raise ValueError("DataFrame has no 'region' level on either axis.")

    return df


def zero_out_regional_noise(intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
                             imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand,
                             threshold=1e-8):
    """
    Apply _zero_negligible_values_by_region to every pre-build component,
    including each per-agent DataFrame inside cons_taxes. Called right
    before check_unbalance in both reformat_EXIOBASE and reformat_GLORIA so
    the cleaned values flow through the rest of the pipeline (unbalance
    attribution, build_regional_IOTs, CSV output) too.
    """
    intermediate_dom = _zero_negligible_values_by_region(intermediate_dom, threshold)
    intermediate_imp = _zero_negligible_values_by_region(intermediate_imp, threshold)
    L = _zero_negligible_values_by_region(L, threshold)
    K = _zero_negligible_values_by_region(K, threshold)
    R = _zero_negligible_values_by_region(R, threshold)
    M = _zero_negligible_values_by_region(M, threshold)
    X = _zero_negligible_values_by_region(X, threshold)
    production_taxes = _zero_negligible_values_by_region(production_taxes, threshold)
    imp_intermediate_cons_tax = _zero_negligible_values_by_region(imp_intermediate_cons_tax, threshold)
    dom_intermediate_cons_tax = _zero_negligible_values_by_region(dom_intermediate_cons_tax, threshold)
    cons_taxes = {
        imp_dom: {agent: _zero_negligible_values_by_region(df, threshold) for agent, df in agents.items()}
        for imp_dom, agents in cons_taxes.items()
    }
    total_demand = _zero_negligible_values_by_region(total_demand, threshold)

    return (intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
            imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand)


def check_unbalance(regions, intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
                     imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand,
                     error_threshold=0.01, warning_threshold=1e-4):
    """
    Check, for every region and sector, that total production cost matches
    total use -- computed directly from the pre-build components, before
    `build_regional_IOTs` assembles the final GTAP-shaped table. Same
    print/warn/raise contract as `check_unbalance_final_format`.
    """
    per_region = {}
    for r in regions:
        cost, use, sub_demand = _region_cost_and_use(
            r, intermediate_dom, intermediate_imp, L, K, R, M, production_taxes,
            imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand, X)
        per_region[r] = (sub_demand.index, cost.to_numpy(), use.to_numpy())
    return _report_unbalance(per_region, error_threshold, warning_threshold)


def attribute_unbalance_to_final_consumers(regions, intermediate_dom, intermediate_imp, L, K, R, M, X,
                                            production_taxes, imp_intermediate_cons_tax,
                                            dom_intermediate_cons_tax, cons_taxes, total_demand):
    """
    Attribute each (region, sector)'s cost/use discrepancy to its final
    consumers -- household (C), government (G), investment (I) -- split
    across their Imp/Dom cells in `total_demand`, proportional to each
    cell's existing value. Sectors with zero combined C+G+I base are left
    unchanged -- there's no basis for a proportional split, and any
    residual unbalance for them will surface from
    `check_unbalance_final_format` after building the table.

    Mutates and returns `total_demand`.
    """
    demand_cols = [("imp", "C"), ("dom", "C"), ("imp", "G"), ("dom", "G"), ("imp", "I"), ("dom", "I")]
    for r in regions:
        cost, use, sub_demand = _region_cost_and_use(
            r, intermediate_dom, intermediate_imp, L, K, R, M, production_taxes,
            imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand, X)
        diff = (cost - use).to_numpy()

        base = sub_demand[demand_cols].to_numpy()
        row_sums = base.sum(axis=1)
        weights = np.divide(base, row_sums[:, None], out=np.zeros_like(base), where=row_sums[:, None] != 0)
        new_base = base + weights * diff[:, None]

        if (new_base < 0).any():
            warnings.warn(
                f"attribute_unbalance_to_final_consumers produced negative final-demand values in "
                f"region {r!r}: the per-sector unbalance exceeded that sector's existing C/G/I base "
                f"for at least one sector."
            )

        total_demand.loc[r, demand_cols] = new_base

    return total_demand


def check_unbalance_final_format(regional_IOTs_dict, len_sectors, error_threshold=0.01, warning_threshold=1e-4):
    """
    Check that each region's assembled IOT balances: for every sector, total
    output (row sum) must match total input (column sum). The unbalance is
    measured relative to the sector's own scale (mean of the row and column
    sum) so it is comparable across sectors of very different size.

    Run this after `build_regional_IOTs` (and ideally after
    `check_unbalance`/`attribute_unbalance_to_final_consumers` have already
    closed most of the gap upstream) as a final self-check before writing
    output. Same print/warn/raise contract as `check_unbalance`.
    """
    per_region = {}
    for r, df in regional_IOTs_dict.items():
        sum_rows = df.xs('∑', level='Subcategory').sum()[:len_sectors]
        sum_cols = df.xs('∑', level='Subcategory', axis=1).sum(axis=1)[:len_sectors]
        sector_labels = sum_rows.index.get_level_values('Subcategory')
        per_region[r] = (sector_labels, sum_rows.to_numpy().flatten(), sum_cols.to_numpy().flatten())
    return _report_unbalance(per_region, error_threshold, warning_threshold)


def build_regional_IOTs(regions, sectors, map_GTAP_cost_structure, map_GTAP_consumption_structure,
                         intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
                         imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand,
                         add_inventories):
    """Assemble the regional_IOT_format-shaped table (per region) from already-computed components.

    Dataset-agnostic: only cares about the shapes of its inputs, not how they were computed, so
    it's shared between reformat_EXIOBASE (least-squares tax reconciliation) and reformat_GLORIA
    (direct proportional tax allocation).
    """
    ########################
    #### create indexes ####
    ########################

    # Apply the transformation to your row and column index source DataFrames
    expanded_rows_indexes = expand_All_sectors(map_GTAP_cost_structure, sectors)  # For row MultiIndex
    expanded_col_indexes = expand_All_sectors(map_GTAP_consumption_structure.T, sectors)  # For column MultiIndex

    row_index = pd.MultiIndex.from_frame(expanded_rows_indexes.iloc[:, :2], names=[
                                         "Category", "Subcategory"])

    col_index = pd.MultiIndex.from_frame(expanded_col_indexes.iloc[:, :2], names=[
                                         "Category", "Subcategory"])



    # Inizializza tutto a NaN
    arr = np.full((len(row_index) , len(col_index) ), np.nan)

    # Imposta bande di zeri
    arr[:len(sectors), :] = 0.0       # prime N righe
    arr[:, :len(sectors)] = 0.0       # prime N colonne

    regional_IOT_format = pd.DataFrame(arr, index=row_index, columns=col_index, dtype=np.float64)

    ############################################
    ##### fill in the reformatted database #####
    ############################################



    df_dict = {}  # empty dictionary
    max_unbalance = 0
    for r in regions:
        # Create a random DataFrame for each r
        df_dict[r] = regional_IOT_format.copy()

        col_start = 0
        col_end = len(sectors)
        row_start = 0
        row_end = len(sectors)

        ####### fill in the rows #######

        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, intermediate_imp[r], col_start, col_end)
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, intermediate_imp[r].sum(axis=0), col_start, col_end)
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, intermediate_dom[r], col_start, col_end)
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, intermediate_dom[r].sum(axis=0), col_start, col_end)
        sum_sum = intermediate_imp[r].sum(
            axis=0) + intermediate_dom[r].sum(axis=0)
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, sum_sum, col_start, col_end)

        # VA
        row_start += 1
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, L[r], col_start, col_end)
        row_start += 4
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, pd.concat([K, R]).sum(axis=0)[r], col_start, col_end)
        row_start += 1
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, pd.concat([L, K, R]).sum(axis=0)[r], col_start, col_end)

        # M
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, M[r], col_start, col_end)
        row_start += 1
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, M[r], col_start, col_end)

        # prod taxes
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, production_taxes[r], col_start, col_end)
        row_start += 8
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, production_taxes[r], col_start, col_end)

        # cons taxes
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, imp_intermediate_cons_tax.loc[r], col_start, col_end)
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, imp_intermediate_cons_tax.loc[r].sum(axis=0), col_start, col_end)
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, dom_intermediate_cons_tax.loc[r], col_start, col_end)
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, dom_intermediate_cons_tax.loc[r].sum(axis=0), col_start, col_end)
        row_start = fill_reformat_df_row_wise(df_dict[r], row_start, pd.concat(
            [imp_intermediate_cons_tax.loc[r], dom_intermediate_cons_tax.loc[r]]).sum(axis=0), col_start, col_end)

        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, cons_taxes["imp"]["I"][r], col_start, col_end)
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, cons_taxes["dom"]["I"][r], col_start, col_end)
        row_start = fill_reformat_df_row_wise(df_dict[r], row_start, pd.concat(
            [cons_taxes["imp"]["I"][r], cons_taxes["dom"]["I"][r]]).sum(axis=0), col_start, col_end)

        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, cons_taxes["imp"]["C"][r], col_start, col_end)
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, cons_taxes["dom"]["C"][r], col_start, col_end)
        row_start = fill_reformat_df_row_wise(df_dict[r], row_start, pd.concat(
            [cons_taxes["imp"]["C"][r], cons_taxes["dom"]["C"][r]]).sum(axis=0), col_start, col_end)

        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, cons_taxes["imp"]["G"][r], col_start, col_end)
        row_start = fill_reformat_df_row_wise(
            df_dict[r], row_start, cons_taxes["dom"]["G"][r], col_start, col_end)
        row_start = fill_reformat_df_row_wise(df_dict[r], row_start, pd.concat(
            [cons_taxes["imp"]["G"][r], cons_taxes["dom"]["G"][r]]).sum(axis=0), col_start, col_end)

        row_start += 4
        #total TLSP
        row_start = fill_reformat_df_row_wise(df_dict[r], row_start, pd.concat(
            [imp_intermediate_cons_tax.loc[r], dom_intermediate_cons_tax.loc[r],
            cons_taxes["imp"]["G"][r], cons_taxes["dom"]["G"][r],
            cons_taxes["imp"]["C"][r], cons_taxes["dom"]["C"][r],
            cons_taxes["imp"]["I"][r], cons_taxes["dom"]["I"][r]]
            ).sum(axis=0), col_start, col_end)

        # sum

        sum_rows = df_dict[r].xs('∑', level='Subcategory').sum()[:col_end]
        fill_reformat_df_row_wise(
            df_dict[r], row_start, sum_rows, col_start, col_end)

        ####### fill in the columns #######
        col_start = len(sectors)
        row_start = 0
        row_end = len(sectors)

        # intermediate demand
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, intermediate_imp[r].sum(axis=1), row_start, row_end)

        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, intermediate_dom[r], row_start, row_end)
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, intermediate_dom[r].sum(axis=1), row_start, row_end)

        col_start = fill_reformat_df_columnwise(df_dict[r], col_start, pd.concat(
            [intermediate_imp[r], intermediate_dom[r]], axis=1).sum(axis=1), row_start, row_end)

        # C
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, total_demand.loc[r, ("imp", "C")], row_start, row_end)
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, total_demand.loc[r, ("dom", "C")], row_start, row_end)
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, total_demand.loc[r, ("imp", "C")] + total_demand.loc[r, ("dom", "C")], row_start, row_end)

        # G
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, total_demand.loc[r, ("imp", "G")], row_start, row_end)
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, total_demand.loc[r, ("dom", "G")], row_start, row_end)
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, total_demand.loc[r, ("imp", "G")] + total_demand.loc[r, ("dom", "G")], row_start, row_end)

        # I
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, total_demand.loc[r, ("imp", "I")], row_start, row_end)
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, total_demand.loc[r, ("dom", "I")], row_start, row_end)
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, total_demand.loc[r, ("imp", "I")] + total_demand.loc[r, ("dom", "I")], row_start, row_end)
        #DS
        if add_inventories:
            col_start = fill_reformat_df_columnwise(
                df_dict[r], col_start, total_demand.loc[r, ("imp", "DS")], row_start, row_end)
            col_start = fill_reformat_df_columnwise(
                df_dict[r], col_start, total_demand.loc[r, ("dom", "DS")], row_start, row_end)
            col_start = fill_reformat_df_columnwise(
                df_dict[r], col_start, total_demand.loc[r, ("imp", "DS")] + total_demand.loc[r, ("dom", "DS")], row_start, row_end)

        # X
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, X.loc[r], row_start, row_end)
        col_start += 1
        col_start = fill_reformat_df_columnwise(
            df_dict[r], col_start, X.loc[r], row_start, row_end)
        # sum
        col_start += 1
        sum_col = df_dict[r].xs('∑', level='Subcategory',
                                axis=1).sum(axis=1)[:row_end]
        fill_reformat_df_columnwise(
            df_dict[r], col_start, sum_col, row_start, row_end)

    return df_dict


def write_regional_IOTs(df_dict, reformat_folder):
    for r, df in df_dict.items():
        df = df.copy()

        # Rimuovi i nomi dei livelli (non i valori) se necessario
        df.index.names = [None] * df.index.nlevels

        # Pulizia del MultiIndex delle colonne: sostituisci i NaN con stringhe vuote
        if isinstance(df.columns, pd.MultiIndex):
            col_df = df.columns.to_frame(index=False).fillna('')
            df.columns = pd.MultiIndex.from_frame(col_df)
        else:
            df.columns = df.columns.to_series().fillna('')

        # Salva con na_rep='' per valori (dati)
        df.to_csv(reformat_folder + "/" + r + ".csv", na_rep='', encoding='utf-8-sig')

    print("Reformatted tables available at " + reformat_folder)


def _load_config_and_mapping(config_file, add_inventories):
    with pkg_resources.open_text(mappings, config_file) as f:
        io_config = json.load(f)

    key = "inventories" if add_inventories else "standard"
    mapping_config = io_config["mapping_files"][key]

    with pkg_resources.open_text(mappings, mapping_config["reformat_file"]) as f:
        map_final_demand = pd.read_csv(f)

    with pkg_resources.open_binary(mappings, mapping_config["gtap_file"]) as f:
        map_GTAP_cost_structure = pd.read_excel(f, sheet_name=mapping_config["gtap_sheet_cost"], header=None)
        f.seek(0)
        map_GTAP_consumption_structure = pd.read_excel(f, sheet_name=mapping_config["gtap_sheet_cons"], header=None)

    return io_config, map_final_demand, map_GTAP_cost_structure, map_GTAP_consumption_structure


def _load_FZY(aggregation_folder, io_config, sectors_order):
    input_files = io_config["input_files"]
    F = pd.read_csv(f"{aggregation_folder}/{input_files['factor_inputs_subfolder']}/{input_files['F']}", delimiter="\t",
                    header=[0, 1], index_col=0)  # Factors of productions/stressors/impacts
    Z = pd.read_csv(f"{aggregation_folder}/{input_files['Z']}", delimiter="\t",
                    header=[0, 1], index_col=[0, 1])  # flow/transactions matrix
    Y = pd.read_csv(f"{aggregation_folder}/{input_files['Y']}", delimiter="\t",
                    header=[0, 1], index_col=[0, 1])  # final demand

    regions = F.columns.get_level_values(0).unique()
    sectors = F.columns.get_level_values(1).unique() if sectors_order == [] else sectors_order

    F = reorder_io_columns(F, sectors)
    Y = reorder_io_rows(Y, sectors)
    Z = reorder_io_matrix(Z, sectors)

    return F, Z, Y, regions, sectors


def _value_added_LKR(F, map_final_demand):
    L_raw = F.loc[EXIOBASE_name("L", map_final_demand)].sum(axis=0)
    K_raw = F.loc[EXIOBASE_name("K", map_final_demand)].sum(axis=0)
    R_raw = F.loc[EXIOBASE_name("R", map_final_demand)].sum(axis=0)

    L = L_raw.to_frame().T.rename(index={0: 'L'})
    K = K_raw.to_frame().T.rename(index={0: 'K'})
    R = R_raw.to_frame().T.rename(index={0: 'R'})
    return L, K, R
