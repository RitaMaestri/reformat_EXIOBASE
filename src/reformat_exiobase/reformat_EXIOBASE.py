"""
Created on Thu Mar 13 14:20:45 2025

@author: rita
"""

import pandas as pd
import numpy as np
from . import mappings
import importlib.resources as pkg_resources
from scipy.optimize import least_squares


########################################
########## AGGREGATING DATA ############
########################################


def EXIOBASE_name(SCAF_name, map_final_demand):
    return map_final_demand.loc[map_final_demand['SCAF'] == SCAF_name, 'EXIOBASE_name'].values

def exempt_from_taxes(map_final_demand):
    return map_final_demand.loc[map_final_demand['exempt_cons_taxes'], 'EXIOBASE_name'].values

def final_demand_agents(map_final_demand):
    return map_final_demand["SCAF"].loc[map_final_demand["EXIOBASE_file"] == "Y"].unique()


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



def disaggregate_tax(tax_shares,Z,Y, map_final_demand):

    tot_dem = pd.concat([Z, Y], axis=1)
    tot_dem.columns = tot_dem.columns.set_names('category', level=1)


    taxes_df = pd.DataFrame(0, index=tot_dem.index, columns=tot_dem.columns, dtype=np.float64)

    is_exempt_from_taxes = exempt_from_taxes(map_final_demand)

    row_sector_labels = taxes_df.index.get_level_values('sector')
    col_region_labels = taxes_df.columns.get_level_values('region')

    for (r_ts, s_ts), tax_val in tax_shares.items():

        row_mask = (row_sector_labels == s_ts)

        col_mask = (col_region_labels == r_ts) & \
        (~taxes_df.columns.get_level_values(1).isin(is_exempt_from_taxes))

        taxes_df.loc[row_mask, col_mask] = tax_val

    tax_df = taxes_df * tot_dem

    return tax_df


def adjust_tax_shares(Z: pd.DataFrame, Y: pd.DataFrame, F: pd.DataFrame, map_final_demand) -> pd.Series:

    #the taxes on consumption in the file F.txt for region R and sector S include 
    #taxes that are paid abroad for the consumption of the good S produced in R.
    # we extract: 
    # -import and export net of taxes.
    # -the tax that is paid on good S consumed in region R and of all origin.
    # the resulting national account is balanced

    def reallocate_tax(tax_values, tax_index, Z, Y, F, map_final_demand):
        tax_shares = pd.Series(tax_values, index=tax_index)
        tax_df = disaggregate_tax(tax_shares, Z, Y,map_final_demand)
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


def check_unbalance(regional_IOTs_dict, len_sectors):
    max_unbalance = 0
    unbalance_by_region = {}

    for r, df in regional_IOTs_dict.items():
        sum_rows = df.xs('∑', level='Subcategory').sum()[:len_sectors]
        sum_cols = df.xs('∑', level='Subcategory', axis=1).sum(axis=1)[:len_sectors]
        
        unbalance_vector = abs(sum_cols.to_numpy().flatten() -
                            sum_rows.to_numpy().flatten())

        max_unbalance_r = unbalance_vector.max()

        unbalance_by_region[r] = max_unbalance_r
        if max_unbalance_r > max_unbalance:
            max_unbalance = max_unbalance_r

    print("Max unbalance across regions:", max_unbalance)
    return unbalance_by_region


##########################################
##########################################
################ REFORMAT ################
##########################################
##########################################


def reformat_EXIOBASE(aggregation_folder, reformat_folder, sectors_order=[], add_inventories = True):

    ###########################
    #### IMPORT DATABASES #####
    ###########################
    F = pd.read_csv(f"{aggregation_folder}/factor_inputs/F.txt", delimiter="\t",
                    header=[0, 1], index_col=0)  # Factors of productions/stressors/impacts
    Z = pd.read_csv(f"{aggregation_folder}/Z.txt", delimiter="\t",
                    header=[0, 1], index_col=[0, 1])  # flow/transactions matrix
    Y = pd.read_csv(f"{aggregation_folder}/Y.txt", delimiter="\t",
                    header=[0, 1], index_col=[0, 1])  # final demand

    

    file_map = {
        "standard": {
            "reformat_file": "map_reformat_EXIOBASE.csv",
            "gtap_file": "map_GTAP_format.xlsx",
            "gtap_sheet_cost": "Cost structure",
            "gtap_sheet_cons": "Consumption structure"
        },
        "inventories": {
            "reformat_file": "map_reformat_EXIOBASE_inventories.csv",
            "gtap_file": "map_GTAP_format.xlsx",  # stesso file Excel
            "gtap_sheet_cost": "Cost structure",
            "gtap_sheet_cons": "Cons structure inventories"
        }
    }

    key = "inventories" if add_inventories else "standard"
    config = file_map[key]

    with pkg_resources.open_text(mappings, config["reformat_file"]) as f:
        map_final_demand = pd.read_csv(f)

    with pkg_resources.open_binary(mappings, config["gtap_file"]) as f:
        map_GTAP_cost_structure = pd.read_excel(f, sheet_name=config["gtap_sheet_cost"], header=None)
        f.seek(0)
        map_GTAP_consumption_structure = pd.read_excel(f, sheet_name=config["gtap_sheet_cons"], header=None)

    ###############################
    #### CHANGE SECTORS ORDER #####
    ###############################

    regions = F.columns.get_level_values(0).unique()

    if sectors_order == []:
        sectors = F.columns.get_level_values(1).unique()
    else:
        sectors = sectors_order


    def reorder_io_columns(df, desired_sector_order):
        order_map = {sector: i for i,
                     sector in enumerate(desired_sector_order)}
        new_cols = sorted(
            df.columns,
            key=lambda x: (x[0], order_map.get(x[1], float('inf')))
        )
        return df[new_cols]

    def reorder_io_rows(df, desired_sector_order):
        order_map = {sector: i for i,
                     sector in enumerate(desired_sector_order)}
        new_idx = sorted(
            df.index,
            key=lambda x: (x[0], order_map.get(x[1], float('inf')))
        )
        return df.loc[new_idx]

    def reorder_io_matrix(df, desired_sector_order):
        df = reorder_io_rows(df, desired_sector_order)
        df = reorder_io_columns(df, desired_sector_order)
        return df

    F = reorder_io_columns(F, sectors)
    Y = reorder_io_rows(Y, sectors)
    Z = reorder_io_matrix(Z, sectors)


    
    #####################################
    ### INTERMEDIATE AND FINAL DEMAND ###
    #####################################

    intermediate_dom = compute_intermediate_domestic_demand(Z)

    intermediate_imp = compute_intermediate_imports(Z)

    fd_dom=compute_final_demand_domestic(Y, map_final_demand)

    fd_imp = compute_final_demand_imported(Y, map_final_demand)

    total_demand = concatenate_total_demand(fd_dom, fd_imp, intermediate_dom, intermediate_imp)

    ###################################################
    ##### REALLOCATION OF TAXES ON CONSUMPTION ########
    ###################################################

   
    tax_shares = adjust_tax_shares(Z, Y, F, map_final_demand)

    taxes_df = disaggregate_tax(tax_shares, Z, Y, map_final_demand)

    net_flows = pd.concat([Z, Y], axis=1) - taxes_df

    ############################################
    ##### IMPORT AND EXPORT NET OF TAXES #######
    ############################################

    M = compute_imports(net_flows)

    X = compute_exports(net_flows)

    ##################################################
    ##### ALLOCATE CONSUMPTION TAXES TO CONSUMERS ####
    ##################################################
    
    imp_intermediate_cons_tax = compute_intermediate_imports(taxes_df[Z.columns]).T
    dom_intermediate_cons_tax = compute_intermediate_domestic_demand(taxes_df[Z.columns]).T
    
    fd_taxes_imp = compute_final_demand_imported(taxes_df[Y.columns], map_final_demand)
    fd_taxes_dom = compute_final_demand_domestic(taxes_df[Y.columns], map_final_demand)
    
    cons_taxes = {"imp": {}, "dom": {}}

    for agent in final_demand_agents(map_final_demand):
        cons_taxes["imp"][agent] = fd_taxes_imp[agent].to_frame().T
        cons_taxes["dom"][agent] = fd_taxes_dom[agent].to_frame().T


    ################################
    ##### TAXES ON PRODUCTION ######
    ################################

    production_taxes = F.loc[EXIOBASE_name("Production_taxes", map_final_demand)]

    ################################
    ######### VALUE ADDED ##########
    ################################

    L_raw = F.loc[EXIOBASE_name("L", map_final_demand)].sum(axis=0)
    K_raw = F.loc[EXIOBASE_name("K", map_final_demand)].sum(axis=0)
    R_raw = F.loc[EXIOBASE_name("R", map_final_demand)].sum(axis=0)

    L = L_raw.to_frame().T.rename(index={0: 'L'})
    K = K_raw.to_frame().T.rename(index={0: 'K'})
    R = R_raw.to_frame().T.rename(index={0: 'R'})

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

    ##########################
    ### verify equilibrium ###
    ##########################
    
    unbalance_by_region = check_unbalance(df_dict, len(sectors))

    ####################
    ###### TO CSV ######
    ####################

    for r in regions:
        df = df_dict[r].copy()

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
