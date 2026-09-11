"""
Created on Thu Mar 13 14:20:45 2025

@author: rita
"""

import pandas as pd
import numpy as np

from .reformat_lib import (
    EXIOBASE_name,
    exempt_from_taxes,
    final_demand_agents,
    reallocate_G_I_energy_to_C,
    compute_intermediate_domestic_demand,
    compute_intermediate_imports,
    compute_final_demand_domestic,
    compute_final_demand_imported,
    concatenate_total_demand,
    disaggregate_tax,
    adjust_tax_rates,
    compute_imports,
    compute_exports,
    zero_out_regional_noise,
    check_unbalance,
    attribute_unbalance_to_final_consumers,
    build_regional_IOTs,
    check_unbalance_final_format,
    write_regional_IOTs,
    _load_config_and_mapping,
    _load_FZY,
    _value_added_LKR,
)


##########################################
##########################################
################ REFORMAT ################
##########################################
##########################################


def reformat_EXIOBASE(aggregation_folder, reformat_folder, energy_sectors=None, sectors_order=[], add_inventories=True):

    ###########################
    #### IMPORT DATABASES #####
    ###########################
    io_config, map_final_demand, map_GTAP_cost_structure, map_GTAP_consumption_structure = \
        _load_config_and_mapping("config_EXIOBASE.json", add_inventories)

    F, Z, Y, regions, sectors = _load_FZY(aggregation_folder, io_config, sectors_order)

    if energy_sectors is not None:
        Y = reallocate_G_I_energy_to_C(Y, energy_sectors)

    #####################################
    ### INTERMEDIATE AND FINAL DEMAND ###
    #####################################

    intermediate_dom = compute_intermediate_domestic_demand(Z)

    intermediate_imp = compute_intermediate_imports(Z)

    fd_dom = compute_final_demand_domestic(Y, map_final_demand)

    fd_imp = compute_final_demand_imported(Y, map_final_demand)

    total_demand = concatenate_total_demand(fd_dom, fd_imp, intermediate_dom, intermediate_imp)

    ###################################################
    ##### REALLOCATION OF TAXES ON CONSUMPTION ########
    ###################################################

   #there is a unique tax rate paid by all consumers per product purchased per region
    tax_rates = adjust_tax_rates(Z, Y, F, map_final_demand)

    tax_rates_df = disaggregate_tax(tax_rates, Z, Y, map_final_demand)

    net_flows = pd.concat([Z, Y], axis=1) - tax_rates_df

    ############################################
    ##### IMPORT AND EXPORT NET OF TAXES #######
    ############################################

    M = compute_imports(net_flows)

    X = compute_exports(net_flows)

    ##################################################
    ##### ALLOCATE CONSUMPTION TAXES TO CONSUMERS ####
    ##################################################

    imp_intermediate_cons_tax = compute_intermediate_imports(tax_rates_df[Z.columns]).T
    dom_intermediate_cons_tax = compute_intermediate_domestic_demand(tax_rates_df[Z.columns]).T

    fd_taxes_imp = compute_final_demand_imported(tax_rates_df[Y.columns], map_final_demand)
    fd_taxes_dom = compute_final_demand_domestic(tax_rates_df[Y.columns], map_final_demand)

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

    L, K, R = _value_added_LKR(F, map_final_demand)

    ##########################
    ### verify equilibrium ###
    ##########################

    (intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
     imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand) = zero_out_regional_noise(
        intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
        imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand)

    check_unbalance(regions, intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
                     imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand)

    total_demand = attribute_unbalance_to_final_consumers(
        regions, intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
        imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand)

    df_dict = build_regional_IOTs(regions, sectors, map_GTAP_cost_structure, map_GTAP_consumption_structure,
                                   intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
                                   imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand,
                                   add_inventories)

    check_unbalance_final_format(df_dict, len(sectors))

    write_regional_IOTs(df_dict, reformat_folder)


def reformat_GLORIA(aggregation_folder, reformat_folder, sectors_order=[], add_inventories=False):
    """Reformat a GLORIA aggregation into the SCAF/GTAP-style regional tables.

    Differs from reformat_EXIOBASE in two ways, both because GLORIA's Z/Y (the
    "Basic prices" files) are already a self-consistent basic-price system on
    their own -- confirmed against GLORIA's own National Accounting Identity
    (Release Notes, "Note II") and empirically against aggregate_GLORIA's output:

    - No energy reallocation (reallocate_G_I_energy_to_C hardcodes EXIOBASE-only
      Y category strings and isn't meaningful for GLORIA's sector scheme).
    - No adjust_tax_rates/disaggregate_tax least-squares reconciliation. GLORIA's
      "Consumption_taxes" F-row (derived from the Markup004/005 tax/subsidy
      files at parse time) is the basic-price-to-purchaser-price wedge, not a
      reconciliation target the way EXIOBASE's own row is -- fitting it via
      adjust_tax_rates measurably worsens the resulting table's balance.
      Instead, it's allocated directly (a closed-form ad-valorem rate per
      (region, sector), proportional to each final-demand agent's existing
      basic-price share) and added symmetrically to both the demand side
      (C/G/I) and the matching cost-side "Tax" rows, which is balance-neutral
      by construction (adding the same amount to both sides of the identity
      doesn't change their difference) while producing purchaser-price C/G/I
      figures.
    """

    ###########################
    #### IMPORT DATABASES #####
    ###########################
    io_config, map_final_demand, map_GTAP_cost_structure, map_GTAP_consumption_structure = \
        _load_config_and_mapping("config_GLORIA.json", add_inventories)

    F, Z, Y, regions, sectors = _load_FZY(aggregation_folder, io_config, sectors_order)

    #####################################
    ### INTERMEDIATE AND FINAL DEMAND ###
    #####################################

    intermediate_dom = compute_intermediate_domestic_demand(Z)

    intermediate_imp = compute_intermediate_imports(Z)

    fd_dom = compute_final_demand_domestic(Y, map_final_demand)

    fd_imp = compute_final_demand_imported(Y, map_final_demand)

    ###################################################
    ##### DIRECT ALLOCATION OF SALES TAX ##############
    ###################################################

    net_sales_tax = F.loc[EXIOBASE_name("Consumption_taxes", map_final_demand)].iloc[0]

    exempt_names = set(exempt_from_taxes(map_final_demand))
    agents = final_demand_agents(map_final_demand)
    taxable_agents = [a for a in agents if not set(EXIOBASE_name(a, map_final_demand)).issubset(exempt_names)]

    taxable_base = sum((fd_dom[a] + fd_imp[a] for a in taxable_agents))
    tax_rate = (net_sales_tax / taxable_base).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    fd_dom_taxed = fd_dom.copy()
    fd_imp_taxed = fd_imp.copy()
    cons_taxes = {"imp": {}, "dom": {}}
    for agent in agents:
        if agent in taxable_agents:
            dom_tax = fd_dom[agent] * tax_rate
            imp_tax = fd_imp[agent] * tax_rate
            fd_dom_taxed[agent] = fd_dom[agent] + dom_tax
            fd_imp_taxed[agent] = fd_imp[agent] + imp_tax
        else:
            dom_tax = fd_dom[agent] * 0.0
            imp_tax = fd_imp[agent] * 0.0
        cons_taxes["dom"][agent] = dom_tax.to_frame().T
        cons_taxes["imp"][agent] = imp_tax.to_frame().T

    total_demand = concatenate_total_demand(fd_dom_taxed, fd_imp_taxed, intermediate_dom, intermediate_imp)

    # intermediate (Z) purchases stay at basic price -- tax is only added to
    # final-demand consumption agents, per the above -- so these template rows are zero.
    imp_intermediate_cons_tax = compute_intermediate_imports(Z).T * 0.0
    dom_intermediate_cons_tax = compute_intermediate_domestic_demand(Z).T * 0.0

    ############################################
    ##### IMPORT AND EXPORT #####################
    ############################################

    # No de-taxing needed: Z/Y are basic price already, nothing to strip out.
    flows = pd.concat([Z, Y], axis=1)

    M = compute_imports(flows)

    X = compute_exports(flows)

    ################################
    ##### TAXES ON PRODUCTION ######
    ################################

    production_taxes = F.loc[EXIOBASE_name("Production_taxes", map_final_demand)]

    ################################
    ######### VALUE ADDED ##########
    ################################

    L, K, R = _value_added_LKR(F, map_final_demand)

    ##########################
    ### verify equilibrium ###
    ##########################

    (intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
     imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand) = zero_out_regional_noise(
        intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
        imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand)

    check_unbalance(regions, intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
                     imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand)

    total_demand = attribute_unbalance_to_final_consumers(
        regions, intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
        imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand)

    df_dict = build_regional_IOTs(regions, sectors, map_GTAP_cost_structure, map_GTAP_consumption_structure,
                                   intermediate_dom, intermediate_imp, L, K, R, M, X, production_taxes,
                                   imp_intermediate_cons_tax, dom_intermediate_cons_tax, cons_taxes, total_demand,
                                   add_inventories)

    check_unbalance_final_format(df_dict, len(sectors))

    write_regional_IOTs(df_dict, reformat_folder)
