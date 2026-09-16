"""
Created on Sun Aug 31 2026

@author: rita
"""

import pandas as pd
from .parse_GLORIA import parse_gloria_lowmem


def aggregate_GLORIA(reg_map_path, sec_map_path, output_path, input_path, year):

    sec_map = pd.read_csv(sec_map_path)
    reg_map = pd.read_csv(reg_map_path)

    print("Parsing GLORIA...")

    gloria = parse_gloria_lowmem(path=input_path, year=year)

    print("Parsing complete.")

    sec_col = pd.DataFrame(gloria.get_sectors().tolist(), columns=['GLORIA sector'])
    merged_df_sec = pd.merge(sec_col, sec_map, on='GLORIA sector', how='right')

    # left (not right, unlike the sector merge / aggregate_EXIOBASE's pattern):
    # parse_gloria_lowmem can drop regions that come back empty for a given year
    # (e.g. DYE for 2020), so the raw-region template can list more regions than
    # actually end up in the parsed data. A left join anchored on gloria.get_regions()
    # ignores template rows for regions that aren't present, keeping the aggregation
    # vector the correct length; a right join would inflate it and break .aggregate().
    reg_col = pd.DataFrame(gloria.get_regions().tolist(), columns=['GLORIA region'])
    merged_df_reg = pd.merge(reg_col, reg_map, on='GLORIA region', how='left')

    sector_aggregation_vector = merged_df_sec['SCAF sector'].tolist()
    region_aggregation_vector = merged_df_reg['SCAF region'].tolist()

    print("Aggregating GLORIA...")

    gloria.calc_all()

    # gloria.aggregate() combines rows/columns by position, not by re-matching
    # labels -- it applies the same region/sector concordance matrix to every
    # DataFrame it finds via positional matrix multiplication (conc @ df @
    # conc.T), the same way it aggregates Z itself. tax_on_intermediate/
    # tax_on_final_demand were built earlier in parse_gloria_lowmem from the
    # same region/sector labels as Z/Y, but not necessarily in the same row/
    # column *order* -- this label-based reindex forces that exact order so
    # aggregate()'s positional math lines each row/column up with the right
    # (region, sector) pair. Any NaN after reindexing would mean our tax data
    # doesn't actually cover the same (region, sector) labels as Z/Y.
    gloria.VA.tax_on_intermediate = gloria.VA.tax_on_intermediate.reindex(
        index=gloria.Z.index, columns=gloria.Z.columns
    )
    gloria.VA.tax_on_final_demand = gloria.VA.tax_on_final_demand.reindex(
        index=gloria.Z.index, columns=gloria.Y.columns
    )
    assert not gloria.VA.tax_on_intermediate.isna().any().any(), (
        "tax_on_intermediate has labels that don't match Z after reindexing"
    )
    assert not gloria.VA.tax_on_final_demand.isna().any().any(), (
        "tax_on_final_demand has labels that don't match Z/Y after reindexing"
    )

    io_vec_agg = gloria.aggregate(
        region_agg=region_aggregation_vector, sector_agg=sector_aggregation_vector, inplace=True
    )

    ##### EXPORT AGGREGATED MRIO IN EXIOBASE FORMAT #####

    io_vec_agg.save_all(path=output_path)
    print(f"File saved at {output_path}.")
