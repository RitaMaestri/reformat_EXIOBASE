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

    io_vec_agg = (
        gloria
        .calc_all()
        .aggregate(region_agg=region_aggregation_vector, sector_agg=sector_aggregation_vector, inplace=True)
    )

    ##### EXPORT AGGREGATED MRIO IN EXIOBASE FORMAT #####

    io_vec_agg.save_all(path=output_path)
    print(f"File saved at {output_path}.")
