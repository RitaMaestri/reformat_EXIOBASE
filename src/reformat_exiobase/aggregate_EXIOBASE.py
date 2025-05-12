"""
Created on Thu Mar 13 14:20:45 2025

@author: rita
"""

import pymrio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal


def EXIOBASE_file_name(input_path, year: int, system: Literal["pxp", "ixi"]):
    return Path(input_path+"/IOT_"+str(year)+"_"+system+".zip")
    

def aggregate_EXIOBASE(reg_map_path, sec_map_path, output_path, input_path, year: int, system: Literal["pxp", "ixi"]):
    
    input_file=EXIOBASE_file_name(input_path, year, system)
    
    sec_map=pd.read_csv(sec_map_path)
    reg_map=pd.read_csv(reg_map_path)
    
    print("Parsing EXIOBASE...")
    
    exio3 = pymrio.parse_exiobase3( path=input_file )
    
    print("Parsing complete.")
    
    sec_col = pd.DataFrame(exio3.get_sectors().tolist(), columns=['EXIOBASE sector'])
    merged_df_sec = pd.merge(sec_col, sec_map, on='EXIOBASE sector', how='right')

    reg_col = pd.DataFrame(exio3.get_regions().tolist(), columns=['EXIOBASE region'])
    merged_df_reg = pd.merge(reg_col, reg_map, on='EXIOBASE region', how='right')

    sector_aggregation_vector = merged_df_sec['SCAF sector'].tolist()
    region_aggregation_vector = merged_df_reg['SCAF region'].tolist()
    
    print("Aggregating EXIOBASE...")
    
    io_vec_agg = (
        exio3
        .calc_all()
        .aggregate(region_agg=region_aggregation_vector, sector_agg=sector_aggregation_vector, inplace=True)
    )
    
    ##### EXPORT AGGREGATED MRIO IN EXIOBASE FORMAT #####

    io_vec_agg.save_all(path=output_path)
    print(f"File saved at {output_path}.")



###### USE EXAMPLE ########

# current_file = Path(__file__).resolve()
# repo_root = str(current_file.parent.parent.parent.parent)+"/"

# #Import mappings
# mapping_path = repo_root + "Data/Mappings/aggregate EXIOBASE/"
# sec_map = pd.read_csv(mapping_path + "map_sector.csv")
# reg_map = pd.read_csv(mapping_path + "map_regions.csv")

# sec = np.unique(sec_map["SECTOR SCAF"], return_index=False)
# reg = np.unique(reg_map["Agglomeration"], return_index=False)

# #Define output file
# out_path = repo_root + "Data/Preprocessed/aggregate EXIOBASE/"
# output_file = out_path + str(len(sec)) + " sectors " + str(len(reg)) + " regions"


# #Define input file
# exio_path= repo_root + "Data/Raw Data/MRIOs/EXIOBASE/"
# exio_file = exio_path + "IOT_2020_pxp.zip"

# aggregate_EXIOBASE(reg_map, sec_map, output_file, exio_file)


