#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:42:38 2025

@author: rita
"""
import pymrio

#download EXIOBASE
def download_EXIOBASE(EXIOBASE_folder,system, years, version, overwrite_existing=False):
    print("Downloading EXIOBASE...")
    pymrio.download_exiobase3(EXIOBASE_folder, system=system, years=years, doi=version, overwrite_existing=overwrite_existing)
    
        
        



