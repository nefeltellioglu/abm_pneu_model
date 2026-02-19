#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:04:26 2023

@author: ntellioglu
"""
import polars as pl
from scipy.stats import logistic
import numpy as np

def is_sublist(sublist, mainlist):
    return (sublist in 
            [mainlist[i:i+len(sublist)] for i in 
             range(len(mainlist) - len(sublist) + 1)])

def create_vaccine_antibody_df(vaccines, strains):
    
    entrybirthcohort_vacc = None
    earlycohort_vacc = None
    vaccine_antibody_df = pl.read_csv(
        "src/pneu_abm/data/immunity/vaccine_induced_antibody_levels.csv", 
        comment_prefix="#")
    vaccine_antibody_df = vaccine_antibody_df.sort(vaccine_antibody_df.columns)
    for vaccine, value in vaccines.items():
        if vaccine.endswith("catchup"):
            doses = list(range(
                len(vaccines[value["previous_vacc"]]["daily_schedule"]) + 1, 
                     len(value["daily_schedule"]) + \
                len(vaccines[value["previous_vacc"]]["daily_schedule"]) + 1))
            
        else:
            doses =  list(range(0, len(value["daily_schedule"]) + 1))
        if vaccine.endswith(("ppsv23_2")):
            continue
        if vaccine.endswith(("ppsv23_1")):
            vaccine = vaccine.strip("_1")
        vaccine_name_end = vaccine.split("_")[-1]   
        existing_ends = ["child", "30", "21", "31", "adult", "1", "2",
                         "catchup"] 
        
        if vaccine_name_end not in existing_ends:
            #print(f"{vaccine_name_end} not in {existing_ends}")
            #print(f"checking if {vaccine} in vaccine_antibody_df")
            vaccine_name = vaccine.split(f"_{vaccine_name_end}")[0]
            
            if (vaccine_name in list(vaccines.keys())):
                continue
            else:
                vaccine = vaccine.split(f"_{vaccine_name_end}")[0]
        cur_vac_antibody_levels = (vaccine_antibody_df
                              .filter((pl.col("vaccine_type") == vaccine)))
        assert cur_vac_antibody_levels.height, \
        f"{vaccine} not in {vaccine_antibody_df['vaccine_type'].unique()}"
        cur_no_doses = cur_vac_antibody_levels["no_of_doses"].unique().to_list() 
        assert is_sublist(doses, cur_no_doses), \
        f"{doses, vaccine} not in {cur_no_doses}"
        
    vaccine_antibody_df = vaccine_antibody_df.unique()

    return vaccine_antibody_df, existing_ends

    
def waning_ratio(vaccine_type, day, final_vaccine_time,
                 waning_halflife_day_adult, waning_halflife_day_child,age):
    """
    Parameters
    ----------
    vaccine_type = possibly_infected["vaccines"].struct.field("vaccine_type")
    day = day
    final_vaccine_time = (possibly_infected["vaccines"]
                          .struct.field("final_vaccine_time"))
    waning_halflife_day_adult = self.waning_halflife_day_adult
  
    waning_halflife_day_child = self.waning_halflife_day_child
    age = possibly_infected["age"]
    

    """
       
    halflife = pl.DataFrame([
        vaccine_type,
        final_vaccine_time, age])
    halflife = halflife.with_columns(
        
        pl.when(
            pl.col("age") >= 18)
        .then(np.log(0.5)/waning_halflife_day_adult)
        .otherwise(np.log(0.5)/waning_halflife_day_child).alias("daily_decay")
        ).with_columns(
            np.exp(pl.col("daily_decay") * (day - pl.col("final_vaccine_time"))).alias("waning_ratio")
            )
            
    return halflife["waning_ratio"]
            
if __name__ == '__main__':
    import sys,os

    repo_path = ".."
    repo_path = os.path.abspath(os.path.join(repo_path))
    if repo_path not in sys.path:
        sys.path.append(repo_path)
    os.chdir(os.path.join(repo_path))
