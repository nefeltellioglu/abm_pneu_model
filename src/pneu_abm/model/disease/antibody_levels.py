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
                 waning_halflife_day_adult, waning_halflife_day_child, age):
    """
    Return the multiplicative waning factor applied to vaccine-induced
    antibody levels.

    ``waning_halflife_day_adult`` and ``waning_halflife_day_child`` are
    model parameters specifying the decay halflife used for adults (age >= 18)
    and children respectively.  The formula matches the original Fortran model
    and results in a value between 0 and 1 when halflife inputs are positive.

    This implementation prefers :mod:`numpy` over :mod:`polars` so that it can
    gracefully handle a variety of input types (scalars, NumPy arrays,
    Polars/Spark-like series, etc.).  All inputs are converted to NumPy arrays
    using ``np.asarray``; the function always returns a NumPy array.  Callers
    in the simulation pipeline expect an array-like result and perform further
    NumPy arithmetic on the output.
    """
    # coerce inputs to numpy arrays; np.asarray works with scalars, lists,
    # Polars Series, etc.
    vaccine_type = np.asarray(vaccine_type)
    day = np.asarray(day)
    final_vaccine_time = np.asarray(final_vaccine_time)
    age = np.asarray(age)

    # compute age-dependent decay rate
    daily_decay = np.where(
        age >= 18,
        np.log(0.5) / waning_halflife_day_adult,
        np.log(0.5) / waning_halflife_day_child,
    )

    waning = np.exp(daily_decay * (day - final_vaccine_time))
    return waning
            
if __name__ == '__main__':
    import sys,os

    repo_path = ".."
    repo_path = os.path.abspath(os.path.join(repo_path))
    if repo_path not in sys.path:
        sys.path.append(repo_path)
    os.chdir(os.path.join(repo_path))
