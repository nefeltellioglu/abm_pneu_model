#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:04:26 2023

@author: ntellioglu
"""
import polars as pl
from scipy.stats import logistic
import numpy as np

def create_vaccine_antibody_df(vaccines, strains):
    
    entrybirthcohort_vacc = None
    earlycohort_vacc = None
    vaccine_antibody_df = pl.DataFrame()
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
        
        if vaccine.endswith(("_entrybirthcohort")):
            entrybirthcohort_vacc = vaccine.split("_entrybirthcohort")[0]
            
            if (entrybirthcohort_vacc in list(vaccines.keys())):
                continue
            else:
                vaccine = vaccine.split("_entrybirthcohort")[0]
        if vaccine.endswith(("_earlycohort")):
            earlycohort_vacc = vaccine.split("_earlycohort")[0]
            
            if (earlycohort_vacc in list(vaccines.keys())):
                continue
            else:
                vaccine = vaccine.split("_earlycohort")[0]
        if vaccine.endswith(("_earlyentrycohort")):
            earlycohort_vacc = vaccine.split("_earlyentrycohort")[0]
            
            if (earlycohort_vacc in list(vaccines.keys())):
                continue
            else:
                vaccine = vaccine.split("_earlyentrycohort")[0]
        
        if vaccine in ["pcv13_31_final_dose", "pcv20_31_final_dose"]:
            vaccine = vaccine.strip("_final_dose")
            doses = [4]
            
        cur_vaccine_antibody_df = pl.DataFrame(
        [pl.Series("vaccine_type",
                   [vaccine] * len(strains) * \
                       len(doses)),
         pl.Series("no_of_doses", 
           doses * \
               len(strains)
               ).cast(pl.Int32),
         pl.Series("exposed_strains", 
        strains.to_list() * len(doses)),
        
             
             ])
        #print("vaccine %s %s"%(vaccine, cur_vaccine_antibody_df.height))
        if vaccine_antibody_df.height:
            vaccine_antibody_df = pl.concat([vaccine_antibody_df, 
                        cur_vaccine_antibody_df], rechunk = True)
        else:
            vaccine_antibody_df = cur_vaccine_antibody_df
    
    if entrybirthcohort_vacc and not (entrybirthcohort_vacc in 
                            vaccine_antibody_df["vaccine_type"].unique()):
        assert False, (f"{entrybirthcohort_vacc} not in vacc list " + \
                      f"{vaccine_antibody_df['vaccine_type'].unique()}")
    if earlycohort_vacc and not (earlycohort_vacc in 
                            vaccine_antibody_df["vaccine_type"].unique()):
        assert False, (f"{earlycohort_vacc} not in vacc list " + \
                      f"{vaccine_antibody_df['vaccine_type'].unique()}")
        
    non_vacc =  pl.DataFrame(
            [pl.Series("vaccine_type", [""] * len(strains) ),
             pl.Series("no_of_doses", [0] * len(strains) ).cast(pl.Int32),
             pl.Series("exposed_strains", strains.to_list() ),
            ])
    vaccine_antibody_df = pl.concat([vaccine_antibody_df, 
                non_vacc], rechunk = True)
    vaccine_antibody_df = add_antibody_levels(vaccine_antibody_df)
    vaccine_antibody_df = vaccine_antibody_df.unique()
    #vaccine_antibody_df.write_csv("output/vaccine_antibody_df.csv")
    
    return vaccine_antibody_df

def add_antibody_levels(vaccine_antibody_df):
    antibody_distributions = pl.read_csv(
        "data/immunity/antibody_distributions.csv")
    serotype_groups = pl.read_csv(
        "data/immunity/immune_group_list_20231010.csv")
    
    
    schedule_groups = pl.DataFrame([
        pl.Series("age", ["Elderly"] * 2),
        pl.Series("schedule", ["1-dose", "2-dose (60-64Y)"]),
        ])
    antibody_distributions = antibody_distributions.join(schedule_groups, 
            on = "schedule", how = "left").with_columns(
                pl.col("age").fill_null(
                    value = "Infant_toddler")).drop("ratio", 
                                                    "sd_square", "mean")
    antibody_distributions = antibody_distributions.with_columns(
       (pl.when((pl.col("age") == "Elderly") & \
               (pl.col("vaccine_type") == "PCV13"))
       .then("pcv13_adult")
       .otherwise(pl.col("vaccine_type")).alias("vaccine_type"))) 
    
    serotype_groups = serotype_groups.with_columns(
       (pl.when((pl.col("group") == "Elderly") & \
               (pl.col("vaccine") == "PCV13"))
       .then("pcv13_adult")
       .otherwise(pl.col("vaccine")).alias("vaccine"))) 
    
    serotype_groups = serotype_groups.join(antibody_distributions, 
                         left_on=["vaccine","immune_group", "group"], 
                         right_on = ["vaccine_type","Group", "age"],
                         how = "left")
    
    updated_cols = pl.DataFrame([
        pl.Series("vaccine", ["PCV7","PCV7","PCV7","PCV7",
                              "PCV13", "pcv13_adult","PCV13","PCV13","PCV13",
                              "PPV23","PPV23" ]),
        pl.Series("schedule", ["Post dose1","Post dose2: 2-4M or 2-4-6M",
                     "Post dose3: 2-4-6M", "Post toddler: 2-4M/12M",\
                   "Post dose1", "1-dose", "Post dose2: 2-4M or 2-4-6M",\
                    "Post dose3: 2-4-6M", "Post toddler: 2-4M/12M",\
                    "1-dose","2-dose (60-64Y)" ]),
        pl.Series("new_schedule", [1,2,3,4,\
                        1,1,2,3,3,\
                        1,2]),
        pl.Series("new_vaccine", ["pcv7","pcv7","pcv7","pcv13_catchup", \
                      "pcv13", "pcv13_adult","pcv13","pcv13_30","pcv13_21",\
                       "ppsv23_adult", "ppsv23_adult"])
        ])
        
    serotype_groups = (serotype_groups.join(updated_cols, 
        on=["vaccine", "schedule"], how= "left")
                              .drop(["vaccine", "schedule"])
          .rename({"new_schedule": "schedule","new_vaccine": "vaccine_type"})
          .with_columns(pl.col("schedule").cast(pl.Int32)))
    """serotype_groups.filter(
    (pl.col("schedule") == 1) & (pl.col("serotype") == "4"))"""
    pcv1330_groups = (serotype_groups.filter(
        (pl.col("vaccine_type") == "pcv13") & 
        (pl.col("group") == "Infant_toddler"))
                    .with_columns(pl.lit("pcv13_30").alias("vaccine_type")))
    pcv1321_groups = (serotype_groups.filter(
        (pl.col("vaccine_type") == "pcv13") & 
        (pl.col("group") == "Infant_toddler"))
                    .with_columns(pl.lit("pcv13_21").alias("vaccine_type")))
    pcv13adult_groups = (serotype_groups.filter(
        (pl.col("vaccine_type") == "pcv13_adult") & 
        (pl.col("group") == "Elderly"))
                    .with_columns(pl.lit("pcv13_adult").alias("vaccine_type")))
    
    serotype_groups = serotype_groups.filter(
        pl.col("vaccine_type") != "pcv13")
    
    serotype_groups = pl.concat([serotype_groups,
                     pcv1330_groups, pcv1321_groups,pcv13adult_groups], 
                                rechunk = True).drop("age_right")
    
    serotype_groups.filter(
        pl.col("vaccine_type") == "pcv13_31")["schedule"].unique()
    vaccine_antibody_df["vaccine_type"].unique()
    vaccine_antibody_df.filter(
        pl.col("vaccine_type") == "pcv13_31")["no_of_doses"].unique()
    #serotype_groups["vaccine_type"].unique().to_list()
    #serotype_groups.filter(pl.col("vaccine_type") == "atrisk_pcv13_child")
    ppsv23 = (serotype_groups.filter(pl.col("vaccine_type") == "ppsv23_adult")
              .with_columns(pl.lit("ppsv23").alias("vaccine_type")))
    serotype_groups = pl.concat([serotype_groups, ppsv23], 
                                how= "diagonal")
    vaccine_antibody_df = (vaccine_antibody_df.join(serotype_groups,
                                left_on=["vaccine_type", "no_of_doses",
                                         "exposed_strains"],  
                                right_on=["vaccine_type", 
                                          "schedule","serotype"],
                                how= "left").drop("group", 
                                                 "immune_group", "dist")
                            .with_columns(
                                pl.col(["meanlog"]).fill_null(-9),
                                pl.col(["sdlog"]).fill_null(0.000000000001)
                                ))
    curr_vaccines = set(vaccine_antibody_df["vaccine_type"]
                                .unique().to_list())
    curr_vaccines.remove("")
    are_all_vaccines_included = curr_vaccines.issubset(
                    set(serotype_groups["vaccine_type"].unique().to_list()))
    if not are_all_vaccines_included:
        print("Error, check the vaccines in vaccine_antibody_df")
        print(curr_vaccines)
        print(set(serotype_groups["vaccine_type"].unique().to_list()))
    return vaccine_antibody_df

    
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
