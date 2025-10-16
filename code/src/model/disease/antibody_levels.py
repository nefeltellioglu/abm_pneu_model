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
    """
    vaccine_antibody_df["vaccine_type"].unique().to_list()
    vdf = vaccine_antibody_df.group_by("vaccine_type", "no_of_doses",
                                       "meanlog").agg(pl.col("sdlog").first(),
                                   (pl.col("exposed_strains").explode())
                            ).filter(pl.col("meanlog") != -9).sort(
                                "vaccine_type", "no_of_doses", "meanlog").with_columns(
   pl.format("[{}]",
      pl.col("exposed_strains").cast(pl.List(str)).list.join(", ")).alias("exposed_strains"))
                                
                                
                              
    import os
    vdf.write_csv( os.path.join("/Users/tellioglun/Documents/pneumonia/git_v5/output/imm.csv"))
    
    vdf2 = vaccine_antibody_df.filter(pl.col("no_of_doses") == 1).group_by(
        "vaccine_type", "meanlog").agg(
        pl.exclude("exposed_strains").first(),
         (pl.col("exposed_strains").explode())
         ).filter(pl.col("meanlog") != -9).sort(
                  "vaccine_type", "no_of_doses", "meanlog").with_columns(
                      pl.format("[{}]",
      pl.col("exposed_strains").cast(pl.List(str)).list.join(", ")).alias("exposed_strains"))
    vdf2.write_csv( os.path.join("/Users/ntelliogluce/Documents/pneumonia/git_v5/output/imm_groups.csv"))
    """
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
    serotype_groups2 = add_antibody_levels_pcv15_20()
    serotype_groups = pl.concat([serotype_groups, serotype_groups2], 
                                how= "diagonal")
    serotype_groups3 = add_antibody_levels_v116()
    serotype_groups = pl.concat([serotype_groups, serotype_groups3], 
                                how= "diagonal")
    
    serotype_groups4 = add_antibody_levels_v116_maternal()
    serotype_groups = pl.concat([serotype_groups, serotype_groups4], 
                                how= "diagonal")
    
    
    serotype_groups = add_antibody_levels_at_risk_groups(serotype_groups)
    serotype_groups = add_antibody_levels_indigenous_base(serotype_groups)
    serotype_groups = add_antibody_levels_indigenous_scenarios(serotype_groups)
    
    
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
    #serotype_groups.write_csv("output/serotype_groups.csv")
    
    are_all_vaccines_included = curr_vaccines.issubset(
                    set(serotype_groups["vaccine_type"].unique().to_list()))
    if not are_all_vaccines_included:
        print("Error, check the vaccines in vaccine_antibody_df")
        print(curr_vaccines)
        print(set(serotype_groups["vaccine_type"].unique().to_list()))
    
    return vaccine_antibody_df

def add_antibody_levels_indigenous_base(serotype_groups):
    child_31_last_dose = (serotype_groups.filter(
                        (pl.col("vaccine_type") == "pcv13_21") &
                           (pl.col("schedule") == 3)).with_columns(
                            pl.lit("pcv13_31").alias("vaccine_type"),
                            pl.lit(4).alias("schedule"),
                            pl.lit("Infant_toddler").alias("group")
                        ))
    
    child_ppv23_pcv7_follow = (serotype_groups.filter(
                        (pl.col("vaccine_type") == "atrisk_ppv23_child") &
                           (pl.col("schedule") == 1)).with_columns(
                            pl.lit("ppv23_pcv7_follow").alias("vaccine_type"),
                            pl.lit(1).alias("schedule"),
                            pl.lit("Infant_toddler").alias("group")
                        ))
    
    child_31_first_3_doses = (serotype_groups.filter(
                            (pl.col("vaccine_type") == "pcv13_30") 
                            ).with_columns(
                             pl.lit("pcv13_31").alias("vaccine_type"),
                             pl.lit("Infant_toddler").alias("group") 
                               ))
    serotype_groups = pl.concat([serotype_groups,
                     child_31_first_3_doses,
                     child_31_last_dose, 
                     child_ppv23_pcv7_follow,
                     ], rechunk = True)
    return serotype_groups

def add_antibody_levels_indigenous_scenarios(serotype_groups):
    child_31_last_dose = (serotype_groups.filter(
                        (pl.col("vaccine_type") == "pcv20_21") &
                           (pl.col("schedule") == 3)).with_columns(
                            pl.lit("pcv20_31").alias("vaccine_type"),
                            pl.lit(4).alias("schedule"),
                            pl.lit("Infant_toddler").alias("group")
                        ))
    
    child_31_first_3_doses = (serotype_groups.filter(
                            (pl.col("vaccine_type") == "pcv20_21") 
                            ).with_columns(
                             pl.lit("pcv20_31").alias("vaccine_type"),
                             pl.lit("Infant_toddler").alias("group") 
                               ))
                                
    adult_1dose_ppv233_pcv13 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv13_adult_follow").alias("vaccine_type"),
                     pl.lit("Elderly").alias("group"),
                     )
    adult_1dose_ppv233_pcv13_2 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv13_adult_follow_2").alias("vaccine_type"),
                     pl.lit("Elderly").alias("group"),
                     )
    adult_1dose_ppv233_pcv20 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv20_adult_follow").alias("vaccine_type"),
                     pl.lit("Elderly").alias("group"),
                     )
    adult_1dose_ppv233_pcv20_2 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv20_adult_follow_2").alias("vaccine_type"),
                     pl.lit("Elderly").alias("group"),
                     )
    adult_1dose_ppv233_pcv15 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv15_adult_follow").alias("vaccine_type"),
                     pl.lit("Elderly").alias("group"),
                     )
    adult_1dose_ppv233_pcv15_2 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv15_adult_follow_2").alias("vaccine_type"),
                     pl.lit("Elderly").alias("group"),
                     )   
    adult_1dose_ppv233_v116 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_V116_adult_follow").alias("vaccine_type"),
                     pl.lit("Elderly").alias("group"),
                     )
    adult_1dose_ppv233_v116_2 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_V116_adult_follow_2").alias("vaccine_type"),
                     pl.lit("Elderly").alias("group"),
                     )   
                    
    ppv23_pcv13_child = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv13_follow").alias("vaccine_type"),
                     pl.lit("Infant_toddler").alias("group"),
                     )
    ppv23_pcv13_child_2 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv13_follow_2").alias("vaccine_type"),
                     pl.lit("Infant_toddler").alias("group"),
                     )
    ppv23_pcv20_child = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv20_follow").alias("vaccine_type"),
                     pl.lit("Infant_toddler").alias("group"),
                     )
    ppv23_pcv20_child_2 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv20_follow_2").alias("vaccine_type"),
                     pl.lit("Infant_toddler").alias("group"),
                     )
    #ind runs with 2 adult doses
    adult_1dose_pcv20_50 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "pcv20_adult"))).with_columns(
                     pl.lit("pcv20_adult_50").alias("vaccine_type"),
                     pl.lit("Elderly").alias("group"),
                     )
    
    adult_1dose_ppv233_pcv20_50 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv20_adult_follow_50").alias("vaccine_type"),
                     pl.lit("Elderly").alias("group"),
                     )
    adult_1dose_ppv233_pcv20_2_50 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("ppv23_pcv20_adult_follow_2_50").alias("vaccine_type"),
                     pl.lit("Elderly").alias("group"),
                     )

    
    serotype_groups = pl.concat([serotype_groups,
                     child_31_first_3_doses,child_31_last_dose, 
                     ppv23_pcv13_child, ppv23_pcv13_child_2,
                     ppv23_pcv20_child, ppv23_pcv20_child_2,
                     adult_1dose_ppv233_pcv13, adult_1dose_ppv233_pcv13_2,
                     adult_1dose_ppv233_pcv20, adult_1dose_ppv233_pcv20_2,
                     adult_1dose_pcv20_50,
                 adult_1dose_ppv233_pcv20_50, adult_1dose_ppv233_pcv20_2_50,
                 adult_1dose_ppv233_pcv15, adult_1dose_ppv233_pcv15_2,
                 adult_1dose_ppv233_v116, adult_1dose_ppv233_v116_2
                     ], rechunk = True)
    return serotype_groups

def add_antibody_levels_at_risk_groups(serotype_groups):
    serotype_groups.filter(((pl.col("schedule") == 1) & 
                            (pl.col("vaccine_type") == "pcv13_21") & 
                            pl.col("serotype").is_in(["4", "9V", "14"])) | 
                           (pl.col("schedule") == 1) & 
                           (pl.col("vaccine_type") == "pcv13_adult") &
                           pl.col("serotype").is_in(["4", "9V", "14"])
                           ).unique().sort("serotype")
    
    child_1dose_pcv13 = serotype_groups.filter(((pl.col("schedule") == 3) & 
                    (pl.col("vaccine_type") == "pcv13_21"))).with_columns(
                        pl.lit("atrisk_pcv13_child").alias("vaccine_type"),
                        pl.lit("atrisk_child").alias("group"),
                        pl.lit(1).alias("schedule")
                        )
    adult_1dose_pcv13 = serotype_groups.filter(((pl.col("schedule") == 1) & 
         (pl.col("vaccine_type") == "pcv13_adult"))).unique().with_columns(
                         pl.lit("atrisk_pcv13_adult").alias("vaccine_type"),
                         pl.lit("atrisk_adult").alias("group"),
                         )

    child_1dose_pcv15 = serotype_groups.filter(((pl.col("schedule") == 3) & 
        (pl.col("vaccine_type") == "pcv15_21"))).sort("serotype").with_columns(
            pl.lit("atrisk_pcv15_child").alias("vaccine_type"),
            pl.lit("atrisk_child").alias("group"),
            pl.lit(1).alias("schedule")
            )
    adult_1dose_pcv15 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                   (pl.col("vaccine_type") == "pcv15_adult"))).with_columns(
                        pl.lit("atrisk_pcv15_adult").alias("vaccine_type"),
                        pl.lit("atrisk_adult").alias("group"),
                        )
    
    child_1dose_pcv20 = serotype_groups.filter(((pl.col("schedule") == 3) & 
                    (pl.col("vaccine_type") == "pcv20_21"))).with_columns(
                        pl.lit("atrisk_pcv20_child").alias("vaccine_type"),
                        pl.lit("atrisk_child").alias("group"),
                        pl.lit(1).alias("schedule")
                        )
    adult_1dose_pcv20 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "pcv20_adult"))).with_columns(
                     pl.lit("atrisk_pcv20_adult").alias("vaccine_type"),
                     pl.lit("atrisk_adult").alias("group"),
                     )
    
    adult_1dose_ppv23 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("atrisk_ppv23_adult").alias("vaccine_type"),
                     pl.lit("atrisk_adult").alias("group"),
                     )
    adult_1dose_ppv23_2 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("atrisk_ppv23_adult_2").alias("vaccine_type"),
                     pl.lit("atrisk_adult").alias("group"),
                     )
    atrisk_ppv23_child = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("atrisk_ppv23_child").alias("vaccine_type"),
                     pl.lit("atrisk_child").alias("group"),
                     )
    atrisk_ppv23_child_2 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "ppsv23_adult"))).with_columns(
                     pl.lit("atrisk_ppv23_child_2").alias("vaccine_type"),
                     pl.lit("atrisk_child").alias("group"),
                     )
    adult_1dose_v116 = serotype_groups.filter(((pl.col("schedule") == 1) & 
                (pl.col("vaccine_type") == "V116_adult"))).with_columns(
                     pl.lit("atrisk_V116_adult").alias("vaccine_type"),
                     pl.lit("atrisk_adult").alias("group"),
                     )
    serotype_groups = pl.concat([serotype_groups,
                     child_1dose_pcv13, adult_1dose_pcv13,
                     child_1dose_pcv15,adult_1dose_pcv15,
                     child_1dose_pcv20, adult_1dose_pcv20,
                     adult_1dose_ppv23, adult_1dose_ppv23_2,
                     atrisk_ppv23_child, atrisk_ppv23_child_2,
                     adult_1dose_v116
                     ], rechunk = True)
    return serotype_groups

def add_antibody_levels_pcv15_20():
    antibody_distributions = pl.read_csv(
    "data/immunity/antibody_distributions_new_vaccines.csv")
    serotype_groups = pl.read_csv(
    "data/immunity/immune_group_list_20231027_sen.csv")
    antibody_distributions = antibody_distributions.filter(
                            (pl.col("vaccine_type")
                                  .is_in(["PCV20","PCV15","PPV23"])) & \
         (pl.col("Vaccine_naive") != "receive_PPV23_morethan_3y_before")
                                  )
    #antibody_distributions["schedule"].unique()
    serotype_groups = serotype_groups.filter((pl.col("vaccine")
                                  .is_in(["PCV20","PCV15","PPV23"])) )
    
    schedule_groups = pl.DataFrame([
        pl.Series("age", ["Elderly"] ),
        pl.Series("schedule", ["1-dose"]),
        ])
    antibody_distributions = antibody_distributions.join(schedule_groups, 
            on = "schedule", how = "left").with_columns(
                pl.col("age").fill_null(
                    value = "Infant_toddler")).drop("ratio", 
                                                    "sd_square", "mean")
    imm_groups = pl.DataFrame([
    pl.Series("immune_group", ["Reference","High-multi","Low-multi","ST3" ]),
        pl.Series("n_immune_group", ["Ref", "High", "Low", "ST3"]),
        ])
    serotype_groups = serotype_groups.join(imm_groups, on = "immune_group", 
                                           how="left")
    #sum(serotype_groups["n_immune_group"].is_null())
    serotype_groups = serotype_groups.join(antibody_distributions, 
                         left_on=["vaccine","n_immune_group", "group"], 
                         right_on = ["vaccine_type","Group", "age"],
                         how = "left")
    serotype_groups["schedule"].unique()
    updated_cols = pl.DataFrame([
        pl.Series("vaccine", ["PCV15"] * 5 + ["PCV20"] * 5),
        pl.Series("schedule", ["Post dose1","Post dose2: 2-4M or 2-4-6M",
                "Post dose3: 2-4-6M", "Post toddler: 2-4M/12M","1-dose"] * 2
                    ),
        pl.Series("new_schedule", [1,2,3,3,1] * 2),
        pl.Series("new_vaccine", ["pcv15","pcv15","pcv15_30","pcv15_21", \
         "pcv15_adult", "pcv20","pcv20","pcv20_30","pcv20_21", "pcv20_adult"])
        ])
        
    serotype_groups = (serotype_groups.join(updated_cols, 
        on=["vaccine", "schedule"], how= "left")
                              .drop(["vaccine", "schedule"])
          .rename({"new_schedule": "schedule","new_vaccine": "vaccine_type"})
          .with_columns(pl.col("schedule").cast(pl.Int32)))
    """serotype_groups.filter(
    (pl.col("schedule") == 1) & (pl.col("serotype") == "4"))"""
    pcv1530_groups = (serotype_groups.filter(
        (pl.col("vaccine_type") == "pcv15") & 
        (pl.col("group") == "Infant_toddler"))
                    .with_columns(pl.lit("pcv15_30").alias("vaccine_type")))
    pcv1521_groups = (serotype_groups.filter(
        (pl.col("vaccine_type") == "pcv15") & 
        (pl.col("group") == "Infant_toddler"))
                    .with_columns(pl.lit("pcv15_21").alias("vaccine_type")))
    pcv2030_groups = (serotype_groups.filter(
        (pl.col("vaccine_type") == "pcv20") & 
        (pl.col("group") == "Infant_toddler"))
                    .with_columns(pl.lit("pcv20_30").alias("vaccine_type")))
    pcv2021_groups = (serotype_groups.filter(
        (pl.col("vaccine_type") == "pcv20") & 
        (pl.col("group") == "Infant_toddler"))
                    .with_columns(pl.lit("pcv20_21").alias("vaccine_type")))

    serotype_groups = serotype_groups.filter(
        (pl.col("vaccine_type") != "pcv15")).filter(
            (pl.col("vaccine_type") != "pcv20"))
    
    serotype_groups = pl.concat([serotype_groups,
                     pcv1530_groups, pcv1521_groups,pcv2030_groups,
                     pcv2021_groups], rechunk = True)
    
    serotype_groups = serotype_groups.drop("Vaccine_naive",
                         "immune_group").rename(
                             {"n_immune_group": "immune_group"})                                        
    
    return serotype_groups
    
def add_antibody_levels_v116():
    antibody_distributions = pl.read_csv(
    "data/immunity/v116_immune_group_parameters.csv")
    
    serotype_groups = (antibody_distributions
                       .drop("Sub", "Total", "Proportion")
                       .rename({"vaccine": "vaccine_type",
                                "ln_mu": "meanlog",
                                "ln_sigma2": "sdlog"}))
    
    imm_groups = pl.DataFrame([
    pl.Series("immune_group", ["Mod","High","Low","ST3" ]),
        pl.Series("n_immune_group", ["Ref", "High", "Low", "ST3"]),
        ])
    serotype_groups = (serotype_groups
                       .join(imm_groups, on = "immune_group", how="left")
                       .drop("immune_group")
                       .rename({"n_immune_group": "immune_group"})
                       .with_columns(pl.lit(1).alias("schedule"),
                                   pl.lit("V116_adult").alias("vaccine_type")))
    return serotype_groups

def add_antibody_levels_v116_maternal():
    antibody_distributions = pl.read_csv(
    "data/immunity/v116_immune_group_parameters.csv")
    
    serotype_groups = (antibody_distributions
                       .drop("Sub", "Total", "Proportion")
                       .rename({"vaccine": "vaccine_type",
                                "ln_mu": "meanlog",
                                "ln_sigma2": "sdlog"}))
    
    imm_groups = pl.DataFrame([
    pl.Series("immune_group", ["Mod","High","Low","ST3" ]),
        pl.Series("n_immune_group", ["Ref", "High", "Low", "ST3"]),
        ])
    serotype_groups = (serotype_groups
                       .join(imm_groups, on = "immune_group", how="left")
                       .drop("immune_group")
                       .rename({"n_immune_group": "immune_group"})
                       .with_columns(pl.lit(1).alias("schedule"),
                          pl.lit("V116_maternal").alias("vaccine_type")))
    
    
    return serotype_groups
    
    
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
            
def waning_ratio_extended(vaccine_type, day, final_vaccine_time,
                 waning_halflife_day_adult, waning_halflife_day_child,
                 waning_halflife_day_maternal,
                 age, age_days):
    """
    

    Parameters
    ----------
    vaccine_type = possibly_infected["vaccines"].struct.field("vaccine_type")
    day = day
    final_vaccine_time = possibly_infected["vaccines"].struct.field("final_vaccine_time")
    waning_halflife_day_adult = self.waning_halflife_day_adult
  
    waning_halflife_day_child = self.waning_halflife_day_child
    waning_halflife_day_maternal = self.waning_halflife_day_maternal
    age = possibly_infected["age"]
    age_days = possibly_infected["age_days"]
    

    """

    
    halflife = pl.DataFrame([
        vaccine_type,
        final_vaccine_time, age, age_days])
    
    halflife = halflife.with_columns(
        pl.when(
            pl.col("age") >= 18)
        .then(np.log(0.5)/waning_halflife_day_adult)
        .otherwise( 
                    pl.when((pl.col("age")== 0) & (pl.col("age_days") < 56))
                    .then(np.log(0.5)/waning_halflife_day_maternal)
                    .otherwise(np.log(0.5)/waning_halflife_day_child))
        .alias("daily_decay")
        ).with_columns(
            np.exp(pl.col("daily_decay") * 
                   (day - pl.col("final_vaccine_time"))).alias("waning_ratio")
            )
            
    return halflife["waning_ratio"]
                  
if __name__ == '__main__':
    import sys,os

    repo_path = ".."
    repo_path = os.path.abspath(os.path.join(repo_path))
    if repo_path not in sys.path:
        sys.path.append(repo_path)
    os.chdir(os.path.join(repo_path))
