#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:04:26 2023

@author: ntellioglu
"""
import polars as pl
from scipy.stats import logistic
import numpy as np

def remove_v116_maternal(popDf, vaccine, value, t, day, t_per_year, vaccines, 
              at_risk_vaccine_target_groups, rng):
    
    #no prev vaccination
    vacc_target_group = (
         popDf.filter( (
             (pl.col("vaccines").struct.field("vaccine_type").str
                     .starts_with(vaccine))
             ) &\
            (pl.col("age_days") >= 56) 
             )
            .select(["id","age", "age_days", 
                     "vaccines", "random"])
     )
    
    
    remove_mat_vacc = (vacc_target_group.with_columns(
        pl.struct(no_of_doses = pl.lit(0), 
                   on_time = pl.lit(0).cast(pl.Int64),
                    vaccine_type = pl.lit("").cast(pl.Utf8),
                    final_vaccine_time = \
                            pl.Series('final_vaccine_time', 
                    [None] * vacc_target_group.height, dtype = pl.Int32),
                    #antibody_level_after_final_dose = pl.lit(0.0),
            ).alias("vaccines")
        ))                                                    
    #print(cont_vacc_target_group["vaccines"].struct.schema)
    updated_target_group = (remove_mat_vacc.unnest("vaccines"))
    popDf = popDf.unnest("vaccines")
    popDf = popDf.update(updated_target_group, on="id", how="left") 
    popDf = (popDf.with_columns(pl.struct(pl.col(["no_of_doses",
                      "on_time", "vaccine_type", 
                      "final_vaccine_time" ]))
                             .alias("vaccines"))
                .drop(["no_of_doses","on_time", "vaccine_type", 
                                 "final_vaccine_time"]))
    return popDf

def roll_v116_maternal(P, vaccine, value, t, day, t_per_year, vaccines, 
              at_risk_vaccine_target_groups, rng):
    cur_year = day // 364 
    period = day / t
    vacc_rollout_year = cur_year - value["years"][0]
    
    on_time_coverage = value["on_time_coverage_frac"][\
                     vacc_rollout_year]
    late_coverage = value["late_coverage_frac"][\
                     vacc_rollout_year]
    #no prev vaccination
    vacc_target_group = (
         P.I.filter((pl.col("age")
             .is_between(value["vaccination_age_range"][0],
                         value["vaccination_age_range"][1])
             & \
            (pl.col("vaccines").struct.field("vaccine_type").str
                     .starts_with(vaccine).is_not())
             
             ) )
            .select(["id","age", "age_days", 
                     "vaccines", "random"])
     )
    
    vacc_target_group = (
         vacc_target_group.filter(
             (pl.col("vaccines").struct.field("no_of_doses")
                  < len(value["daily_schedule"])) 
                          ))
    #on time first dose
    on_time_first_vacc = (vacc_target_group.filter(
     (pl.col("age_days") < period) &
           (pl.col("random") <= on_time_coverage)
          ))
    late_first_vacc = (vacc_target_group.filter(#on time 
       (pl.col("age_days") < period ) &
           ((on_time_coverage < pl.col("random")) & \
            (pl.col("random") <= \
            on_time_coverage + (late_coverage))
          )))
                         
    cont_vacc_target_group = (vacc_target_group.filter(
        (pl.col("age_days") >= period) |
            (pl.col("random") > (on_time_coverage + \
                                (late_coverage)))
          ))
    on_time_first_vacc = (on_time_first_vacc.with_columns(
             (pl.lit(rng.rand(on_time_first_vacc.height))
                  .alias("random")),
        pl.struct(
             pl.lit(1).alias("no_of_doses"),
             #(pl.col("vaccines").struct.field("no_of_doses") + 1),
             (pl.col("vaccines").struct.field("on_time")).alias("on_time"),
             (pl.lit(vaccine)).alias("vaccine_type"),
             pl.lit(day).alias("final_vaccine_time"),
             ).alias("vaccines")
             ))                                                    
        
    #update random column
    cont_vacc_target_group = cont_vacc_target_group.with_columns(
          (pl.lit(rng.rand(cont_vacc_target_group.height))
               .alias("random")),
          )
     
    #print(cont_vacc_target_group["vaccines"].struct.schema)
    updated_target_group = (pl.concat([on_time_first_vacc, 
                                     late_first_vacc,
                                 cont_vacc_target_group])
                             .unnest("vaccines"))
    P.I = P.I.unnest("vaccines")
    P.I = P.I.update(updated_target_group, on="id", how="left") 
    P.I = (P.I.with_columns(pl.struct(pl.col(["no_of_doses",
                      "on_time", "vaccine_type", 
                      "final_vaccine_time" ]))
                             .alias("vaccines"))
                .drop(["no_of_doses",
                                 "on_time", "vaccine_type", 
                                 "final_vaccine_time"]))
    return P

def roll_v116(P, vaccine, value, t, day, t_per_year, vaccines, 
              at_risk_vaccine_target_groups, rng):
    cur_year = day // 364 
    
    vacc_rollout_year = cur_year - value["years"][0]
    
    on_time_coverage = value["on_time_coverage_frac"][\
                     vacc_rollout_year]
    late_coverage = value["late_coverage_frac"][\
                     vacc_rollout_year]
    #time to vaccinate individuals 
    if (vaccine.startswith("atrisk")) & (value["previous_vacc"] == ""):
        vacc_target_group = (
            P.I.filter((pl.col("age")
                .is_between(value["vaccination_age_range"][0],
                            value["vaccination_age_range"][1])
                
                ) &
                (pl.col("at_risk")
                 .is_in(at_risk_vaccine_target_groups)))
               .select(["id","age", "age_days", 
                        "vaccines", "random"])
        )
        #make sure they don't receive the vaccine beforehand
        vacc_target_group = (
            vacc_target_group.filter(
            ((pl.col("vaccines")
             .struct.field("vaccine_type") != vaccine)) &
            ((pl.col("vaccines")
             .struct.field("vaccine_type")).str
             .starts_with("atrisk").is_not())
                         ))
    elif ((vaccine.startswith("atrisk")) & 
                (value["previous_vacc"] == "atrisk_pcv20_adult")):
        vacc_target_group = (
            P.I.filter((pl.col("age")
                .is_between(value["vaccination_age_range"][0],
                            value["vaccination_age_range"][1])
                ) &
                #(pl.col("at_risk")
                # .is_in(at_risk_vaccine_target_groups)
                # )&\
          (pl.col("vaccines").struct.field("no_of_doses") == 1) & \
                (pl.col("vaccines").struct.field("vaccine_type")
                 .is_in([value["previous_vacc"]])) &\
                ((day - 
  pl.col("vaccines").struct.field("final_vaccine_time")) == (364 * 
                                            value["year_since_prev_vacc"]))
                             )
               .select(["id","age", "age_days", 
                        "vaccines", "random"])
        )
        
    
    elif ((not vaccine.startswith(("atrisk"))) & 
            (value["previous_vacc"] == "")):
        #no prev vaccination
        vacc_target_group = (
             P.I.filter((pl.col("age")
                 .is_between(value["vaccination_age_range"][0],
                             value["vaccination_age_range"][1])
                 & \
                (pl.col("vaccines").struct.field("vaccine_type").str
                         .starts_with(("atrisk")).is_not())
                 
                 ) )
                .select(["id","age", "age_days", 
                         "vaccines", "random"])
         )
        """vacc_target_group = (
             vacc_target_group.filter(
                 (pl.col("vaccines").struct.field("no_of_doses")
                      < len(value["daily_schedule"]))
                              ))"""
        vacc_target_group = (
             vacc_target_group.filter(
                 (pl.col("vaccines").struct.field("no_of_doses")
                      < len(value["daily_schedule"])) |
                 ( (pl.col("vaccines").struct.field("vaccine_type")
                   .str.contains("adult").is_not()) )
                              ))
        #if vacc_target_group.height:
        #    print("here")
             
    elif ((not vaccine.startswith(("atrisk"))) & 
          (value["previous_vacc"] == "pcv20_adult")):
        vacc_target_group = (
            P.I.filter((pl.col("age")
                .is_between(value["vaccination_age_range"][0],
                    value["vaccination_age_range"][1])) & \
                   (pl.col("vaccines").struct.field("vaccine_type").str
                            .starts_with(("atrisk")).is_not()) &\
          (pl.col("vaccines").struct.field("no_of_doses") == 1) & \
                (pl.col("vaccines").struct.field("vaccine_type")
                 .is_in([value["previous_vacc"]])) &\
                ((day - 
  pl.col("vaccines").struct.field("final_vaccine_time")) == (364 * 
                                            value["year_since_prev_vacc"]))
                             )
               .select(["id","age", "age_days", 
                        "vaccines", "random"])
        )
    
    else:
        print("specify the condition!!!!!!!")
    #on time first dose
    on_time_first_vacc = (vacc_target_group.filter(
     (pl.col("age_days") == value["daily_schedule"][0]) &
           (pl.col("random") <= on_time_coverage)
          ))
    late_first_vacc = (vacc_target_group.filter(#on time 
       (pl.col("age_days") == value["daily_schedule"][0]) &
           ((on_time_coverage < pl.col("random")) & \
            (pl.col("random") <= \
            on_time_coverage + (late_coverage))
          )))
                         
    cont_vacc_target_group = (vacc_target_group.filter(
        (pl.col("age_days") != value["daily_schedule"][0]) |
            (pl.col("random") > (on_time_coverage + \
                                (late_coverage)))
          ))
    on_time_first_vacc = (on_time_first_vacc.with_columns(
             (pl.lit(rng.rand(on_time_first_vacc.height))
                  .alias("random")),
        pl.struct(
             pl.lit(1).alias("no_of_doses"),
             #(pl.col("vaccines").struct.field("no_of_doses") + 1),
             (pl.col("vaccines").struct.field("on_time")).alias("on_time"),
             (pl.lit(vaccine)).alias("vaccine_type"),
             pl.lit(day).alias("final_vaccine_time"),
             ).alias("vaccines")
             ))                                                    
        
    #update random column
    cont_vacc_target_group = cont_vacc_target_group.with_columns(
          (pl.lit(rng.rand(cont_vacc_target_group.height))
               .alias("random")),
          )
     
     
    
     
    #print(cont_vacc_target_group["vaccines"].struct.schema)
    updated_target_group = (pl.concat([on_time_first_vacc, 
                                     late_first_vacc,
                                 cont_vacc_target_group])
                             .unnest("vaccines"))
    P.I = P.I.unnest("vaccines")
    P.I = P.I.update(updated_target_group, on="id", how="left") 
    P.I = (P.I.with_columns(pl.struct(pl.col(["no_of_doses",
                      "on_time", "vaccine_type", 
                      "final_vaccine_time" ]))
                             .alias("vaccines"))
                .drop(["no_of_doses",
                                 "on_time", "vaccine_type", 
                                 "final_vaccine_time"]))
    return P
      
if __name__ == '__main__':
    import sys,os

    repo_path = ".."
    repo_path = os.path.abspath(os.path.join(repo_path))
    if repo_path not in sys.path:
        sys.path.append(repo_path)
    os.chdir(os.path.join(repo_path))
