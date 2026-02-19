#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:29:30 2023

@author: ntellioglu
"""

############
#plotting

import sys,os
#from math import ceil, sqrt, log
import h5py    
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.transforms import Affine2D

import polars as pl
import json
import re
#os.chdir(os.path.join(repo_path))

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(100)

#print(os.getcwd())
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
code_path = os.path.join(repo_path, "src")
code_path = os.path.abspath(os.path.join(repo_path, "src"))

if code_path not in sys.path:
    sys.path.append(code_path)

os.chdir(os.path.join(repo_path))

import pneu_abm
from pneu_abm.plot_outputs.plotting_utils import (get_prev_data, 
                                                get_disease_data,
                                                get_vaccine_data, 
                                                get_vaccine_by_type_data, 
                                                get_vaccine_delivered)

if __name__ == '__main__':
    
    
    output_directory = os.path.join(repo_path, 
        'src/pneu_abm/output',
        'historical_run/params_save_population=True/seed_0')
     
    data_dir = os.path.join(repo_path, 'src/pneu_abm/data')
    
   
    filename = 'dummy'
    years = (0,1)
    vac_df = get_vaccine_delivered(os.path.join(output_directory), data_dir,
                  filename, years[0])
    
    cur_vac_df = vac_df

    figsize=(7,3.5)#no in x axis, no in yaxis
    fig, axes = plt.subplots(figsize=figsize)
    
    all_df_agg = cur_vac_df.group_by("t", "vaccine_type").agg(
        pl.col(["no_delivered"]
              ).mean().name.suffix("_mean"),
        pl.col(["no_delivered"]
              ).quantile(0.025).name.suffix("_min"),
        pl.col(["no_delivered"]
              ).quantile(0.975).name.suffix("_max"),
        ).sort("t")
    
        
    vacc_types = cur_vac_df["vaccine_type"].unique()
    colors = cm.rainbow(np.linspace(0, 1, len(vacc_types) + 1))
    
    for v,vacc_type in enumerate(vacc_types):
        df_agg = all_df_agg.filter(
            pl.col("vaccine_type") == vacc_type).sort("t")
        err_min = [i if i > 0 else 0 for i in
            (df_agg["no_delivered_mean"] - \
              df_agg["no_delivered_min"])]
        err_max = [i if i > 0 else 0 for i in 
                   (df_agg["no_delivered_max"] - \
                    df_agg["no_delivered_mean"])]
        times = [(2002 + time/365) \
                 for time in df_agg["t"].unique().sort()]
        color = colors[-v]
        axes.errorbar(times,
            df_agg["no_delivered_mean"], 
            yerr=[err_min,err_max], 
            marker = 'o', ms = 3,
            color=color, label= "%s"%vacc_type)
    axes.set_xticks(np.arange(int(times[0]) - 1 ,int(times[-1]) + 1, 2)) 
    
    axes.set_xlabel('Years')
    axes.set_ylabel('Number of vaccines delivered')
    
     
    #axes.set_ylim(ymin=ymin, ymax = ymax)
    # Shrink current axis by 20%
    box = axes.get_position()
    axes.set_position(
        [box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    #axes.legend(legend_lines, legend_labels, title = legend_title,
    #            loc='center left', bbox_to_anchor=(1, 0.5))
    #leg = axes.legend(legend_lines, legend_labels, title = legend_title)
            
    handles, labels = plt.gca().get_legend_handles_labels()
    #order = np.flip(np.argsort(labels))
    #handles = [handles[idx] for idx in order]
    #labels = [labels[idx] for idx in order]
    by_label = dict(zip(labels, handles))
    bbox_to_anchor = (1.05,1)
    axes.legend(by_label.values(), by_label.keys(),
                    title = "Vaccine",
                 bbox_to_anchor = bbox_to_anchor)
    
    fig.savefig(os.path.join(output_directory, 
     'vacc_delivered.png'),bbox_inches = "tight",dpi=300)
    