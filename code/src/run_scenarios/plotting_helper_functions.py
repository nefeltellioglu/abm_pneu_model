#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 16:22:51 2025

@author: tellioglun
"""
import os
import polars as pl
import numpy as np
import h5py    
from matplotlib.pyplot import cm
import json
import matplotlib.pyplot as plt

def get_prev_data(directory,data_dir, filename, year, indigenous = None):
    
    full_file_address = os.path.join('%s/%s'%(directory,filename))
    #data filenames
    
    strain_fname = os.path.join(data_dir,'disease/strain_list.dat')
    strain_list = []
    for l in open(strain_fname, 'r'):
        if l[0] == '#': continue
        line = l.strip().split(' ')
        strain_list.append(line[0])
    strain_list = np.sort(strain_list)
    if filename == "dummy":
        file_addresses = os.path.join(directory)
        filenames = os.listdir(file_addresses)#os.listdir(directory)
        #display(filenames)
        file_addresses = filenames.copy()
        ind_file_addresses = []
        for name in filenames:
            if name.endswith("hd5"):
                ind_file_addresses.append(os.path.join(directory, name))
        #for folder in os.path.join(directory, file_addresses):
        if not ind_file_addresses:
            for cur_address in file_addresses:
                for path, subdirs, files in os.walk(os.path.join(directory, 
                                                                 cur_address)):
                    for name in files:
                        if name.endswith("hd5"):
                            ind_file_addresses.append(os.path.join(path, name))
    else:
        ind_file_addresses = [full_file_address]
    #savingaddress = os.path.join(directory, "output")
    
    all_df = pl.DataFrame()
    for filename in ind_file_addresses:#[vaccination_file_addresses[0],vaccination_file_addresses[1]]:
        try:
            with h5py.File(filename, "r") as f:
                #extract t_per_year 
                os.path.dirname(filename)
                try:
                    run_no = int(filename.split("/")[-2].split("_")[1])
                except:
                    print("error, run no will be assigned as 0")
                    run_no = 0
                #child_vacc = re.split('\_|\.dat',filename.split("/")[-3])[-4]
                #adult_vacc = re.split('\_|\.dat',filename.split("/")[-3])[-2]
                params = dict(f['params'].attrs.items())
                t_per_year = 364#params['t_per_year']
                
                #
                group_name = 'prevalence'
                key = 'base_data'
                raw_data = f[group_name][key][()].tolist()
                names = f[group_name][key][()].dtype.names
                df = pl.from_numpy(f[group_name][key][()],
                    schema=f[group_name][key][()].dtype.names, orient="col")
                df = df.with_columns(
                    pl.lit(run_no).alias("run_no"),
                    #pl.lit(child_vacc).alias("child_vacc"),
                    #pl.lit(adult_vacc).alias("adult_vacc"),
                    )
            f.close()
            all_df = pl.concat([all_df, df], how="diagonal", rechunk=True)
        except:
            with h5py.File(filename, "r") as f:
                #extract t_per_year 
                os.path.dirname(filename)
                #child_vacc = re.split('\_|\.dat',filename.split("/")[-3])[-4]
                #adult_vacc = re.split('\_|\.dat',filename.split("/")[-3])[-2]
                params = dict(f['params'].attrs.items())
                t_per_year = 364#params['t_per_year']
                
                #
                group_name = 'prevalence'
                key = 'base_data'
                raw_data = f[group_name][key][()].tolist()
                names = f[group_name][key][()].dtype.names
                df = pl.from_numpy(f[group_name][key][()],
                    schema=f[group_name][key][()].dtype.names, orient="col")
                
            f.close()
            all_df = pl.concat([all_df, df], how="diagonal", rechunk=True)

    return (all_df, strain_list)

def plot_base_prev(axes, data_dir,
                   df, strain_list, years = None,ymax = 1.5,
                   pcv_strains = False,
                   overall_prev = False, plot_vaccine_lines = True, 
                   indigenous = None, df_overall = pl.DataFrame()):
    
    """
    legend_labels: a list of two labels followed by legend title
    
    """
    vaccine_fname = os.path.join(data_dir, 'vaccine_configs/vaccine_list.dat')
    f = open(vaccine_fname)
    vaccines = json.load(f)
    start_year = 2002
    if indigenous:
        start_year = 2001
    if pcv_strains:
        ipcv7color = iter(cm.rainbow(np.linspace(0, 0.2, 7)))
        ipcv13color = iter(cm.rainbow(np.linspace(0.4, 0.9, 7)))

        colors = cm.rainbow(np.linspace(0, 1, 20))
        c_i = 0
    else:
        colors = cm.rainbow(np.linspace(0, 1, 5))
    if years:
        weeks = df["t"].unique().sort()
        weeks = weeks[:((years[1] - start_year) * 52)]
        total_years = years[1] - years[0]
        #times = [(years[0] + time / 52) for time in weeks]
        times = [(start_year + time / 52) for time in df["t"].unique().sort()]
        #interval = ((52 * (years[0] - 2002)),
        #                       ((years[1] - 2002) * 52))
        interval = (0, (len(times) - 1))
        
        
    else:
        times = [(start_year + time / 52) for time in df["t"].unique().sort()]
        interval = (0, (len(times) - 1))
    df_agg = df
    if plot_vaccine_lines:
        if indigenous:
            if 2001 in range(years[0], years[1]):
                plt.vlines(x=[2001], ymin=0, ymax=ymax, 
                       colors='#7f18f5', ls='--', 
                       lw=2,) #label='PCV13 3+0 schedule')
                plt.text(2001.1,ymax*0.85,'7vPCV\nschedule', color ='#7f18f5', 
                     fontsize=9)
            if 2011 in range(years[0], years[1]):
                plt.vlines(x=[2011], ymin=0, ymax=ymax, 
                       colors="#0d90db", ls='--', 
                       lw=2,) #label='PCV13 3+0 schedule')
                plt.text(2011.1,ymax*0.85,'13vPCV 3+1\nschedule', 
                     color ="#0d90db", fontsize=9)
        else:
            if 2005 in range(years[0], years[1]):
                plt.vlines(x=[2005], ymin=0, ymax=ymax, 
                       colors='#7f18f5', ls='--', 
                       lw=2,) #label='PCV13 3+0 schedule')
                plt.text(2005.1,ymax*0.85,'7vPCV\nschedule', color ='#7f18f5', 
                     fontsize=9)
            if 2011 in range(years[0], years[1]):
                plt.vlines(x=[2011], ymin=0, ymax=ymax, 
                       colors="#0d90db", ls='--', 
                       lw=2,) #label='PCV13 3+0 schedule')
                plt.text(2011.1,ymax*0.85,'13vPCV\nschedule',
                         #'13vPCV 3+0\nschedule', 
                     color ="#0d90db", fontsize=9)
            """if 2019 in range(years[0], years[1]):
                plt.vlines(x=[2017], ymin=0, ymax=ymax, 
                       colors='#0d90db', ls='--', 
                       lw=2,) #label='PCV13 2+1 schedule')
                plt.text(2018.1,ymax*0.85,'13vPCV 2+1\nschedule', color='#0d90db',
                     fontsize=9)"""
    
    for s,strain in enumerate(df_agg["strain_names"].unique().sort()):
        if (strain in vaccines["pcv7"]["serotypes"]) or (strain == "7vPCV"):
            if pcv_strains:
                color = next(ipcv7color)#colors[c_i]
                c_i += 1
                label = "7vPCV - " + strain
            else:
                color = colors[0]
                label = "7vPCV"
            if (strain in vaccines["pcv7"]["serotypes"]):
                alpha = 1
                alpha_fill = 0.05
            else:
                alpha = 1
                alpha_fill = 0.5
        elif (strain in vaccines["pcv13_30"]["serotypes"]) or \
                    (strain == "13vPCV"):
            if pcv_strains:
                #color = colors[c_i]
                color = next(ipcv13color)#colors[c_i]
                c_i += 1
                label = "13vPCV - " +strain
            else:
                
                color = colors[1]
                label = "13vPCV (not 7vPCV)"
            if (strain in vaccines["pcv13_30"]["serotypes"]):
                alpha = 1
                alpha_fill = 0.05
            else:
                alpha = 1
                alpha_fill = 0.5
        elif (strain in vaccines["ppsv23_1"]["serotypes"]) or \
            (strain == "23vPPV"):
            color = colors[-1]
            label = "23vPPV (not 13vPCV)"
            if (strain in vaccines["ppsv23_1"]["serotypes"]):
                alpha = 1
                alpha_fill = 0.05
            else:
                alpha = 1
                alpha_fill = 0.5
        else:
            color = "grey"
            label = "Rest of the serotypes"
            alpha = 0.5
            alpha_fill = 0.3
        strain_prev = df_agg.filter(
            pl.col("strain_names") == strain).sort("t")
        
        if sum(strain_prev["prev_mean"]*100):
            #ever existed
            axes.plot(times[interval[0]: interval[1]],
                 strain_prev["prev_mean"][interval[0]: interval[1]]*100,
                  color=color,alpha = alpha,#subdata2['%s'%y_var],
                 label=label
                 )
            axes.fill_between(times[interval[0]: interval[1]], 
                        strain_prev["prev_min"][interval[0]: interval[1]]*100, 
                        strain_prev["prev_max"][interval[0]: interval[1]]*100,
                    color=color,alpha = alpha_fill,#subdata2['%s'%y_var],
                            #label=label
                            
                             )

    """if overall_prev:
        color = "pink"
        label = "Overall"
        alpha = 1
        alpha_fill = 0.05
        df_agg = df_agg.group_by("t").agg(
            prev_mean = pl.col("prev_mean").sum(),
            prev_min = pl.col("prev_min").sum(),
            prev_max = pl.col("prev_max").sum(),).sort("t")
        axes.plot(times[interval[0]: interval[1]],
                  df_agg["prev_mean"][interval[0]: interval[1]]*100,
                         color=color,alpha = alpha,
                         label=label)  """
    if overall_prev:
        if not (df_overall.height):
            print("provide df for overall_prev")
            pass
        color = "#eb07d0"
        label = "All serotypes"
        alpha = 0.5
        alpha_fill = 0.3
        overall_prev_dt = df_overall.group_by("t","run_no").agg(
                     pl.col("strain_list").sum()
                     ).group_by("t").agg(
             prev_mean = pl.col("strain_list").mean(),
             prev_min = pl.col("strain_list").quantile(0.025),
             prev_max = pl.col("strain_list").quantile(0.975),).sort("t")
        
        axes.plot(times,
             overall_prev_dt["prev_mean"]*100,
              color=color,alpha = alpha,#subdata2['%s'%y_var],
             label=label
             )
        axes.fill_between(times, 
                         overall_prev_dt["prev_min"]*100, 
                         overall_prev_dt["prev_max"]*100,
                color=color,alpha = alpha_fill,)
    #legend_lines.append(x[0])
    
    axes.set_xlabel('Years')
    axes.set_ylabel('Prevalence (%)')
    axes.set_xticks(np.arange(int(times[0]),int(times[-1]) + 1, 2)) 
     
    axes.set_ylim(ymin=0.0, ymax = ymax)
    # Shrink current axis by 20%
    box = axes.get_position()
    axes.set_position(
        [box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    #axes.legend(legend_lines, legend_labels, title = legend_title,
    #            loc='center left', bbox_to_anchor=(1, 0.5))
    #leg = axes.legend(legend_lines, legend_labels, title = legend_title)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = np.flip(np.argsort(labels))
    handles = [handles[idx] for idx in order]
    labels = [labels[idx] for idx in order]
    by_label = dict(zip(labels, handles))

    if pcv_strains:
        axes.legend(
                    by_label.values(), by_label.keys(),
                    title = "Vaccine Strains",
                bbox_to_anchor=(1.5,1.1))
    else:
        axes.legend(by_label.values(), by_label.keys(),
                    title = "Vaccine Strains",
                bbox_to_anchor=(1.03,0.5))
   
    
    return axes

