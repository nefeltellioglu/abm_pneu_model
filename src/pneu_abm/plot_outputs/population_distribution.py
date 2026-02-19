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

#print(os.getcwd())
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
code_path = os.path.join(repo_path, "src")
code_path = os.path.abspath(os.path.join(repo_path, "src"))

if code_path not in sys.path:
    sys.path.append(code_path)

os.chdir(os.path.join(repo_path))
#print(sys.path)

#os.chdir(os.path.join(repo_path))

def get_age_dist_from_hdffile(directory, filename, year):
    
    full_file_address = os.path.join('%s/%s'%(directory,filename))
    
    if not os.path.isfile(full_file_address):
        raise ValueError(f"File {full_file_address} does not exist")
    else: #file exists
        with h5py.File(full_file_address, "r") as f:
            #extract t_per_year 
            params = dict(f['params'].attrs.items())
            t_per_year = params['t_per_year']
            
            #
            group_name = 'population'
            key = 'age_dists'
            raw_data = f[group_name][key][()].tolist()
            
        f.close()
        
        recorded_times = [raw_data[i][1] for i in range(len(raw_data))]
        first_recorded_time = recorded_times[0]
        print(recorded_times)
        year_in_recorded_times = first_recorded_time + year * t_per_year
        
        year_index = recorded_times.index(year_in_recorded_times)
        age_dist_from_year = raw_data[year_index][0]
        pop_size = sum(age_dist_from_year)
        age_dist_from_year_in_fractions = age_dist_from_year / pop_size
        
        return age_dist_from_year_in_fractions.tolist()
    
def get_age_dist_from_datfile(directory, filename, group_by_year = 5):
    """
    takes a file dist, for individual ages and 
    returns a list of age dist 
    
    ex inputs:
    repo_path = '/Users/ntellioglu/Documents/pneumonia/code_v4'
    directory = os.path.join(repo_path, 'data')

    filename = 'age_dist_2012.dat'
    group_by_year = 5
    
    """
    
    full_file_address = os.path.join('%s/%s'%(directory,filename))
    
    if not os.path.isfile(full_file_address):
        raise ValueError(f"File {full_file_address} does not exist")
    else: #file exists
        t = []
        for l in open(full_file_address, 'r'):
            if l[0] == '#': continue
            t.append([eval(x) for x in l.strip().split(' ')])
            
        age_dist = [line[0] for line in t]
        return age_dist#grouped_age_dist
    
      
def plot_age_dist_multi(axes, data, comp_dat,years = None, 
                        group_by_year=5, legend_labels = []):
    
    """
    legend_labels: a list of two labels followed by legend title
    
    """
    
    #group ages
    bins = list(range(0,105,group_by_year))
    x_tick_labels = [f'{x}-{x+4}' for x in bins]
    #grouped_age_dists = [(key, sum(list(group))) for (key, group) in itertools.groupby(age_dist0, key=lambda x: age_dist0.index(x) // 5)]
    
    grouped_data = [sum([x[1] for x in list(group)]) for (key, group) in 
                    itertools.groupby(list(enumerate(data)),  
                                      key=lambda x: x[0] // group_by_year)]
    
    grouped_comp_dat = [sum([x[1] for x in list(group)]) for (key, group) in 
                    itertools.groupby(list(enumerate(comp_dat)),  
                                      key=lambda x: x[0] // group_by_year)]
    
    if legend_labels:
        legend_title = legend_labels[2]
        legend_labels = legend_labels[ : -1]
        
    legend_lines = []
    
    x = None
    x = axes.bar(list(range(len(grouped_data))), grouped_data, 
                 align='edge', width=1.0,alpha = 0.5, color='red',
                 edgecolor = 'black')
    d = axes.bar(list(range(len(grouped_comp_dat))), 
                  grouped_comp_dat,align='edge', 
                  width=1.0,alpha = 0.5, color='blue', edgecolor = 'black')
    legend_lines.append(x[0])
    if not legend_labels and years:
        legend_labels.append(years[0])
        
    else:
        legend_labels.append('initial')
    legend_lines.append(d[0])
    
    if not legend_labels and years:
        legend_labels.append(years[1])
    else:
        legend_labels.append('final')
    
    axes.set_xlabel('Age')
    axes.set_ylabel('Fraction of population')
    axes.set_xticks(np.arange(len(grouped_data)))
    if len(grouped_data) == len(x_tick_labels):
        axes.set_xticklabels(x_tick_labels, rotation = 45)
    
    axes.set_ylim(ymin=0.0)
    # Shrink current axis by 20%
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    axes.legend(legend_lines, legend_labels, title = legend_title,
                loc='center left', bbox_to_anchor=(1, 0.5))
    #leg = axes.legend(legend_lines, legend_labels, title = legend_title)
    
    #leg.get_frame().set_alpha(0.0)
    
    return axes



if __name__ == '__main__':
    
    output_directory = os.path.join(repo_path, 
                             'src/pneu_abm/output/historical_run3',
                             'params_save_population=True/seed_0')

    filename = 'disease_0_21.hd5'
    years = (0,10)
    age_dist2002 = get_age_dist_from_hdffile(output_directory,
                                             filename, years[0])
    age_dist2012 = get_age_dist_from_hdffile(output_directory, 
                                             filename, years[1])
    years = (20,30)
    age_dist2022 = get_age_dist_from_hdffile(output_directory, 
                                             filename, years[0])
    
    #upload comp data
    data_directory = os.path.join(repo_path, 'src/pneu_abm/data/population')   
    #2002
    age_dist_filename = os.path.join('age_dist_2002.dat')
    comp_age_dist2002 = get_age_dist_from_datfile(data_directory, 
                                                  age_dist_filename,\
                                               group_by_year = 5)
    #2012
    age_dist_filename = os.path.join('age_dist_2012.dat')
    comp_age_dist2012 = get_age_dist_from_datfile(data_directory, 
                                                  age_dist_filename,\
                                               group_by_year = 5)

    #2022
    age_dist_filename = os.path.join('age_dist_2022.dat')
    comp_age_dist2022 = get_age_dist_from_datfile(data_directory, 
                                                  age_dist_filename,\
                                               group_by_year = 5)
    #a l
    legend_labels2002 = ["Simulation", "ABS Data", "2002"]
    legend_labels2012 = ["Simulation", "ABS Data", "2012"]
    legend_labels2022 = ["Simulation","ABS Data", "2022"]
    
    figsize=(7,3.5)#no in x axis, no in yaxis
    
    fig, ax = plt.subplots(figsize=figsize)
    ax =  plot_age_dist_multi(ax,age_dist2002, comp_age_dist2002,\
                years=years, legend_labels = legend_labels2002)
    
    saving_directory = output_directory
    fig.savefig(os.path.join(saving_directory, 
                'age_dist_2002.tiff'),
                bbox_inches = "tight",dpi=300)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax =  plot_age_dist_multi(ax,age_dist2012, comp_age_dist2012,\
                years=years, legend_labels = legend_labels2012)
    
    fig.savefig(os.path.join(saving_directory, 
               'age_dist_2012.tiff'),
               bbox_inches = "tight",dpi=300)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax =  plot_age_dist_multi(ax,age_dist2022, comp_age_dist2022,\
                years=years, legend_labels = legend_labels2022)
    
    fig.savefig(os.path.join(saving_directory, 
               'age_dist_2022.tiff'),
               bbox_inches = "tight",dpi=300)
    