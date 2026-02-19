"""
Observer object for carriage percentages for circulating strains.
"""


from .obs_base import Observer
import tables as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl

class PrevalenceObserver(Observer):
    def __init__(self, h5file, strains):
        self.state_labels = ['t', 'infected_population', \
                             "strain_list"]
                         #' '.join(list(np.sort(strains)))
                         #] #+ strains.to_list()
        self.strains = pl.DataFrame({"strain_list": list(np.sort(strains)),
                              "counts": pl.Series([0] * len(strains),
                                                  dtype = pl.Float64)})
                              
            
        
        desc = {'t': tb.UInt32Col(pos=0),
                    'infected_population': tb.Float64Col(pos=1),
                    'infected_frac_0_2_year_olds': tb.Float64Col(pos=2),
                    'strain_list': tb.Float64Col(pos=3, shape=(len(strains))),
                    'infected_frac_65_100_year_olds': tb.Float64Col(pos=4),
                    'infected_frac_0_4_year_olds': tb.Float64Col(pos=5),
                    'infected_frac_5_17_year_olds': tb.Float64Col(pos=6),
                    'infected_frac_18_39_year_olds': tb.Float64Col(pos=7),
                    'infected_frac_0_10_year_olds': tb.Float64Col(pos=8),
                    'infected_frac_40_100_year_olds': tb.Float64Col(pos=9),
                    'infected_frac_40_64_year_olds': tb.Float64Col(pos=10)
                    #'strain_list_0_2mo': tb.Float64Col(pos=11, shape=(len(strains))),
                    #'strain_list_2_12mo': tb.Float64Col(pos=12, shape=(len(strains)))
                    }
                        
                       # tb.ComplexCol(pos=2,itemsize = 32)}
                    
        super(PrevalenceObserver, self).__init__(h5file =h5file, \
                            label = 'prevalence', description = desc,
                            title = 'Prevalence Observer')
        
    
    def update(self, t, pop, **kwargs):
        self.row['t'] = t
        
        #frac no of infected 
        self.row["infected_population"] = (pop.I["no_of_strains"]
                                           .cast(bool).sum()/pop.I.height)
        
        child_02_yearolds = pop.I.filter(pl.col("age") <= 2)
        adult_65_100_yearolds = pop.I.filter(pl.col("age") >= 65)
        
        child_04_yearolds = pop.I.filter(pl.col("age") < 5)
        child_010_yearolds = pop.I.filter(pl.col("age") <= 10)
        child_5_17_yearolds = pop.I.filter((pl.col("age") >= 5) & \
                                           (pl.col("age") <= 17))
        adult_18_39_yearolds = pop.I.filter((pl.col("age") >= 18) & \
                                           (pl.col("age") <= 39))
        adult_40_100_yearolds = pop.I.filter((pl.col("age") >= 40))
        adult_40_64_yearolds = pop.I.filter((pl.col("age") >= 18) & \
                                           (pl.col("age") <= 64))
        
        
        
        
        self.row["infected_frac_0_2_year_olds"] = (
                                child_02_yearolds["no_of_strains"]
                             .cast(bool).sum()/child_02_yearolds.height)
        self.row["infected_frac_65_100_year_olds"] = (
                                adult_65_100_yearolds["no_of_strains"]
                             .cast(bool).sum()/adult_65_100_yearolds.height)
        self.row["infected_frac_0_4_year_olds"] = (
                                child_04_yearolds["no_of_strains"]
                             .cast(bool).sum()/child_04_yearolds.height)
        self.row["infected_frac_5_17_year_olds"] = (
                                child_5_17_yearolds["no_of_strains"]
                             .cast(bool).sum()/child_5_17_yearolds.height)
        self.row["infected_frac_18_39_year_olds"] = (
                                adult_18_39_yearolds["no_of_strains"]
                             .cast(bool).sum()/adult_18_39_yearolds.height)
        self.row["infected_frac_0_10_year_olds"] = (
                                child_010_yearolds["no_of_strains"]
                             .cast(bool).sum()/child_010_yearolds.height)
        self.row["infected_frac_40_100_year_olds"] = (
                                adult_40_100_yearolds["no_of_strains"]
                             .cast(bool).sum()/adult_40_100_yearolds.height)
        self.row["infected_frac_40_64_year_olds"] = (
                                adult_40_64_yearolds["no_of_strains"]
                             .cast(bool).sum()/adult_40_64_yearolds.height)
        #frac no of infections
        #pop.I["no_of_strains"].sum()/pop.I.height
        
        #fraction of strains
        prev_data = (pop.I["strain_list"].explode().value_counts()
                    .with_columns(counts = pl.col("count")/pop.I.height)
                     )
        
        self.row["strain_list"] = (self.strains
                                   .update(prev_data, on="strain_list")
                                   .sort("strain_list")["counts"].to_list())
        """
        
        pop_02mo = pop.I.filter((pl.col("age") == 0) & (pl.col("age_days") < 56))
        prev_data_0_2mo = (pop_02mo["strain_list"].explode().value_counts()
                    .with_columns(counts = pl.col("count")/pop_02mo.height)
                     )
        self.row["strain_list_0_2mo"] = (self.strains
                                   .update(prev_data_0_2mo, on="strain_list")
                                   .sort("strain_list")["counts"].to_list())
        
        pop_2_12mo = pop.I.filter((pl.col("age") == 0) & (pl.col("age_days") >= 56))
        prev_data_0_2mo = (pop_2_12mo["strain_list"].explode().value_counts()
                    .with_columns(counts = pl.col("count")/pop_2_12mo.height)
                     )
        self.row["strain_list_2_12mo"] = (self.strains
                                   .update(prev_data_0_2mo, on="strain_list")
                                   .sort("strain_list")["counts"].to_list())
        """
        
        self.row.append()
        self.h5file.flush()
#        if introduction:
#            self.introductions.append((introduction, t))

    def get_counts_by_state(self, label):
        return self.data.col(label)

    def get_props_by_state(self):
        """
        Return a dictionary mapping state labels to proportions.
        """
        sizes = np.sum([self.data.col(x) for x in self.state_labels[1:]],axis=0)
        return dict((x, self.data.col(x)/sizes.astype(float)) for x in self.state_labels[1:])

    def get_pop_sizes(self):
        return np.sum([self.data.col(x) for x in self.state_labels[1:]],axis=0)

    

    def get_first_fadeout_time(self, state_labels):
        timeseries = sum(self.get_counts_by_state(y) for y in state_labels)
        try:
            zeros = np.where(timeseries==0)[0]
            #print len(zeros)
            if len(zeros) > 0:
                first_zero = zeros[0]
            else:
                first_zero = len(timeseries)-1
        except Exception:
            print("first zero problem!")
        time = self.data[first_zero]['t']
        return time
    
    def get_max(self, state_label):
        timeseries = [x for x in self.get_counts_by_state(state_label)]
        return max(timeseries)

    def get_time_at_max(self, state_label):
        timeseries = [x for x in self.get_counts_by_state(state_label)]
        peak_inc = max(timeseries)
        index = timeseries.index(peak_inc)
        time = self.data[index]['t']
        return time

    ### Experimental peak detection stuff below here...

    
            
    