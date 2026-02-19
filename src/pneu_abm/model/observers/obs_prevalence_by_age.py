"""
Observer object for carriage percentages for circulating strains.
"""


from .obs_base import Observer
import tables as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl

class PrevalenceByAgeObserver(Observer):
    def __init__(self, h5file, vaccines, interval):
        
        self.interval = interval
        self.no_age_groups = 101
            
        
        desc = {'t': tb.UInt32Col(pos=0),
                "total_inds_per_age_group": tb.UInt32Col(pos=1,
                                        shape=(self.no_age_groups)),
                "total_infecteds_per_age_group": tb.UInt32Col(pos=2,
                                        shape=(self.no_age_groups)), 
                "total_infections_per_age_group": tb.UInt32Col(pos=3,
                                        shape=(self.no_age_groups))
                    
                    }
                        
                       # tb.ComplexCol(pos=2,itemsize = 32)}
                    
        super(PrevalenceByAgeObserver, self).__init__(h5file =h5file, \
                            label = 'prevalencebyage', description = desc,
                            title = 'Prevalence By Age Observer')
        

    def update(self, t, pop, **kwargs):
        
        
        
        """popsize_age_groups = pl.DataFrame([
            pl.Series("age", list(range(self.no_age_groups))),
            pl.Series("count", [0] * self.no_age_groups),
            ]).update(
            pop.I.group_by("age").agg(pl.count()),
            on = "age", how= "left" ).sort("age")"""
        age_groups = (
            pl.DataFrame([
                pl.Series("age", list(range(self.no_age_groups))),
                pl.Series("count", [0] * self.no_age_groups),
                pl.Series("no_infecteds", [0] * self.no_age_groups),
                pl.Series("no_infections", [0] * self.no_age_groups),
            
            ])
            .update(
               pop.I.group_by("age").agg(pl.count(),
               pl.col("no_of_strains").cast(bool).sum().alias("no_infecteds"),
               pl.col("no_of_strains").sum().alias("no_infections")),
              on = "age", how= "left" )
            .sort("age")
        )
        self.row['t'] = t
        
        self.row["total_inds_per_age_group"] = \
                                        age_groups["count"].to_list()
        
        self.row["total_infecteds_per_age_group"] = \
                                        age_groups["no_infecteds"].to_list()
        
        self.row["total_infections_per_age_group"] = \
                                        age_groups["no_infections"].to_list()
        
        
    
        
        
        
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

    
            
    