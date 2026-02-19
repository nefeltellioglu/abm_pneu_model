"""
Observer object for carriage percentages for circulating strains.
"""


from .obs_base import Observer
import tables as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl

class VaccinationObserver(Observer):
    def __init__(self, h5file, vaccines, interval):
        #interval is in time ticks not days
        self.state_labels = ['t', 'vaccine_type', \
                             "dose", "age", "on_time"]
                         #' '.join(list(np.sort(strains)))
                         #] #+ strains.to_list()
        self.vaccines = vaccines
        self.interval = interval
        self.observed_ages = [1,2,5] + list(range(65,101))
        self.observed_on_age_day = 0                  
        self.sumdata = pl.DataFrame({})
        self.observed_day = self.interval - 1
        #pl.Series("strain_list", list(np.sort(strains))))
        
        desc = {'t': tb.UInt32Col(pos=0),
                    'vaccine_type': tb.StringCol(pos=1, itemsize = 20),
                    'dose': tb.UInt32Col(pos=2),
                    'age': tb.UInt32Col(pos=3),
                    "on_time": tb.StringCol(pos=4, itemsize = 20),
                    "fraction": tb.Float64Col(pos=5),
                    'birth_cohort': tb.UInt32Col(pos=6),
                    }
                        
                       # tb.ComplexCol(pos=2,itemsize = 32)}
                    
        super(VaccinationObserver, self).__init__(h5file =h5file, \
                            label = 'vaccination', description = desc,
                            title = 'Vaccination Observer')
        

    def old_update(self, t, pop, **kwargs):
        
        
        year = t / self.interval
        #agg data
        target_group = (pop.I.filter(pl.col("age")
                                     .is_in(self.observed_ages) & \
                    (pl.col("age_days") ==  self.observed_on_age_day) #& \
                    #(pl.col("vaccines").struct.field("no_of_doses") > 0)
                                    ).select("age", "vaccines")
                        .unnest("vaccines"))
       
        total_inds = target_group.group_by(["age"]).agg(
            total_ind = pl.count()
            )
        target_group = target_group.join(
                        total_inds,
                        on=['age'],
                        how='left',
                        )                                      
                                      
        target_group = ((target_group.filter(pl.col("vaccine_type")
                                           .str.lengths() > 0))
                                .group_by(["age", "vaccine_type", 
                                              "no_of_doses","total_ind",
                               ((pl.col("on_time") > 0)
                               .alias("on_time_received"))
                             ]).agg(no_inds = pl.count(),
                                 fraction = \
                                  (pl.count()/pl.col("total_ind").first())
                                  .round(3)))
        if len(target_group):
            self.sumdata = pl.concat([self.sumdata, target_group]) 
            
            if t % self.interval == 0:
                #agg data
                total_inds = self.sumdata.group_by(["age"]).agg(
                    total_ind_age = pl.sum("total_ind")
                    )
                self.sumdata = self.sumdata.join(
                                total_inds,
                                on=['age'],
                                how='left',
                                )   
                
                self.sumdata = (self.sumdata.group_by(["age", "vaccine_type", 
                                                      "no_of_doses",
                                       "on_time_received"
                                     ]).agg(no_inds = pl.sum("no_inds"),
                                         fraction = \
                                 (pl.sum("no_inds")/pl.sum("total_ind"))
                                          .round(3)))
                for row in self.sumdata.rows(named=True):
                    self.row['t'] = year
                    self.row['vaccine_type'] = row['vaccine_type']
                    self.row['dose'] = row['no_of_doses']
                    self.row['birth_cohort'] = year - row["age"]
                    self.row['age'] = row["age"]
                    self.row['on_time'] = row["on_time_received"]
                    self.row['fraction'] = row["fraction"]
                    self.row.append()
                    self.h5file.flush()
                
                self.sumdata = pl.DataFrame({})
    
    def update(self, t, pop, **kwargs):
        
        if t % self.interval == self.observed_day:
            
            year = t / self.interval
            #agg data
            target_group = (pop.I.filter(pl.col("age")
                                         .is_in(self.observed_ages) #& \
                        
                        #(pl.col("age_days") ==  self.observed_on_age_day) #& \
                        #(pl.col("vaccines").struct.field("no_of_doses") > 0)
                                        ).select("age", "vaccines")
                            .unnest("vaccines"))
            adult_target_group = target_group.filter(pl.col("age") >= 65)
            adult_target_group1 = adult_target_group.filter(pl.col("age") < 75)
            adult_target_group2 = adult_target_group.filter(
                                                pl.col("age") >= 75)
            #if (adult_target_group1.filter(pl.col("no_of_doses") > 0).height):
            #    print("here")
                        
            adult_target_group1 = ((adult_target_group1.filter(
                                        pl.col("no_of_doses") > 0))
                                    .group_by(["vaccine_type", 
                                                  "no_of_doses",
                                   ((pl.col("on_time") > 0)
                                   .alias("on_time_received"))
                                 ]).agg(no_inds = pl.count(),
                                     fraction = \
                                      (pl.count()/adult_target_group1.height)
                                      .round(3))).with_columns(
                                        pl.lit(65).cast(pl.Int64).alias("age"),
                                          pl.lit(adult_target_group1.height)
                                        .cast(pl.UInt32).alias("total_ind"),)
            
            adult_target_group2 = ((adult_target_group2.filter(
                                        pl.col("no_of_doses") > 0))
                                     .group_by(["vaccine_type", 
                                                   "no_of_doses",
                                    ((pl.col("on_time") > 0)
                                    .alias("on_time_received"))
                                  ]).agg(no_inds = pl.count(),
                                      fraction = \
                                       (pl.count()/adult_target_group2.height)
                                       .round(3))).with_columns(
                                     pl.lit(75).cast(pl.Int64).alias("age"),
                                           pl.lit(adult_target_group2.height)
                                         .cast(pl.UInt32).alias("total_ind"),)
            
            
            target_group = target_group.filter(pl.col("age") < 65)
            total_inds = target_group.group_by(["age"]).agg(
                total_ind = pl.count()
                )
            target_group = target_group.join(
                            total_inds,
                            on=['age'],
                            how='left',
                            )                                      
                                          
            target_group = ((target_group.filter(pl.col("no_of_doses") > 0))
                                    .group_by(["age", "vaccine_type", 
                                                  "no_of_doses","total_ind",
                                   ((pl.col("on_time") > 0)
                                   .alias("on_time_received"))
                                 ]).agg(no_inds = pl.count(),
                                     fraction = \
                                      (pl.count()/pl.col("total_ind").first())
                                      .round(3)))
            #if adult_target_group1.height or adult_target_group2.height:
            target_group = pl.concat([target_group, adult_target_group1,
                                      adult_target_group2], rechunk = True,
                                         how= "diagonal")
                      
            if len(target_group):
                for row in target_group.rows(named=True):
                    self.row['t'] = year
                    self.row['vaccine_type'] = row['vaccine_type']
                    self.row['dose'] = row['no_of_doses']
                    if row["age"] in [65,75]:
                        self.row['birth_cohort'] = row["age"]
                    else:
                        self.row['birth_cohort'] = year - row["age"]
                    self.row['age'] = row["age"]
                    self.row['on_time'] = row["on_time_received"]
                    self.row['fraction'] = row["fraction"]
                    self.row.append()
                    self.h5file.flush()
                

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

    
            
    