"""
Observer object for carriage percentages for circulating strains.
"""


from .obs_base import Observer
import tables as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl

class VaccinationDeliveredObserver(Observer):
    def __init__(self, h5file, vaccines):
        #interval is in time ticks not days
        self.state_labels = ['t', 'vaccine_type', \
                             "dose", "age", "on_time"]
                         #' '.join(list(np.sort(strains)))
                         #] #+ strains.to_list()
        self.vaccines = vaccines
        
        desc = {'t': tb.UInt32Col(pos=0),
                    'vaccine_type': tb.StringCol(pos=1, itemsize = 30),
                    'no_delivered': tb.UInt32Col(pos=2),
                    'total_inds': tb.UInt32Col(pos=3),
                    
                    }              
        super(VaccinationDeliveredObserver, self).__init__(h5file =h5file, \
                        label = 'vaccination_delivered', description = desc,
                            title = 'Vaccination Delivered Observer')
        

    
    def update(self, t, day, pop, **kwargs):
        
            
        #agg data
        vaccine_delivered_pop = (pop.I.filter(
            pl.col("vaccines")
            .struct.field("final_vaccine_time") == day
                                    ).select("vaccines")
                        .unnest("vaccines"))
        
        vaccine_delivered_pop = (vaccine_delivered_pop
                                .group_by("vaccine_type")
                             .agg(no_delivered = pl.count(),
                                 total_inds = pop.I.height))
        
                  
        if len(vaccine_delivered_pop):
            for row in vaccine_delivered_pop.rows(named=True):
                self.row['t'] = day
                self.row['vaccine_type'] = row['vaccine_type']
                self.row['no_delivered'] = row['no_delivered']
                self.row['total_inds'] = row["total_inds"]
                self.row.append()
                self.h5file.flush()
                

   

    
            
    