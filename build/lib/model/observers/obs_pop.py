#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for observers objects and
Observer for demographic (population) information.
"""

import tables as tb
import numpy as np



def age_dist(I, num_bins=101, max_age=101, isdensity=True):
    """
    Return the age distribution of a population.

    :param num_bins: the number of bins to group the population into.
    :type num_bins: int
    :param max_age: the maximum possible age.
    :type max_age: int
    :param isdensity: return proportions if `True`, otherwise counts.
    :type isdensity: bool
    :returns: a tuple containing a list of values and a list of bin edges.
    """

    ages = I['age'] 
    return np.histogram(ages, bins=num_bins, range=(0, max_age),
                        density= isdensity)


"""
Base class for observers objects.
"""

class Observer(object):

    def __init__(self, h5file, label, description, title):
        self.label = label
        self.h5file = h5file
        self.exists = False
        self.data = None
        self.row = None
        if self.h5file.mode in ['w', 'a']:
            self.create_storage(description, title)
        else:
            self.load_storage()

    def create_storage(self, description, title):
        if '/%s' % self.label not in self.h5file:
            group = self.h5file.create_group('/', self.label, title)
        filters = tb.Filters(complevel=9)   # TODO: investigate BLOSC (?)

        if description:
            self.data = self.h5file.create_table(group, 'base_data', description, filters=filters)
            self.row = self.data.row
        self.exists = True
#    __create_storage = create_storage

    def load_storage(self):
        if self.h5file.__contains__('/%s' % self.label):
            if 'base_data' in self.h5file.get_node(self.h5file.root, self.label):
                self.data = self.h5file.get_node(self.h5file.root, self.label).base_data
                self.row = self.data.row
            self.exists = True
        else:
            self.exists = False
#    __load_storage = load_storage

    def update(self, t, **kwargs):
        pass



class PopulationObserver(Observer):
    """
    Observer class which only saves the pop age distribution once every year.
    FIX: Age mean recorded as 0; it needs to be updated.
    Although age mean can be calculated from age distribution easily.
    """
    def __init__(self, h5file, interval=364):
        self.interval = interval
        self.age_dists = None
        
        desc = {
            't': tb.UInt32Col(),
            'pop_size': tb.UInt32Col(),
            'age_mean': tb.Float32Col()
        }
        super(PopulationObserver, self).__init__(h5file, 'population', desc, 'Population Observer')

    def create_storage(self, description, title):
        """
        Called when initialised with a h5file opened in 'w'rite mode.
        Adds tables for storage of data (plus local links)
        """

        Observer.create_storage(self, description, title)
        group = self.h5file.get_node('/', 'population')
        self.age_dists = self.h5file.create_table(
            group, 'age_dists',
            {'t': tb.UInt32Col(), 'dist': tb.UInt32Col(shape=(101))}, 'Age Distributions')
        

    def load_storage(self):
        """
        Called when initialised with a h5file opened in 'r'ead or 'a'ppend mode.
        Creates local links to existing tables.
        """

        Observer.load_storage(self)
        self.age_dists = self.h5file.root.cases.age_dists
        

    def update(self, t, pop, **kwargs):
        if t % self.interval > 0:
            return

        self.age_dists.row['t'] = t
        self.age_dists.row['dist'] = age_dist(pop.I, isdensity=False)[0]
        self.age_dists.row.append()
        
        # self.fam_types.append(self.P.sum_hh_stats_group())

        self.row['t'] = t
        self.row['pop_size'] = len(pop.I)
        
        self.row.append()
        self.h5file.flush()
