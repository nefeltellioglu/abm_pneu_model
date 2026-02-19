#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:00:35 2023

@author: ntellioglu
"""

import os
import sys
from math import exp, log
from collections import defaultdict

import numpy as np
import polars as pl
import itertools


####
def update_age_group(cm_yearly_ages, cm_max_age) -> pl.Expr:
    return (
            (pl.when(pl.col("age") >= cm_max_age)
            .then(cm_max_age/cm_yearly_ages - 1)
            .otherwise(pl.col("age") // cm_yearly_ages))
            .cast(pl.Int32)
            .alias("age_group")
        )

def will_live(age_series: pl.Series, prob_list: pl.Series, rng) -> pl.Series:
    """
    :param age_series: a pl series of agent ages
    :param prob_list: a pl series of death probability
    
    returns pl series of boolean values 1: alive, 0:dead
    """
    return pl.Series("alive",
        values=[rng.random() < prob_list[age][0] for age in age_series],
        dtype=pl.Boolean,
    )

# Population generation function #######################################
def death_cdf(death_prop):
    """Convert age-specific death rates to cumulative distribution function.

    Args:
        propn_die: Array of annual death probabilities by age.

    Returns:
        Cumulative death probability array for age-at-death sampling.
    """
    intervals = np.zeros(len(death_prop) + 1)
    for ix, propn in enumerate(death_prop):
        prev_propn = intervals[ix]
        intervals[ix + 1] = prev_propn + propn * (1 - prev_propn)
    return intervals

def pick_age_death_given_cdf(ages, cdf_bins, rng):
    """Sample age at death for individuals given current ages and mortality CDF.

    Args:
        ages: Current ages of individuals.
        cdf_bins: Cumulative death probability bins by age.
        rng: Random number generator for sampling.

    Returns:
        Array of sampled death ages, constrained to be at least current age.
    """
    ages = ages.astype(int)

    # Sample conditional CDF values
    u = rng.uniform(
        low=cdf_bins[ages],
        high=1.0,
        size=len(ages),
    )

    # Avoid u == 1.0 (important with flat CDF tail)
    u = np.minimum(u, np.nextafter(1.0, 0.0))

    # Inverse CDF lookup
    death_ages = np.searchsorted(cdf_bins, u, side="left")
    return death_ages

def gen_deaths(t_per_year, no_of_ind, 
                                  ages, age_days, death_rates, rng):
    
    ages = np.array(ages)
    age_days = np.array(age_days)
    days_alive = ages * 365 + age_days
    
    cdf_bins = death_cdf(death_rates)

    death_ages = np.zeros(no_of_ind)

    death_ages = pick_age_death_given_cdf(ages, cdf_bins, rng)
    # pick a random day into the age of death
    days_at_death = death_ages * 365 + rng.integers(0, 365, size=no_of_ind)
    # if that day ends up less than days_alive
    # (i.e. they are dying in their current age)
    # find a new days_at_death in the remaining days of their current age
    days_at_death = np.where(
        days_at_death > days_alive,
        days_at_death,
        days_alive + rng.integers(days_alive % 365, 365, size=no_of_ind),
    )

    return np.array(days_at_death, dtype=int)


def gen_ages(t_per_year, pop_size, age_probs, rng, isRandAgeDays = True):
    """
    Generates an age list with given age structure.

    :param pop_size: The number of individuals to generate.
    :type pop_size: int
    :param age_probs: A table mapping probabilities to age.
    :type age_probs: list
    :param isRandAgeDays: boolean, if agedays are randomly sampled or not
    :param rng: The random number generator to use.
    :type rng: :class:`random.Random`
    
    :returns a list of samples ages and a list of age days
    """
    
    ages = rng.choice(len(age_probs), p = age_probs, size = pop_size)
    
    period = 365 // t_per_year
    age_days = rng.choice(range(0,365,period), size = pop_size)
    
    return ages, age_days


def gen_age_structured_pop(t_per_year, pop, pop_size, age_probs,
                           death_rates, rng):
    """
    Generate a pl df representing a population
     of individuals with given age structure.

    :param pop: The population
    :type pop: Population
    :param pop_size: The number of individuals to generate.
    :type pop_size: int
    :param age_probs: A table mapping probabilities to age.
    :type age_probs: list
    :type cutoffs: tuple
    :param rng: The random number generator to use.
    :type rng: :class:`random.Random`
    """
    ages, age_days = gen_ages(t_per_year, pop_size, age_probs, rng)
    days_at_death = gen_deaths(t_per_year, pop_size, 
                                ages, age_days, 
                                death_rates, rng)
    
    pop.I, pop.next_id = pop.introduce_individuals(pop_size, 
                                            ages, age_days, days_at_death)
    
    return pop
      
def sample_table(table, rng):
    """
    Given a table of [p, x], sample and return event x with probability p
    """
    

    i = sample(list(zip(*table))[0], rng)
    #    print i
    return table[i][1]

def sample(probs, rng):
    """
    Returns i E [0, len(probs)-1] with probability probs[i]
    """

    x = rng.random();
    prob_sum = 0.0
    for i in range(0, len(probs)):
        prob_sum += probs[i]
        if prob_sum >= x:
            return i
    # In almost all cases, this loop should return before completing,
    # as sum(probs) == 1.0 (which is >= x).  If x is very close to 1.0 
    # however, rounding errors in summing probabilities may mean that     
    # x>sum(probs). Therefore, return final index if this occurs:
    return len(probs) - 1


def load_age_rates(fname):
    """
    Load age-dependent rates from file fname.

    File has the format:
    age rate_1 rate_2 rate_3 ... etc
    """

    t = []
    for l in open(fname, 'r'):
        if l[0] == '#': continue
        t.append([eval(x) for x in l.strip().split(' ')])
        
    return t


def create_path(path_name):
    """
    Create a path if it doesn't already exist.
    """

    if not os.path.exists(path_name):
        while True:
            try:
                os.makedirs(path_name)
                break
            except OSError as e:
                if e.errno != 17:
                    raise
                    # time.sleep might help here
                pass

                #if not os.path.exists(path_name):
                #    os.makedirs(path_name)



def load_probs(fname, sorted=False):
    """
    Return an arbitrary probability table loaded from file fname

    The table has the form:
    prob_0, [list of data_values_0]
    prob_1, [list of data_values_1]
    ...

    A check is made to ensure that probabilities sum to 1.0

    probabilities are sorted from most to least frequent for future efficiency
    """

    t = []
    for l in open(fname, 'r'):
        if l[0] == '#': continue
        line = l.strip().split(' ')
        #t.append([float(line[0]), line[1:]])
        t.append(float(line[0]))

    #if abs(sum(list(zip(*t))[0]) - 1.0) > 0.00001:
    if abs(sum(t) - 1.0):
        #stderr.write("Probs in file %s don't sum to 1.0") % fname;
        #exit(1)
        t[0] += 1.0 - sum(t)

    if sorted: t.sort(reverse=True)

    return t

def load_prob_list(fname):
    """
    Loads a sequence of probabilities/rates and returns them as a list.

    For use in specifying, e.g., time-varying marriage rates.

    File has the format:
    value_1
    value_2
    value_3
    ...
    """
    t = []
    for l in open(fname, 'r'):
        if l[0] == '#': continue
        line = l.strip().split(' ')
        r = eval(line[0])
        t.append(float(line[0]))
    return t


