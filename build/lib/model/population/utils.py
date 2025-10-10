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
    
    
    #ages = [int(sample_table(age_probs, rng)[0]) for _ in range(pop_size)]
    period = 364 // t_per_year
    age_days = rng.choice(range(0,364,period), size = pop_size)
    
    #[rng.randint(0,t_per_year) * period\
    #                                    for i in range(pop_size)]
    
    
    
    return ages, age_days


def gen_age_structured_pop(t_per_year, pop, pop_size, age_probs, rng):
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
    
    pop.I, pop.next_id = pop.introduce_individuals(pop_size, 
                                            ages, age_days, rng)
    
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


