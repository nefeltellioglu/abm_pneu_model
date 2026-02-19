import os

from random import Random
from itertools import product


class ParamComboIt(object):
    """
    An iterator object for parameter sweeps.  Generates dictionaries containing
    unique combinations of parameter values and random seeds.

    :param p: The simulation parameters
    :type p: dict
    :param p_values: A list of dictionaries containing parameters and values to sweep over
    :type p_values: list
    """

    def __init__(self, p, p_values, start_seed_index = 0):
        self.p = p
        # create a RNG for seeds (we will use same seeds for each param combo)
        pop_seed_rng = Random() if p['random_seed'] else Random(p['pop_seed'] + 
                                                        start_seed_index * 99)
        transmission_seed_rng = Random() if p['random_seed'] else Random(
                               p['transmission_seed'] +  start_seed_index * 99)
        disease_seed_rng = Random() if p['random_seed'] else Random(
                                    p['disease_seed'] + start_seed_index * 99)
        vaccine_seed_rng = Random() if p['random_seed'] else Random(
                                    p['vaccine_seed'] + start_seed_index * 99)
        self.pop_seeds = [pop_seed_rng.randint(0, 99999999) 
                          for _ in range(p['num_runs'])]
        self.transmission_seeds = [transmission_seed_rng.randint(0, 99999999)
                                   for _ in range(p['num_runs'])]
        self.disease_seeds = [disease_seed_rng.randint(0, 99999999) 
                              for _ in range(p['num_runs'])]
        self.vaccine_seeds = [vaccine_seed_rng.randint(0, 99999999) 
                              for _ in range(p['num_runs'])]

        self.base_prefix = p['prefix']
        self.p_values = p_values
        # generate all necessary parameter combinations (indexed by sample_ID)
        self.combos = [a for a in product(*[list(range(len(x['values']))) 
                                            for x in p_values])]

        self.combo_index = 0
        self.start_seed_index = start_seed_index
        self.seed_index = self.start_seed_index
        
    def print_combos(self):
        """
        Print all parameter combinations.
        """

        for cur_combo in self.combos:
            print(("".join(['{0:>10}'.format(self.p_values[i]['values'][x])
                           for i, x in enumerate(cur_combo)])))

    def __iter__(self):
        return self

    def __next__(self):
        """
        Get the next parameter combination object.

        :returns:parameter combination object (dictionary).
        """

        if self.combo_index == len(self.combos):
            raise StopIteration
        else:
            # make a local copy of params to modify
            p = self.p.copy()
            # set parameter values for this combination
            cur_combo = self.combos[self.combo_index]
            output_dir = "params_"
            param_strings = []
            for i, cur_p_name in enumerate([x['name'] for x in self.p_values]):
                p[cur_p_name] = self.p_values[i]['values'][cur_combo[i]]
                param_strings.append("%s=%s" % (cur_p_name,
                                    self.p_values[i]['values'][cur_combo[i]]))
            output_dir += "_".join(param_strings)
            p['prefix'] = os.path.join(self.base_prefix, output_dir)
            # only use seed directories if running more than one seed
            if len(self.pop_seeds) > 1:
                p['prefix'] = os.path.join(p['prefix'], 
                                           'seed_%d' % self.seed_index)
                p['pop_seed'] = self.pop_seeds[
                                self.seed_index - self.start_seed_index]
                p['transmission_seed'] = self.transmission_seeds[
                                    self.seed_index - self.start_seed_index]
                p['disease_seed'] = self.disease_seeds[
                                    self.seed_index - self.start_seed_index]
                p['vaccine_seed'] = self.vaccine_seeds[
                                    self.seed_index - self.start_seed_index]

            
            if (self.seed_index - self.start_seed_index) < len(self.pop_seeds) - 1:
                # print "incrementing seed"
                self.seed_index += 1
            else:
                if self.combo_index < len(self.combos):
                    # print "incrementing combo"
                    self.combo_index += 1
                    # print "resetting seed"
                    self.seed_index = self.start_seed_index

            return p
