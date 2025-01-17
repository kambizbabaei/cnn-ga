"""
number of conv/pool
                    ----add conv/pool
                    ----remove conv/pool
properties of conv
                    ----kernel size (not in your code, but could be extended)
                    ----pooling type
                    ----in/out channels
                    ----groups (ADDED for grouped conv)

for each individual, use probabilities to decide whether to mutate. If so, 
pick from 4 mutation types: add / remove / channels / pool type. 
(We also add a sub-step to possibly mutate the 'groups' param in conv.)
"""
import random
import numpy as np
import copy
from utils import StatusUpdateTool, Utils

class CrossoverAndMutation(object):
    def __init__(self, prob_crossover, prob_mutation, _log, individuals, _params=None):
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.individuals = individuals
        self.params = _params  # storing extra info if needed, e.g. gen_no
        self.log = _log
        self.offspring = []

    def process(self):
        crossover = Crossover(self.individuals, self.prob_crossover, self.log)
        offspring = crossover.do_crossover()
        self.offspring = offspring
        Utils.save_population_after_crossover(self.individuals_to_string(), self.params['gen_no'])

        mutation = Mutation(self.offspring, self.prob_mutation, self.log)
        mutation.do_mutation()

        for i, indi in enumerate(self.offspring):
            indi_no = 'indi%02d%02d' % (self.params['gen_no'], i)
            indi.id = indi_no

        Utils.save_population_after_mutation(self.individuals_to_string(), self.params['gen_no'])
        return offspring

    def individuals_to_string(self):
        _str = []
        for ind in self.offspring:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)


class Crossover(object):
    def __init__(self, individuals, prob_, _log):
        self.individuals = individuals
        self.prob = prob_
        self.log = _log
        # For counting how many pool layers are permissible
        self.pool_limit = StatusUpdateTool.get_pool_limit()[1]

    def _choose_one_parent(self):
        count_ = len(self.individuals)
        idx1 = int(np.floor(np.random.random() * count_))
        idx2 = int(np.floor(np.random.random() * count_))
        while idx2 == idx1:
            idx2 = int(np.floor(np.random.random() * count_))

        # Pick the one with higher accuracy
        if self.individuals[idx1].acc > self.individuals[idx2].acc:
            return idx1
        else:
            return idx2

    def _choose_two_diff_parents(self):
        idx1 = self._choose_one_parent()
        idx2 = self._choose_one_parent()
        while idx2 == idx1:
            idx2 = self._choose_one_parent()
        assert idx1 < len(self.individuals)
        assert idx2 < len(self.individuals)
        return idx1, idx2

    def _calculate_pool_numbers(self, parent1, parent2):
        """
        Calculates the number of pooling layers before and after crossover
        positions to ensure pool limits aren't exceeded.
        """
        t1, t2 = 0, 0
        for unit in parent1.units:
            if unit.type == 2:  # PoolUnit
                t1 += 1
        for unit in parent2.units:
            if unit.type == 2:
                t2 += 1

        len1, len2 = len(parent1.units), len(parent2.units)
        pos1 = int(np.floor(np.random.random() * len1))
        pos2 = int(np.floor(np.random.random() * len2))
        assert pos1 < len1
        assert pos2 < len2

        p1_left = sum(1 for i in range(0, pos1) if parent1.units[i].type == 2)
        p1_right = sum(1 for i in range(pos1, len1) if parent1.units[i].type == 2)
        p2_left = sum(1 for i in range(0, pos2) if parent2.units[i].type == 2)
        p2_right = sum(1 for i in range(pos2, len2) if parent2.units[i].type == 2)

        new_pool_number1 = p1_left + p2_right
        new_pool_number2 = p2_left + p1_right
        return pos1, pos2, new_pool_number1, new_pool_number2

    def do_crossover(self):
        """
        Main crossover routine:
        1) Chooses two different parents.
        2) Randomly decides whether to perform crossover based on 'self.prob'.
        3) Exchanges slices of layers between the two parents.
        4) Ensures the resulting offspring do not start with a pooling layer.
        5) Re-numbers the layers, adjusts 'in_channel' for the newly formed lists.
        6) Validates the resulting offspring. If invalid, discards (or reverts) them.
        """
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0}
        new_offspring_list = []

        # We'll pair up individuals in halves (assuming an even population).
        for _ in range(len(self.individuals) // 2):
            # 1) Pick two different parents
            idx1, idx2 = self._choose_two_diff_parents()

            # Keep a backup in case we revert due to invalid offspring
            parent1_backup = self.individuals[idx1]
            parent2_backup = self.individuals[idx2]

            # Make deep copies to actually modify
            parent1 = copy.deepcopy(parent1_backup)
            parent2 = copy.deepcopy(parent2_backup)

            # 2) Decide if we do crossover
            p_ = random.random()
            if p_ < self.prob:
                _stat_param['offspring_new'] += 2

                # 3) Exchange slices, ensuring we don't start with a pool layer
                first_begin_is_pool, second_begin_is_pool = True, True
                while first_begin_is_pool or second_begin_is_pool:
                    pos1, pos2, pool_len1, pool_len2 = self._calculate_pool_numbers(parent1, parent2)
                    try_count = 1
                    while pool_len1 > self.pool_limit or pool_len2 > self.pool_limit:
                        pos1, pos2, pool_len1, pool_len2 = self._calculate_pool_numbers(parent1, parent2)
                        try_count += 1
                        self.log.warn(f'The {try_count}-th try to find the position for crossover within pool limit.')

                    self.log.info(f'Position {pos1} for {parent1.id}, position {pos2} for {parent2.id}')

                    # Slicing units
                    unit_list1 = parent1.units[:pos1] + parent2.units[pos2:]
                    unit_list2 = parent2.units[:pos2] + parent1.units[pos1:]

                    first_begin_is_pool = (unit_list1 and unit_list1[0].type == 2)
                    second_begin_is_pool = (unit_list2 and unit_list2[0].type == 2)
                    if first_begin_is_pool:
                        self.log.warn('Crossovered individual#1 starts with a pooling layer, redoing pos selection...')
                    if second_begin_is_pool:
                        self.log.warn('Crossovered individual#2 starts with a pooling layer, redoing pos selection...')

                # 4) Re-number the layers in each new list
                for i, unit in enumerate(unit_list1):
                    unit.number = i
                for i, unit in enumerate(unit_list2):
                    unit.number = i

                # 5) Re-adjust the 'in_channel' in newly formed lists (for both conv type=1,3)
                #    First for unit_list1
                if pos1 == 0:
                    last_output_from_list1 = StatusUpdateTool.get_input_channel()
                else:
                    # find the last conv or grouped conv before pos1
                    last_output_from_list1 = 0
                    for i_ in range(pos1 - 1, -1, -1):
                        if unit_list1[i_].type in [1, 3]:
                            last_output_from_list1 = unit_list1[i_].out_channel
                            break

                for j in range(pos1, len(unit_list1)):
                    if unit_list1[j].type in [1, 3]:
                        unit_list1[j].in_channel = last_output_from_list1
                        break
                self.log.info(f'Changed the input channel in unit_list1 after crossover at pos1={pos1} -> {last_output_from_list1}')

                #    Then for unit_list2
                if pos2 == 0:
                    last_output_from_list2 = StatusUpdateTool.get_input_channel()
                else:
                    last_output_from_list2 = 0
                    for i_ in range(pos2 - 1, -1, -1):
                        if unit_list2[i_].type in [1, 3]:
                            last_output_from_list2 = unit_list2[i_].out_channel
                            break

                for j in range(pos2, len(unit_list2)):
                    if unit_list2[j].type in [1, 3]:
                        unit_list2[j].in_channel = last_output_from_list2
                        break
                self.log.info(f'Changed the input channel in unit_list2 after crossover at pos2={pos2} -> {last_output_from_list2}')

                # Assign back the new lists
                parent1.units = unit_list1
                parent2.units = unit_list2

                # 6) Post-check with validate_layers()
                if not parent1.validate_layers():
                    self.log.info(f'Invalid offspring1 {parent1.id} after crossover, discarding or reverting.')
                    parent1 = copy.deepcopy(parent1_backup)

                if not parent2.validate_layers():
                    self.log.info(f'Invalid offspring2 {parent2.id} after crossover, discarding or reverting.')
                    parent2 = copy.deepcopy(parent2_backup)

                parent1.reset_acc()
                parent2.reset_acc()
                new_offspring_list.append(parent1)
                new_offspring_list.append(parent2)

            else:
                # no crossover, just copy the parents as is
                _stat_param['offspring_from_parent'] += 2
                new_offspring_list.append(parent1)
                new_offspring_list.append(parent2)

        self.log.info('CROSSOVER-%d offspring are generated, new:%d, others:%d'
                      % (len(new_offspring_list),
                         _stat_param['offspring_new'],
                         _stat_param['offspring_from_parent']))
        return new_offspring_list

class Mutation(object):
    def __init__(self, individuals, prob_, _log):
        self.individuals = individuals
        self.prob = prob_
        self.log = _log

    def do_mutation(self):
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0,
                       'ADD': 0, 'REMOVE': 0, 'CHANNEL': 0, 'POOLING_TYPE': 0}
        mutation_probs = StatusUpdateTool.get_mutation_probs_for_each()  # e.g. [0.7, 0.1, 0.1, 0.1]

        for indi in self.individuals:
            p_ = random.random()
            if p_ < self.prob:
                _stat_param['offspring_new'] += 1
                mutation_type = self.select_mutation_type(mutation_probs)
                backup_units = copy.deepcopy(indi.units)
                
                if mutation_type == 0:
                    _stat_param['ADD'] += 1
                    self.do_add_unit_mutation(indi)
                elif mutation_type == 1:
                    _stat_param['REMOVE'] += 1
                    self.do_remove_unit_mutation(indi)
                elif mutation_type == 2:
                    _stat_param['CHANNEL'] += 1
                    self.do_modify_conv_mutation(indi)
                elif mutation_type == 3:
                    _stat_param['POOLING_TYPE'] += 1
                    self.do_modify_pooling_type_mutation(indi)
                else:
                    raise TypeError(f'Error mutation type :{mutation_type}, valid range:0-3')

                # After the chosen mutation, fix channel mismatches by adjusting only next-layer in_channel
                self.fix_mismatches_inchannel_only(indi)

                # Validate layers. If invalid, revert
                if not indi.validate_layers():
                    self.log.info(f'Invalid architecture after mutation for {indi.id}, reverting...')
                    indi.units = backup_units
                    indi.reset_acc()
                else:
                    indi.reset_acc()
            else:
                _stat_param['offspring_from_parent'] += 1

        self.log.info(
            'MUTATION-mutated:%d [ADD:%2d, REMOVE:%2d, CHANNEL:%2d, POOL:%2d], no_change:%d'
            % (_stat_param['offspring_new'], _stat_param['ADD'],
               _stat_param['REMOVE'], _stat_param['CHANNEL'],
               _stat_param['POOLING_TYPE'], _stat_param['offspring_from_parent'])
        )

    def do_add_unit_mutation(self, indi):
        self.log.info('Do the ADD mutation for indi:%s' % (indi.id))
        if len(indi.units) >= indi.max_len:
            self.log.info(f'Already reached max length ({indi.max_len}), skip add.')
            return

        mutation_position = int(np.floor(np.random.random() * len(indi.units)))
        self.log.info(f'Mutation position (add) at {mutation_position}')

        # decide whether to add a conv or a pool
        u_ = random.random()
        type_ = 1 if u_ < 0.5 else 2
        self.log.info('A %s unit is added (u_=%.2f)' % ('CONV' if type_ == 1 else 'POOLING', u_))

        if type_ == 2:
            num_exist_pool_units = sum([1 for u in indi.units if u.type == 2])
            if num_exist_pool_units > StatusUpdateTool.get_pool_limit()[1] - 1:
                type_ = 1
                self.log.info('Change to CONV because #pool > pool_limit')

        if type_ == 2:
            # Add a pool unit
            add_unit = indi.init_a_pool(mutation_position + 1, _max_or_avg=None)
        else:
            # Add a conv unit
            _in_channel = indi.image_channel  # fallback if none found
            for i in range(mutation_position, -1, -1):
                if indi.units[i].type in [1, 3]:
                    _in_channel = indi.units[i].out_channel
                    break
            # We'll pass is_first_layar=False because it's presumably in the middle
            add_unit = indi.init_a_conv(
                mutation_position + 1,
                _in_channel=_in_channel,
                _out_channel=None,
                is_first_layar=False
            )
            # Adjust the next layer's input channel to match the newly added layer's out_channel
            for i in range(mutation_position + 1, len(indi.units)):
                if indi.units[i].type in [1, 3]:
                    indi.units[i].in_channel = add_unit.out_channel
                    break

        new_unit_list = []
        for i in range(mutation_position + 1):
            new_unit_list.append(indi.units[i])
        new_unit_list.append(add_unit)
        for i in range(mutation_position + 1, len(indi.units)):
            unit = indi.units[i]
            unit.number += 1
            new_unit_list.append(unit)

        indi.number_id += 1
        indi.units = new_unit_list
        indi.reset_acc()

    def do_remove_unit_mutation(self, indi):
        self.log.info(f'Do the REMOVE mutation for indi:{indi.id}')
        if len(indi.units) > 1:
            mutation_position = int(np.floor(np.random.random() * (len(indi.units) - 1))) + 1
            self.log.info(f'Mutation position (remove) at {mutation_position}')

            if indi.units[mutation_position].type in [1, 3]:
                # connect the out_channel to the previous conv's out_channel if possible
                removed_out = indi.units[mutation_position].out_channel
                for i in range(mutation_position - 1, -1, -1):
                    if indi.units[i].type in [1, 3]:
                        indi.units[i].out_channel = removed_out
                        break

            new_unit_list = []
            for i in range(mutation_position):
                new_unit_list.append(indi.units[i])
            for i in range(mutation_position + 1, len(indi.units)):
                unit = indi.units[i]
                unit.number -= 1
                new_unit_list.append(unit)

            indi.number_id -= 1
            indi.units = new_unit_list
            indi.reset_acc()
        else:
            self.log.warn('REMOVE mutation not possible, only one unit remains.')

    def do_modify_conv_mutation(self, indi):
        self.log.info(f'Do the CHANNEL mutation for indi:{indi.id}')
        conv_index_list = [i for i, u in enumerate(indi.units) if u.type in [1, 3]]
        if not conv_index_list:
            self.log.warn('No CONV or GroupedBlock unit exist, skip conv mutation.')
            return

        selected_index = int(np.floor(np.random.random() * len(conv_index_list)))
        conv_idx = conv_index_list[selected_index]
        self.log.info(f'Mutation on conv index {conv_idx} (global unit idx)')

        channel_list = StatusUpdateTool.get_output_channel()

        # Possibly change groups if it's a grouped conv
        change_groups_chance = 0.4
        if indi.units[conv_idx].type == 3 and random.random() < change_groups_chance:
            old_groups = indi.units[conv_idx].groups
            in_ch = indi.units[conv_idx].in_channel
            out_ch = indi.units[conv_idx].out_channel
            possible_groups = [g for g in indi.groups_count if (in_ch % g == 0 and out_ch % g == 0)]
            if possible_groups:
                new_groups = np.random.choice(possible_groups)
                if new_groups != old_groups:
                    indi.units[conv_idx].groups = new_groups
                    self.log.info(f'Changed groups from {old_groups} to {new_groups} at unit idx={conv_idx}')
                    indi.reset_acc()

        # Step 2: Possibly change in_channel/out_channel
        old_in = indi.units[conv_idx].in_channel
        new_in = random.choice(channel_list)
        if new_in != old_in:
            if conv_idx > 0:
                self.log.info(f'Conv idx={conv_idx} changes in_channel from {old_in} to {new_in}')
                indi.units[conv_idx].in_channel = new_in
                prev_conv_idx = -1
                for pidx in range(conv_idx - 1, -1, -1):
                    if indi.units[pidx].type in [1, 3]:
                        prev_conv_idx = pidx
                        break
                if prev_conv_idx >= 0:
                    old_out = indi.units[prev_conv_idx].out_channel
                    indi.units[prev_conv_idx].out_channel = new_in
                    self.log.info(f'Also changed out_channel of unit idx={prev_conv_idx} '
                                  f'from {old_out} to {new_in}')
                indi.reset_acc()
            else:
                self.log.warn('Mutation tries to change in_channel of the first conv, ignoring.')

        old_out = indi.units[conv_idx].out_channel
        new_out = random.choice(channel_list)
        if new_out != old_out:
            self.log.info(f'Conv idx={conv_idx} changes out_channel from {old_out} to {new_out}')
            indi.units[conv_idx].out_channel = new_out
            next_conv_idx = -1
            for nxt in range(conv_idx + 1, len(indi.units)):
                if indi.units[nxt].type in [1, 3]:
                    next_conv_idx = nxt
                    break
            if next_conv_idx >= 0:
                old_next_in = indi.units[next_conv_idx].in_channel
                indi.units[next_conv_idx].in_channel = new_out
                self.log.info(f'Due to above, changed in_channel of unit idx={next_conv_idx} '
                              f'from {old_next_in} to {new_out}')
            indi.reset_acc()

    def do_modify_pooling_type_mutation(self, indi):
        self.log.info(f'Do the POOLING TYPE mutation for indi:{indi.id}')
        pool_list_index = [i for i, unit in enumerate(indi.units) if unit.type == 2]
        if not pool_list_index:
            self.log.warn('No POOL unit exist, skip pool mutation.')
            return

        selected_index = int(np.floor(np.random.random() * len(pool_list_index)))
        self.log.info(f'Mutation on pool index {selected_index}')
        if indi.units[pool_list_index[selected_index]].max_or_avg > 0.5:
            indi.units[pool_list_index[selected_index]].max_or_avg = 0.2
            self.log.info('Pool changes from avg to max')
        else:
            indi.units[pool_list_index[selected_index]].max_or_avg = 0.8
            self.log.info('Pool changes from max to avg')
        indi.reset_acc()

    def select_mutation_type(self, _a):
        """
        Weighted random selection among the 4 mutation types:
        0) ADD
        1) REMOVE
        2) CHANNEL
        3) POOL
        """
        a = np.asarray(_a)
        k = 1
        idx = np.argsort(a)[::-1]  # descending order
        sort_a = a[idx]
        sum_a = np.sum(a).astype(np.float32)
        selected_index = []
        for _ in range(k):
            u = np.random.rand() * sum_a
            acc = 0
            for i in range(sort_a.shape[0]):
                acc += sort_a[i]
                if acc > u:
                    selected_index.append(idx[i])
                    break
        return selected_index[0]

    #############################################
    # ADDED: fix_mismatch_inchannel_only method #
    #############################################

    def fix_mismatches_inchannel_only(self, indi):
        i = 0
        while i < len(indi.units) - 1:
            current_layer = indi.units[i]
            next_layer   = indi.units[i + 1]

            # 1) Determine current_layer.out_channel
            if current_layer.type in [1, 3]:  # BasicBlock or GroupedPointwiseBlock
                current_out = current_layer.out_channel
            elif current_layer.type == 2:     # Pooling doesn't change channels
                current_out = indi._get_last_known_channel(i)
            else:
                current_out = indi.image_channel  # fallback

            # 2) If next_layer is a conv or grouped conv, compare in_channel
            if next_layer.type in [1, 3]:
                if next_layer.in_channel != current_out:
                    old_in = next_layer.in_channel
                    self.log.info(f'Fix mismatch: layer {i} out_channel={current_out} vs next in_channel={old_in}. '
                                  f'Setting next_layer.in_channel={current_out}.')
                    # Check group constraints if next_layer is type=3 or uses groups
                    if next_layer.type == 3:
                        g_ = next_layer.groups
                        # If not divisible, we do the fix anyway, and rely on validate_layers() to catch it
                        if (current_out % g_ != 0) or (next_layer.out_channel % g_ != 0):
                            self.log.warn(f'Potential group mismatch in next layer {i+1}, leftover check in validate.')
                    elif next_layer.type == 1 and hasattr(next_layer, 'groups'):
                        g_ = next_layer.groups
                        if g_ > 1 and (current_out % g_ != 0):
                            self.log.warn(f'Potential group mismatch in next layer {i+1}, leftover check in validate.')
                    # Actually fix it
                    next_layer.in_channel = current_out
            i += 1

if __name__ == '__main__':
    m = Mutation(None, None, None)
    m.do_mutation()
