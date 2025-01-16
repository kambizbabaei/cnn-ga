import numpy as np
import hashlib
import copy
import random

class Unit(object):
    def __init__(self, number):
        self.number = number

class ResUnit(Unit):
    def __init__(self, number, in_channel, out_channel, groups=1):  # added groups
        super().__init__(number)
        self.type = 1
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.groups = groups  # added groups parameter

class PoolUnit(Unit):
    def __init__(self, number, max_or_avg):
        super().__init__(number)
        self.type = 2
        self.max_or_avg = max_or_avg  # max_pool for < 0.5, avg_pool for >= 0.5

class GroupedPointwiseBlock(Unit):
    def __init__(self, number, in_channel, out_channel, groups=2):
        super().__init__(number)
        self.type = 3  # Added to denote this as a grouped pointwise block
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.groups = groups

class Individual(object):
    def __init__(self, params, indi_no):
        self.acc = -1.0
        self.id = indi_no
        self.number_id = 0
        self.min_conv = params['min_conv']
        self.max_conv = params['max_conv']
        self.min_pool = params['min_pool']
        self.max_pool = params['max_pool']
        self.max_len = params['max_len']
        self.image_channel = params['image_channel']
        self.output_channles = params['output_channel']
        self.groups_count = params['groups_count']
        self.group_block_percentage = params['group_block_percentage']
        self.units = []

    def reset_acc(self):
        self.acc = -1.0

    def initialize(self):
        num_conv = np.random.randint(self.min_conv, self.max_conv + 1)
        num_pool = np.random.randint(self.min_pool, self.max_pool + 1)
        availabel_positions = list(range(num_conv))
        np.random.shuffle(availabel_positions)
        select_positions = np.sort(availabel_positions[0:num_pool])

        all_positions = []
        for i in range(num_conv):
            all_positions.append(1)
            for j in select_positions:
                if j == i:
                    all_positions.append(2)
                    break

        input_channel = self.image_channel
        for i in all_positions:
            if i == 1:
                is_first_layar = len(self.units) == 0
                conv = self.init_a_conv(_number=None, _in_channel=input_channel, _out_channel=None,is_first_layar=is_first_layar)
                input_channel = conv.out_channel
                self.units.append(conv)
            elif i == 2:
                pool = self.init_a_pool(_number=None, _max_or_avg=None)
                self.units.append(pool)
                
    def calculate_new_size(self, current_size, conv_layer):
        # Calculate the new image size after a convolution (assuming padding=1, stride=1)
        # Convolution with kernel size 3x3 reduces the size by 2 (padding=1, stride=1)
        return current_size - 2

    def init_a_conv(self, _number, _in_channel, _out_channel,is_first_layar):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]
        groups = np.random.choice(self.groups_count)
        # Decide between a basic convolutional layer or a grouped pointwise block (with a probability)
        if(is_first_layar):
            groups = 1
        if np.random.rand() < self.group_block_percentage :
            block_type = 'grouped_pointwise'
            conv = GroupedPointwiseBlock(number, _in_channel, out_channel, groups=groups)
        else:
            block_type = 'basic'
            conv = ResUnit(number, _in_channel, out_channel,groups)

        return conv

    def init_a_pool(self, _number, _max_or_avg):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _max_or_avg:
            max_or_avg = _max_or_avg
        else:
            max_or_avg = np.random.rand()

        pool = PoolUnit(number, max_or_avg)
        return pool

    def uuid(self):
        """
        Generate a unique identifier for this individual by combining all units' details
        """
        _str = []
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:  # BasicBlock (ResUnit)
                _sub_str.append('conv')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'in:{unit.in_channel}')
                _sub_str.append(f'out:{unit.out_channel}')
            elif unit.type == 2:  # PoolUnit
                _sub_str.append('pool')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'type:{unit.max_or_avg}')
            elif unit.type == 3:  # GroupedPointwiseBlock
                _sub_str.append('grouped_pointwise')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'in:{unit.in_channel}')
                _sub_str.append(f'out:{unit.out_channel}')
                _sub_str.append(f'groups:{unit.groups}')

            _str.append(f'[{",".join(_sub_str)}]')
        _final_str_ = '-'.join(_str)
        _final_utf8_str_ = _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _final_str_

    def __str__(self):
        _str = [f'indi:{self.id}', f'Acc:{self.acc:.5f}']
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:  # BasicBlock (ResUnit)
                _sub_str.append('conv')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'in:{unit.in_channel}')
                _sub_str.append(f'out:{unit.out_channel}')
            elif unit.type == 2:  # PoolUnit
                _sub_str.append('pool')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'type:{unit.max_or_avg}')
            elif unit.type == 3:  # GroupedPointwiseBlock
                _sub_str.append('grouped_pointwise')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'in:{unit.in_channel}')
                _sub_str.append(f'out:{unit.out_channel}')
                _sub_str.append(f'groups:{unit.groups}')
            _str.append(f'[{",".join(_sub_str)}]')
        return '\n'.join(_str)


class Population(object):
    def __init__(self, params, gen_no):
        self.gen_no = gen_no
        self.number_id = 0  # for recording how many individuals have been generated
        self.pop_size = params['pop_size']
        self.params = params
        self.individuals = []

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = f'indi{self.gen_no:02d}{self.number_id:02d}'
            self.number_id += 1
            indi = Individual(self.params, indi_no)
            indi.initialize()
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = f'indi{self.gen_no:02d}{self.number_id:02d}'
            indi.id = indi_no
            self.number_id += 1
            indi.number_id = len(indi.units)
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)


# Example Testing Functions
def test_individual():
    params = {
        'min_conv': 3,
        'max_conv': 5,
        'min_pool': 1,
        'max_pool': 2,
        'max_len': 10,
        'image_channel': 3,
        'output_channel': [64, 128, 256, 512]
    }
    ind = Individual(params, 'indi00')
    ind.initialize()
    print(ind)
    print(ind.uuid())

def test_population():
    params = {
        'pop_size': 20,
        'min_conv': 3,
        'max_conv': 6,
        'min_pool': 1,
        'max_pool': 2,
        'max_len': 10,
        'image_channel': 3,
        'output_channel': [64, 128, 256, 512]
    }
    pop = Population(params, 0)
    pop.initialize()
    print(pop)

if __name__ == '__main__':
    test_individual()
    test_population()
