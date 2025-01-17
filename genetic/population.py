import numpy as np
import hashlib
import copy
import random

class Unit(object):
    def __init__(self, number):
        self.number = number

class ResUnit(Unit):
    def __init__(self, number, in_channel, out_channel, groups=1):
        super().__init__(number)
        self.type = 1
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.groups = groups

class PoolUnit(Unit):
    def __init__(self, number, max_or_avg):
        super().__init__(number)
        self.type = 2
        self.max_or_avg = max_or_avg  # max_pool if < 0.5, avg_pool otherwise

class GroupedPointwiseBlock(Unit):
    def __init__(self, number, in_channel, out_channel, groups=2):
        super().__init__(number)
        self.type = 3
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
                conv = self.init_a_conv(
                    _number=None, 
                    _in_channel=input_channel,
                    _out_channel=None,
                    is_first_layar=is_first_layar
                )
                input_channel = conv.out_channel
                self.units.append(conv)
            elif i == 2:
                pool = self.init_a_pool(_number=None, _max_or_avg=None)
                self.units.append(pool)

    def init_a_conv(self, _number, _in_channel, _out_channel, is_first_layar):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]

        # Decide groups
        groups = np.random.choice(self.groups_count)
        # Force first layer to be groups=1 if that is your design choice
        if is_first_layar:
            groups = 1

        # Decide whether to build a grouped pointwise block
        if np.random.rand() < self.group_block_percentage:
            conv = GroupedPointwiseBlock(number, _in_channel, out_channel, groups=groups)
        else:
            conv = ResUnit(number, _in_channel, out_channel, groups)
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

    def validate_layers(self):
        """
        Checks if layers in this individual have consistent in/out channels
        and valid group configurations for grouped convs.
        """
        for i in range(len(self.units) - 1):
            current_layer = self.units[i]
            next_layer = self.units[i + 1]

            # Determine out_channel of current layer
            if current_layer.type == 1 or current_layer.type == 3:
                current_out = current_layer.out_channel
            elif current_layer.type == 2:
                # For a pool unit, we don't change channels. 
                # We must find the last known out_channel from a prior conv-like layer.
                current_out = self._get_last_known_channel(i)
            else:
                return False  # Unknown layer type

            # Determine in_channel of next layer if it's a conv or grouped conv
            if next_layer.type == 1 or next_layer.type == 3:
                next_in = next_layer.in_channel
                if current_out != next_in:
                    print(f"Channel mismatch at layers {i} -> {i+1}: {current_out} != {next_in}")
                    return False
            elif next_layer.type == 2:
                # Next layer is pooling, usually doesn't need to match channels exactly in 'in_channel' 
                # because pooling does not specify in_channel/out_channel in the same sense.
                continue
            else:
                return False  # Unknown layer type

        # Check group validity for GroupedPointwiseBlock
        for i, layer in enumerate(self.units):
            if layer.type == 3:  # GroupedPointwiseBlock
                if (layer.in_channel % layer.groups != 0) or (layer.out_channel % layer.groups != 0):
                    print(f"GroupedBlock channel/group mismatch at layer {i}")
                    return False

        return True

    def _get_last_known_channel(self, index):
        """
        Returns the last known out_channel from a conv or grouped block
        going backwards from 'index'.
        """
        for i in range(index, -1, -1):
            if self.units[i].type == 1 or self.units[i].type == 3:
                return self.units[i].out_channel
        return self.image_channel  # fallback if none is found

    def uuid(self):
        _str = []
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('conv')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'in:{unit.in_channel}')
                _sub_str.append(f'out:{unit.out_channel}')
            elif unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'type:{unit.max_or_avg}')
            elif unit.type == 3:
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
            if unit.type == 1:
                _sub_str.append('conv')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'in:{unit.in_channel}')
                _sub_str.append(f'out:{unit.out_channel}')
            elif unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'type:{unit.max_or_avg}')
            elif unit.type == 3:
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
