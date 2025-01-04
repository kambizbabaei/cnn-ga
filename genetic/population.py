import numpy as np
import hashlib
import copy

class Unit(object):
    def __init__(self, number):
        self.number = number


class ResUnit(Unit):
    def __init__(self, number, in_channel, out_channel, groups=1):  # << ADDED groups param
        """
        A convolutional unit with optional grouping.
        type=1 identifies this as a "conv" type in the pipeline.
        """
        super().__init__(number)
        self.type = 1
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.groups = groups  # << Store number of groups


class PoolUnit(Unit):
    def __init__(self, number, max_or_avg):
        super().__init__(number)
        self.type = 2
        self.max_or_avg = max_or_avg  # max_pool for < 0.5, otherwise avg_pool


class Individual(object):
    def __init__(self, params, indi_no):
        self.acc = -1.0
        self.id = indi_no  # for record the id of current individual
        self.number_id = 0  # for record the latest number assigned to a basic unit
        self.min_conv = params['min_conv']
        self.max_conv = params['max_conv']
        self.min_pool = params['min_pool']
        self.max_pool = params['max_pool']
        self.max_len = params['max_len']
        self.image_channel = params['image_channel']
        self.output_channles = params['output_channel']
        self.units = []

    def reset_acc(self):
        self.acc = -1.0

    def initialize(self):
        """
        Randomly initialize how many convolution + pooling layers this Individual will have,
        and in which order they appear.
        """
        # 1) Determine how many conv + pool layers will be used
        num_conv = np.random.randint(self.min_conv, self.max_conv + 1)
        num_pool = np.random.randint(self.min_pool, self.max_pool + 1)

        # 2) Randomly choose positions where pooling layers will appear among conv layers
        available_positions = list(range(num_conv))
        np.random.shuffle(available_positions)
        select_positions = np.sort(available_positions[0:num_pool])

        # 3) Build a list, e.g. [conv, conv, pool, conv, ...] to indicate the order
        all_positions = []
        for i in range(num_conv):
            all_positions.append(1)  # 1 denotes conv
            for j in select_positions:
                if j == i:
                    all_positions.append(2)  # 2 denotes pool
                    break

        # 4) Initialize the layers based on all_positions
        input_channel = self.image_channel
        for pos_type in all_positions:
            if pos_type == 1:  # conv
                conv = self.init_a_conv(_number=None, _in_channel=input_channel, _out_channel=None)
                input_channel = conv.out_channel
                self.units.append(conv)
            elif pos_type == 2:  # pool
                pool = self.init_a_pool(_number=None, _max_or_avg=None)
                self.units.append(pool)

    def init_a_conv(self, _number, _in_channel, _out_channel):
        """
        Create a ResUnit (conv) with random output_channel and random grouping,
        subject to the in_channel constraints.
        """
        if _number is not None:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _out_channel is not None:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]

        # << ADDED: Randomly choose groups from a small set [1,2,4], ensuring it divides in_channel
        possible_groups = [g for g in [1, 2, 4] if (_in_channel % g == 0)]
        if not possible_groups:
            groups = 1
        else:
            groups = np.random.choice(possible_groups)

        # Create a new conv unit (ResUnit) with the selected groups
        conv = ResUnit(number, _in_channel, out_channel, groups=groups)
        return conv

    def init_a_pool(self, _number, _max_or_avg):
        """
        Create a PoolUnit. max_or_avg < 0.5 => MaxPool, else AvgPool
        """
        if _number is not None:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _max_or_avg is not None:
            max_or_avg = _max_or_avg
        else:
            max_or_avg = np.random.rand()

        pool = PoolUnit(number, max_or_avg)
        return pool

    def uuid(self):
        """
        Create a unique hash key for this Individual by concatenating all
        layer definitions (type, in/out channels, grouping, etc.).
        """
        _str = []
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                # conv
                _sub_str.append('conv')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'in:{unit.in_channel}')
                _sub_str.append(f'out:{unit.out_channel}')
                # << ADDED: include groups
                _sub_str.append(f'groups:{unit.groups}')
            elif unit.type == 2:
                # pool
                _sub_str.append('pool')
                _sub_str.append(f'number:{unit.number}')
                _pool_type = 0.25 if unit.max_or_avg < 0.5 else 0.75
                _sub_str.append(f'type:{_pool_type:.1f}')

            _str.append(f"[{','.join(_sub_str)}]")

        _final_str_ = '-'.join(_str)
        _final_utf8_str_ = _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _final_str_

    def __str__(self):
        """
        String representation of the Individual for logging or saving.
        """
        _str = []
        _str.append(f"indi:{self.id}")
        _str.append(f"Acc:{self.acc:.5f}")
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('conv')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'in:{unit.in_channel}')
                _sub_str.append(f'out:{unit.out_channel}')
                # << ADDED groups to print
                _sub_str.append(f'groups:{unit.groups}')
            elif unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append(f'number:{unit.number}')
                _sub_str.append(f'type:{unit.max_or_avg:.1f}')
            _str.append(f"[{','.join(_sub_str)}]")
        return '\n'.join(_str)


class Population(object):
    def __init__(self, params, gen_no):
        """
        A Population holds a list of Individuals for a specific generation index.
        """
        self.gen_no = gen_no
        self.number_id = 0  # how many individuals have been generated so far
        self.pop_size = params['pop_size']
        self.params = params
        self.individuals = []

    def initialize(self):
        """
        Creates a brand new population of Individuals,
        each randomly initialized.
        """
        for _ in range(self.pop_size):
            indi_no = f'indi{self.gen_no:02d}{self.number_id:02d}'
            self.number_id += 1
            indi = Individual(self.params, indi_no)
            indi.initialize()
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        """
        After crossover + mutation, we have a new set of offspring. 
        This function wraps them into a new Population object 
        and reassigns IDs for clarity.
        """
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = f'indi{self.gen_no:02d}{self.number_id:02d}'
            indi.id = indi_no
            self.number_id += 1
            indi.number_id = len(indi.units)
            self.individuals.append(indi)

    def __str__(self):
        """
        Summarize all Individuals in this Population as a single string.
        """
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)
