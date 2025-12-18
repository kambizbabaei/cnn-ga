import configparser
import os
import numpy as np
from subprocess import Popen, PIPE
from genetic.population import Population, Individual, ResUnit, PoolUnit
import logging
import sys
import multiprocessing
import time
# Note: nvidia-ml-py package still uses 'pynvml' as the module name
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetMemoryInfo,
    nvmlSystemGetProcessName,
    NVMLError,
)

class StatusUpdateTool(object):
    @classmethod
    def clear_config(cls):
        config = configparser.ConfigParser()
        config.read('global.ini')
        secs = config.sections()
        for sec_name in secs:
            if sec_name == 'evolution_status' or sec_name == 'gpu_running_status':
                item_list = config.options(sec_name)
                for item_name in item_list:
                    config.set(sec_name, item_name, " ")
        config.write(open('global.ini', 'w'))

    @classmethod
    def __write_ini_file(cls, section, key, value):
        config = configparser.ConfigParser()
        config.read('global.ini')
        config.set(section, key, value)
        config.write(open('global.ini', 'w'))

    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read('global.ini')
        return config.get(section, key)

    @classmethod
    def begin_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "1")

    @classmethod
    def end_evolution(cls):
        section = 'evolution_status'
        key = 'IS_RUNNING'
        cls.__write_ini_file(section, key, "0")

    @classmethod
    def is_evolution_running(cls):
        rs = cls.__read_ini_file('evolution_status', 'IS_RUNNING')
        if rs == '1':
            return True
        else:
            return False

    @classmethod
    def get_conv_limit(cls):
        rs = cls.__read_ini_file('network', 'conv_limit')
        conv_limit = []
        for i in rs.split(','):
            conv_limit.append(int(i))
        return conv_limit[0], conv_limit[1]

    @classmethod
    def get_pool_limit(cls):
        rs = cls.__read_ini_file('network', 'pool_limit')
        pool_limit = []
        for i in rs.split(','):
            pool_limit.append(int(i))
        return pool_limit[0], pool_limit[1]

    @classmethod
    def get_output_channel(cls):
        rs = cls.__read_ini_file('network', 'output_channel')
        channels = []
        for i in rs.split(','):
            channels.append(int(i))
        return channels

    @classmethod
    def get_input_channel(cls):
        rs = cls.__read_ini_file('network', 'input_channel')
        return int(rs)

    @classmethod
    def get_num_class(cls):
        rs = cls.__read_ini_file('network', 'num_class')
        return int(rs)

    @classmethod
    def get_pop_size(cls):
        rs = cls.__read_ini_file('settings', 'pop_size')
        return int(rs)

    @classmethod
    def get_epoch_size(cls):
        rs = cls.__read_ini_file('network', 'epoch')
        return int(rs)

    @classmethod
    def get_individual_max_length(cls):
        rs = cls.__read_ini_file('network', 'max_length')
        return int(rs)

    @classmethod
    def get_genetic_probability(cls):
        rs = cls.__read_ini_file('settings', 'genetic_prob').split(',')
        p = [float(i) for i in rs]
        return p

    @classmethod
    def get_init_params(cls):
        params = {}
        params['pop_size'] = cls.get_pop_size()
        params['min_conv'], params['max_conv'] = cls.get_conv_limit()
        params['min_pool'], params['max_pool'] = cls.get_pool_limit()
        params['max_len'] = cls.get_individual_max_length()
        params['image_channel'] = cls.get_input_channel()
        params['output_channel'] = cls.get_output_channel()
        params['genetic_prob'] = cls.get_genetic_probability()
        return params

    @classmethod
    def get_mutation_probs_for_each(cls):
        """
        We define 4 probabilities for each type of mutation:
          1) add one conv/pool
          2) remove one conv/pool
          3) change in/out channels
          4) pooling type
        Then we choose one of them according to these probabilities.
        """
        rs = cls.__read_ini_file('settings', 'mutation_probs').split(',')
        assert len(rs) == 4
        mutation_prob_list = [float(i) for i in rs]
        return mutation_prob_list

    @classmethod
    def get_models_per_gpu(cls):
        try:
            rs = cls.__read_ini_file('gpu_scheduling', 'models_per_gpu')
            return int(rs)
        except:
            return 1  # Default value

    @classmethod
    def get_poll_interval(cls):
        try:
            rs = cls.__read_ini_file('gpu_scheduling', 'poll_interval')
            return int(rs)
        except:
            return 5  # Default value

    @classmethod
    def get_min_free_memory_mb(cls):
        try:
            rs = cls.__read_ini_file('gpu_scheduling', 'min_free_memory_mb')
            return int(rs)
        except:
            return 2048  # Default value (2GB)

    @classmethod
    def get_memory_safety_margin_mb(cls):
        try:
            rs = cls.__read_ini_file('gpu_scheduling', 'memory_safety_margin_mb')
            return int(rs)
        except:
            return 512  # Default value (512MB)


def estimate_model_memory(model, batch_size=128, input_size=(3, 32, 32)):
    """
    Estimate GPU memory required for training a model.
    Measures actual CUDA memory usage during multiple forward+backward passes.
    Uses SGD optimizer matching actual training configuration.
    
    Args:
        model: PyTorch nn.Module (on CPU)
        batch_size: Training batch size
        input_size: Input tensor size (C, H, W)
        
    Returns:
        Estimated memory in MB
    """
    import torch
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        # Fallback to CPU estimation if CUDA not available
        param_bytes = sum(p.numel() * 4 for p in model.parameters())
        # Rough estimate: 4x parameters for training (params + grads + optimizer + activations)
        total_bytes = param_bytes * 4
        return total_bytes / (1024 * 1024)
    
    device = torch.device('cuda:0')
    
    try:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Enable cudnn.benchmark to match actual training setup
        # This affects cuDNN workspace memory allocation
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        
        # Move model to GPU (create a copy to avoid modifying original)
        model_gpu = model.to(device)
        
        # Create dummy input on GPU
        dummy_input = torch.randn(batch_size, *input_size, device=device)
        dummy_target = torch.randint(0, 10, (batch_size,), device=device)
        
        # Use SGD optimizer matching actual training configuration
        # Training uses: momentum=0.9, weight_decay=5e-4
        optimizer = torch.optim.SGD(
            model_gpu.parameters(),
            lr=0.1,  # Initial learning rate from training
            momentum=0.9,
            weight_decay=5e-4
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        # Run multiple iterations (5 passes) to capture peak memory across training
        num_iterations = 5
        peak_memory_bytes = 0
        
        for iteration in range(num_iterations):
            # Zero gradients
            optimizer.zero_grad()
            
            # Run forward pass
            output = model_gpu(dummy_input)
            loss = criterion(output, dummy_target)
            
            # Run backward pass (this allocates gradients and stores activations)
            loss.backward()
            optimizer.step()
            
            # Check peak memory after each iteration
            current_peak = torch.cuda.max_memory_allocated(device)
            if current_peak > peak_memory_bytes:
                peak_memory_bytes = current_peak
        
        # Clean up
        del model_gpu, dummy_input, dummy_target, output, loss, optimizer, criterion
        torch.cuda.empty_cache()
        
        # Convert to MB
        memory_mb = peak_memory_bytes / (1024 * 1024)
        
        # Apply correction factor to account for:
        # - cuDNN workspace memory (not tracked by max_memory_allocated)
        # - Memory fragmentation over multiple epochs
        # - Memory allocated outside PyTorch's allocator
        # Based on testing: torch.cuda.max_memory_allocated shows ~807MB but actual usage is ~1420MB
        # Ratio: 1420/807 ≈ 1.76x, using 1.8x for safety margin
        memory_mb = memory_mb * 1.8
        
        return memory_mb
        
    except Exception as e:
        # Fallback: use parameter-based heuristic if GPU measurement fails
        param_bytes = sum(p.numel() * 4 for p in model.parameters())
        # Conservative estimate: 4-5x parameters for training
        total_bytes = param_bytes * 4.5
        return total_bytes / (1024 * 1024)


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("EvoCNN")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warning(_str)
    @classmethod
    def debug(cls, _str):
        cls.__get_logger().debug(_str)
    @classmethod
    def error(cls, _str):
        cls.__get_logger().error(_str)

class GPUTools(object):
    """
    A utility class for managing and querying NVIDIA GPU resources using NVML.
    """

    @classmethod
    def _get_equipped_gpu_ids_and_used_gpu_ids(cls):
        """
        Retrieves the list of equipped GPU IDs and identifies which GPUs are currently in use.

        Returns:
            tuple:
                equipped_gpu_ids (list of str): List of GPU IDs as strings (e.g., ["0", "1"]).
                used_gpu_ids (set of str): Set of GPU IDs that are currently in use.
        """
        equipped_gpu_ids = []
        used_gpu_ids = set()

        try:
            # Initialize NVML
            nvmlInit()
            Log.debug("NVML initialized successfully.")

            # Get the number of GPUs
            device_count = nvmlDeviceGetCount()
            Log.info(f"Number of GPUs detected: {device_count}")

            for i in range(device_count):
                try:
                    # Get handle for each GPU
                    handle = nvmlDeviceGetHandleByIndex(i)

                    # Get GPU name
                    gpu_name = nvmlDeviceGetName(handle)
                    Log.debug(f"GPU {i} Name: {gpu_name}")

                    # Check if GPU is of type GeForce or Tesla
                    if 'GeForce' in gpu_name or 'Tesla' in gpu_name:
                        gpu_id_str = str(i)
                        equipped_gpu_ids.append(gpu_id_str)
                        Log.debug(f"GPU {gpu_id_str} is equipped and recognized as GeForce/Tesla.")

                        # Get list of processes using this GPU
                        processes = nvmlDeviceGetComputeRunningProcesses(handle)
                        process_count = len(processes)
                        Log.debug(f"GPU {gpu_id_str} has {process_count} running compute process(es).")
                        if process_count > 0:
                            used_gpu_ids.add(gpu_id_str)

                except NVMLError as e:
                    Log.error(f"NVML error while processing GPU {i}: {e}")
                except Exception as e:
                    Log.error(f"Unexpected error while processing GPU {i}: {e}")

        except NVMLError as e:
            Log.error(f"Failed to initialize NVML: {e}")
        except Exception as e:
            Log.error(f"Unexpected error during NVML initialization: {e}")
        finally:
            try:
                nvmlShutdown()
                Log.debug("NVML shutdown successfully.")
            except:
                pass  # NVML may not have been initialized; ignore shutdown errors

        return equipped_gpu_ids, used_gpu_ids

    @classmethod
    def get_available_gpu_ids(cls):
        """
        Determines the list of available GPU IDs that are not currently in use.

        Returns:
            list of str: List of available GPU IDs as strings. Empty list if no GPUs are available.
        """
        equipped_gpu_ids, used_gpu_ids = cls._get_equipped_gpu_ids_and_used_gpu_ids()

        # Determine unused GPUs by excluding used GPUs from equipped GPUs
        unused_gpu_ids = [gpu_id for gpu_id in equipped_gpu_ids if gpu_id not in used_gpu_ids]

        if not unused_gpu_ids:
            Log.info("No available GPUs. All equipped GPUs are currently in use.")
        else:
            Log.info(f"Available GPUs: {unused_gpu_ids}")

        Log.debug(f"Equipped GPUs: {equipped_gpu_ids}")
        Log.debug(f"Used GPUs: {sorted(used_gpu_ids)}")
        Log.debug(f"Unused GPUs: {unused_gpu_ids}")

        return unused_gpu_ids

    @classmethod
    def detect_available_gpu_id(cls):
        """
        Selects an available GPU ID for use.

        Returns:
            str or None: The selected GPU ID as a string, or None if no GPUs are available.
        """
        unused_gpu_ids = cls.get_available_gpu_ids()
        if not unused_gpu_ids:
            Log.info('GPU_QUERY-No available GPU')
            return None
        else:
            selected_gpu = unused_gpu_ids[0]
            Log.info(f'GPU_QUERY-Available GPUs are: [{",".join(unused_gpu_ids)}], choose GPU#{selected_gpu} to use')
            return selected_gpu  # Return as string for consistency

    @classmethod
    def all_gpu_available(cls):
        """
        Checks if all equipped GPUs are available (i.e., not occupied by any processes).

        Returns:
            bool: True if all GPUs are available, False otherwise.
        """
        equipped_gpu_ids, used_gpu_ids = cls._get_equipped_gpu_ids_and_used_gpu_ids()

        if not equipped_gpu_ids:
            Log.info("No equipped GPUs of specified types found.")
            return True  # Assuming no GPUs means all (zero) GPUs are available

        if not used_gpu_ids:
            Log.info('GPU_QUERY-None of the GPUs are occupied')
            return True
        else:
            occupied_gpus = sorted(used_gpu_ids)
            Log.info(f'GPU_QUERY- GPUs [{",".join(occupied_gpus)}] are occupying')
            return False

    @classmethod
    def get_gpu_memory_info(cls, gpu_id):
        """
        Get memory information for a specific GPU.

        Args:
            gpu_id: GPU ID as string (e.g., "0", "1")

        Returns:
            tuple: (total_mb, used_mb, free_mb) or None if error
        """
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(int(gpu_id))
            meminfo = nvmlDeviceGetMemoryInfo(handle)
            
            total_mb = meminfo.total / (1024 * 1024)  # Convert bytes to MB
            used_mb = meminfo.used / (1024 * 1024)
            free_mb = meminfo.free / (1024 * 1024)
            
            # Don't shutdown here - let caller manage NVML lifecycle
            # nvmlShutdown() is called in _get_equipped_gpu_ids_and_used_gpu_ids
            
            return (total_mb, used_mb, free_mb)
        except Exception as e:
            Log.error(f"Error getting memory info for GPU {gpu_id}: {e}")
            import traceback
            Log.error(traceback.format_exc())
            return None

    @classmethod
    def has_sufficient_memory(cls, gpu_id, required_mb):
        """
        Check if a GPU has sufficient free memory.

        Args:
            gpu_id: GPU ID as string
            required_mb: Required memory in MB

        Returns:
            bool: True if GPU has sufficient memory, False otherwise
        """
        meminfo = cls.get_gpu_memory_info(gpu_id)
        if meminfo is None:
            return False
        total_mb, used_mb, free_mb = meminfo
        return free_mb >= required_mb

    @classmethod
    def get_gpu_process_count(cls, gpu_id):
        """
        Get the number of processes running on a specific GPU.

        Args:
            gpu_id: GPU ID as string

        Returns:
            int: Number of processes, or 0 if error
        """
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(int(gpu_id))
            processes = nvmlDeviceGetComputeRunningProcesses(handle)
            # Don't shutdown here - let caller manage NVML lifecycle
            return len(processes)
        except Exception as e:
            Log.error(f"Error getting process count for GPU {gpu_id}: {e}")
            return 0

    @classmethod
    def get_available_gpu_with_memory(cls, required_mb, models_per_gpu, tracked_process_count=None):
        """
        Find a GPU that has both an available slot and sufficient memory.

        Args:
            required_mb: Required memory in MB
            models_per_gpu: Maximum number of models per GPU
            tracked_process_count: Optional dict {gpu_id: count} of processes we're tracking ourselves

        Returns:
            tuple: (gpu_id, free_memory_mb) or (None, None) if no GPU available
        """
        try:
            nvmlInit()
            equipped_gpu_ids, _ = cls._get_equipped_gpu_ids_and_used_gpu_ids()
            
            if not equipped_gpu_ids:
                Log.info(f'No equipped GPUs found. Required memory: {required_mb} MB')
                nvmlShutdown()
                return (None, None)
            
            for gpu_id in equipped_gpu_ids:
                try:
                    handle = nvmlDeviceGetHandleByIndex(int(gpu_id))
                    
                    # Check if GPU has available slot - use our own tracking if provided
                    if tracked_process_count is not None and gpu_id in tracked_process_count:
                        our_process_count = tracked_process_count[gpu_id]
                        Log.info(f'GPU {gpu_id}: {our_process_count} tracked processes (max: {models_per_gpu})')
                        if our_process_count >= models_per_gpu:
                            Log.info(f'GPU {gpu_id}: No available slot (tracked={our_process_count} >= {models_per_gpu})')
                            continue
                    else:
                        # Fallback to NVML count if no tracking provided
                        processes = nvmlDeviceGetComputeRunningProcesses(handle)
                        process_count = len(processes)
                        Log.info(f'GPU {gpu_id}: {process_count} total processes on GPU (max: {models_per_gpu})')
                        if process_count >= models_per_gpu:
                            Log.info(f'GPU {gpu_id}: No available slot (process_count={process_count} >= {models_per_gpu})')
                            continue
                    
                    # Check memory
                    meminfo = nvmlDeviceGetMemoryInfo(handle)
                    total_mb = meminfo.total / (1024 * 1024)
                    used_mb = meminfo.used / (1024 * 1024)
                    free_mb = meminfo.free / (1024 * 1024)
                    
                    Log.info(f'GPU {gpu_id}: Memory - Total: {total_mb:.0f} MB, Used: {used_mb:.0f} MB, Free: {free_mb:.0f} MB, Required: {required_mb} MB')
                    
                    if free_mb >= required_mb:
                        Log.info(f'GPU {gpu_id} available: {free_mb:.0f} MB free (required: {required_mb} MB)')
                        nvmlShutdown()
                        return (gpu_id, free_mb)
                    else:
                        Log.warn(f'GPU {gpu_id}: Insufficient memory ({free_mb:.0f} MB < {required_mb} MB)')
                        
                except Exception as e:
                    Log.error(f'Error checking GPU {gpu_id}: {e}')
                    continue
            
            Log.warn(f'No GPU available with sufficient memory. Required: {required_mb} MB')
            nvmlShutdown()
            return (None, None)
        except Exception as e:
            Log.error(f'Error in get_available_gpu_with_memory: {e}')
            import traceback
            Log.error(traceback.format_exc())
            try:
                nvmlShutdown()
            except:
                pass
            return (None, None)

class Utils(object):
    _lock = multiprocessing.Lock()

    @classmethod
    def get_lock_for_write_fitness(cls):
        return cls._lock

    @classmethod
    def load_cache_data(cls):
        file_name = './populations/cache.txt'
        _map = {}
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            for each_line in f:
                rs_ = each_line.strip().split(';')
                _map[rs_[0]] = '%.5f' % (float(rs_[1]))
            f.close()
        return _map

    @classmethod
    def save_fitness_to_cache(cls, individuals):
        _map = cls.load_cache_data()
        for indi in individuals:
            _key, _str = indi.uuid()
            _acc = indi.acc
            if _key not in _map:
                Log.info('Add record into cache, id:%s, acc:%.5f' % (_key, _acc))
                f = open('./populations/cache.txt', 'a+')
                _str = '%s;%.5f;%s\n' % (_key, _acc, _str)
                f.write(_str)
                f.close()
                _map[_key] = _acc

    @classmethod
    def save_population_at_begin(cls, _str, gen_no):
        file_name = './populations/begin_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_after_crossover(cls, _str, gen_no):
        file_name = './populations/crossover_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_after_mutation(cls, _str, gen_no):
        file_name = './populations/mutation_%02d.txt' % (gen_no)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def get_newest_file_based_on_prefix(cls, prefix):
        id_list = []
        for _, _, file_names in os.walk('./populations'):
            for file_name in file_names:
                if file_name.startswith(prefix):
                    id_list.append(int(file_name[6:8]))
        if len(id_list) == 0:
            return None
        else:
            return np.max(id_list)

    @classmethod
    def load_population(cls, prefix, gen_no):
        file_name = './populations/%s_%02d.txt' % (prefix, np.min(gen_no))
        params = StatusUpdateTool.get_init_params()
        pop = Population(params, gen_no)
        f = open(file_name)
        indi_start_line = f.readline().strip()
        while indi_start_line.startswith('indi'):
            indi_no = indi_start_line[5:]
            indi = Individual(params, indi_no)
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    indi_start_line = f.readline().strip()
                    break
                else:
                    if line.startswith('Acc'):
                        indi.acc = float(line[4:])
                    elif line.startswith('[conv'):
                        data_maps = line[6:-1].split(',', 6)  # ADDED: increased split count if needed for groups
                        conv_params = {}
                        for data_item in data_maps:
                            _key, _value = data_item.split(":")
                            if _key == 'number':
                                indi.number_id = int(_value)
                                conv_params['number'] = int(_value)
                            elif _key == 'in':
                                conv_params['in_channel'] = int(_value)
                            elif _key == 'out':
                                conv_params['out_channel'] = int(_value)
                            elif _key == 'groups':  # ADDED: handle "groups" from file
                                conv_params['groups'] = int(_value)
                            else:
                                raise ValueError('Unknown key for load conv unit, key_name:%s' % (_key))
                        # ADDED: pass groups param to ResUnit
                        if 'groups' not in conv_params:
                            conv_params['groups'] = 1
                        conv = ResUnit(
                            conv_params['number'],
                            conv_params['in_channel'],
                            conv_params['out_channel'],
                            conv_params['groups']
                        )
                        indi.units.append(conv)
                    elif line.startswith('[pool'):
                        pool_params = {}
                        for data_item in line[6:-1].split(','):
                            _key, _value = data_item.split(':')
                            if _key == 'number':
                                indi.number_id = int(_value)
                                pool_params['number'] = int(_value)
                            elif _key == 'type':
                                pool_params['max_or_avg'] = float(_value)
                            else:
                                raise ValueError('Unknown key for load pool unit, key_name:%s' % (_key))
                        pool = PoolUnit(pool_params['number'], pool_params['max_or_avg'])
                        indi.units.append(pool)
                    else:
                        print('Unknown key for load unit type, line content:%s' % (line))
            pop.individuals.append(indi)
        f.close()

        # load the fitness to the individuals who have been evaluated (only suitable for the first generation)
        if gen_no == 0:
            after_file_path = './populations/after_%02d.txt' % (gen_no)
            if os.path.exists(after_file_path):
                fitness_map = {}
                with open(after_file_path) as f2:
                    for line in f2:
                        if len(line.strip()) > 0:
                            line = line.strip().split('=')
                            fitness_map[line[0]] = float(line[1])
                for indi in pop.individuals:
                    if indi.id in fitness_map:
                        indi.acc = fitness_map[indi.id]
        return pop

    @classmethod
    def read_template(cls):
        _path = './template/cifar10.py'
        part1 = []
        part2 = []
        part3 = []

        f = open(_path)
        f.readline()  # skip the initial comment line
        line = f.readline().rstrip()
        while line.strip() != '#generated_init':
            part1.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()  # skip '#generated_init'
        while line.strip() != '#generate_forward':
            part2.append(line)
            line = f.readline().rstrip()

        line = f.readline().rstrip()  # skip '#generate_forward'
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()

        return part1, part2, part3

    @classmethod
    def generate_pytorch_file(cls, indi):
        # Query all conv units
        conv_name_list = []
        conv_list = []
        for u in indi.units:
            if u.type == 1:
                # UPDATED: incorporate groups in the conv name
                conv_name = 'self.conv_%d_%d_g%d' % (u.in_channel, u.out_channel, u.groups)  # ADDED
                if conv_name not in conv_name_list:
                    conv_name_list.append(conv_name)
                    # UPDATED: pass groups param into BasicBlock
                    conv = '%s = BasicBlock(in_planes=%d, planes=%d, stride=1, groups=%d)' % (
                        conv_name, u.in_channel, u.out_channel, u.groups
                    )
                    conv_list.append(conv)

        # Query final channel & image size for fully-connected layer
        out_channel_list = []
        image_output_size = 32
        for u in indi.units:
            if u.type == 1:
                out_channel_list.append(u.out_channel)
            else:
                # a pool unit halves the size (2x2 pooling)
                if len(out_channel_list) == 0:
                    # if the first layer is pooling, there's no prior out_channel, default is input channel
                    out_channel_list.append(StatusUpdateTool.get_input_channel())
                image_output_size = int(image_output_size / 2)
                # keep same out_channel
                out_channel_list.append(out_channel_list[-1])

        # final out_channel is the last element
        fully_layer_name = 'self.linear = nn.Linear(%d, %d)' % (
            image_output_size * image_output_size * out_channel_list[-1],
            StatusUpdateTool.get_num_class()
        )

        # Generate the forward part
        forward_list = []
        for i, u in enumerate(indi.units):
            if i == 0:
                last_out_put = 'x'
            else:
                last_out_put = 'out_%d' % (i - 1)
            if u.type == 1:
                # UPDATED: match conv name to what we used above with groups
                _str = 'out_%d = self.conv_%d_%d_g%d(%s)' % (i, u.in_channel, u.out_channel, u.groups, last_out_put)  # ADDED
                forward_list.append(_str)
            else:
                if u.max_or_avg < 0.5:
                    _str = 'out_%d = F.max_pool2d(out_%d, 2)' % (i, i - 1)
                else:
                    _str = 'out_%d = F.avg_pool2d(out_%d, 2)' % (i, i - 1)
                forward_list.append(_str)

        forward_list.append('out = out_%d' % (len(indi.units) - 1))

        part1, part2, part3 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\n        #conv unit')
        for s in conv_list:
            _str.append('        %s' % (s))
        _str.append('\n        #linear unit')
        _str.append('        %s' % (fully_layer_name))

        _str.extend(part2)
        for s in forward_list:
            _str.append('        %s' % (s))
        _str.extend(part3)

        file_name = './scripts/%s.py' % (indi.id)
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()

    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()


if __name__ == '__main__':
    # Example usage: writing a small string to a file
    _str = 'test\n test1'
    _file = './populations/ENV_00.txt'
    Utils.write_to_file(_str, _file)
