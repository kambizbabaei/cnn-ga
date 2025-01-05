from utils import Utils, GPUTools, Log
import importlib
from multiprocessing import Process
import time, os, sys
# Removed "from asyncio.tasks import sleep" since we use time.sleep instead

class FitnessEvaluate(object):
    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log

    def generate_to_python_file(self):
        self.log.info('Begin to generate python files')
        for indi in self.individuals:
            # Generate PyTorch scripts (includes grouped conv if groups>1)
            Utils.generate_pytorch_file(indi)
        self.log.info('Finish the generation of python files')

    def evaluate(self):
        """
        1) Check cache for any pre-computed fitness.
        2) For each individual that needs evaluation (acc < 0):
           - Wait for an available GPU (if none is free, sleep).
           - Spawn one process to train/eval that individual on that GPU.
           - Immediately p.join() so we do them sequentially, 
             ensuring "one model per GPU" at a time.
        3) Once each model finishes, it writes its acc to after_XX.txt.
        4) Finally, read the new fitness from after_XX.txt and cache it.
        """
        self.log.info('Query fitness from cache')
        cache_map = Utils.load_cache_data()
        hit_count = 0

        # Assign cached accuracies where applicable
        for indi in self.individuals:
            key, _ = indi.uuid()
            if key in cache_map:
                indi.acc = float(cache_map[key])
                hit_count += 1
                self.log.info(f'Hit the cache for {indi.id}, key:{key}, acc:{indi.acc:.5f}')

        self.log.info(f'Total hit {hit_count} individuals for fitness')

        has_evaluated_offspring = False
        for indi in self.individuals:
            # Only train if not cached
            if indi.acc < 0:
                has_evaluated_offspring = True

                # 1) Wait until a GPU is free
                gpu_id = GPUTools.detect_available_gpu_id()
                while gpu_id is None:
                    self.log.info('No GPU available; wait 300s...')
                    time.sleep(300)
                    gpu_id = GPUTools.detect_available_gpu_id()

                # 2) Once we have a GPU, train the individual
                file_name = indi.id
                self.log.info(f'Begin to train {file_name} on GPU #{gpu_id}')
                module_name = f'scripts.{file_name}'

                # If already loaded in sys.modules, remove it so we can re-import fresh
                if module_name in sys.modules.keys():
                    self.log.info(f'Module:{module_name} loaded before, deleting it')
                    del sys.modules[module_name]

                # Import the generated script
                _module = importlib.import_module('.', module_name)
                _class = getattr(_module, 'RunModel')
                cls_obj = _class()

                # 3) Spawn the training process for this individual
                p = Process(target=cls_obj.do_work, args=(str(gpu_id), file_name,))
                p.start()
            else:
                # Already has fitness from cache
                file_name = indi.id
                self.log.info(f'{file_name} has inherited the fitness {indi.acc:.5f}, skip evaluate')
                after_file = f'./populations/after_{file_name[4:6]}.txt'
                with open(after_file, 'a+') as f:
                    f.write(f'{file_name}={indi.acc:.5f}\n')
                    f.flush()

        # If we trained any new offspring, read their final accuracies from after_XX.txt
        if has_evaluated_offspring:
            # All individuals in this generation share the same generation substring in their ID
            gen_sub = self.individuals[0].id[4:6]
            after_file = f'./populations/after_{gen_sub}.txt'
            if not os.path.exists(after_file):
                self.log.warn(f'File {after_file} does not exist, no new fitness found.')
            else:
                # Parse new fitnesses
                with open(after_file, 'r') as f:
                    new_map = {}
                    for line in f:
                        if '=' in line.strip():
                            _id, val = line.strip().split('=')
                            new_map[_id] = float(val)

                # Assign new fitness to each previously -1.0 individual
                for indi in self.individuals:
                    if indi.acc < 0:
                        if indi.id not in new_map:
                            self.log.warn(f'The fitness of {indi.id} not found in {after_file}')
                        else:
                            indi.acc = new_map[indi.id]
        else:
            self.log.info('No offspring was evaluated this round')

        # 5) Update the cache with newly evaluated individuals
        Utils.save_fitness_to_cache(self.individuals)
