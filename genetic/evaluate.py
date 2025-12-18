from utils import Utils, GPUTools, StatusUpdateTool, estimate_model_memory
import importlib
from multiprocessing import Process
import time, os, sys
import queue
import threading
import signal


class GPUQueueManager(object):
    """
    Manages GPU job queue with memory-aware scheduling.
    Uses bin-packing to optimally group models on GPUs.
    Oversized models run one-per-GPU.
    """
    
    def __init__(self, log):
        self.queue = queue.Queue()
        self.running_jobs = {}  # {gpu_id: [(process, individual_id, memory_mb), ...]}
        self.gpu_used_memory = {}  # {gpu_id: currently_used_memory_mb}
        self.log = log
        self.poll_interval = StatusUpdateTool.get_poll_interval()
        self.memory_margin = StatusUpdateTool.get_memory_safety_margin_mb()
        self.lock = threading.Lock()
        self.interrupted = False
        
        # Get GPU info once
        self.gpu_total_memory = self._get_all_gpu_memory()
        self.log.info(f'GPU memory available: {self.gpu_total_memory}')
        
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _get_all_gpu_memory(self):
        """Get total memory for all GPUs."""
        gpu_memory = {}
        try:
            equipped_ids, _ = GPUTools._get_equipped_gpu_ids_and_used_gpu_ids()
            for gpu_id in equipped_ids:
                meminfo = GPUTools.get_gpu_memory_info(gpu_id)
                if meminfo:
                    total, used, free = meminfo
                    # Use current free memory as baseline
                    gpu_memory[gpu_id] = free
        except Exception as e:
            self.log.error(f'Failed to get GPU memory: {e}')
        return gpu_memory
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.log.warn('Interrupt received, terminating all running jobs...')
        self.interrupted = True
        self.terminate_all()
        
    def add_job(self, individual):
        """Add an individual to the job queue."""
        self.queue.put(individual)
        self.log.info(f'Added {individual.id} to GPU queue (est. {individual.estimated_memory_mb:.0f} MB)')
    
    def create_optimal_schedule(self, individuals):
        """
        Create optimal groups of models using bin-packing algorithm.
        Models that don't fit together run one-per-GPU.
        
        Args:
            individuals: List of individuals to schedule
            
        Returns:
            tuple: (grouped_batches, oversized_individuals)
                - grouped_batches: List of batches, each batch is list of (gpu_id, [individuals])
                - oversized_individuals: List of individuals that need exclusive GPU access
        """
        if not individuals:
            return [], []
        
        # Refresh GPU memory info
        self.gpu_total_memory = self._get_all_gpu_memory()
        
        # Sort individuals by memory requirement (largest first - better for bin packing)
        sorted_indis = sorted(
            individuals, 
            key=lambda x: (x.estimated_memory_mb or 2048) + self.memory_margin, 
            reverse=True
        )
        
        gpu_ids = list(self.gpu_total_memory.keys())
        if not gpu_ids:
            self.log.error('No GPUs available!')
            return [], sorted_indis
        
        max_gpu_memory = max(self.gpu_total_memory.values()) if self.gpu_total_memory else 0
        
        # Separate oversized models (exceed any single GPU's memory)
        oversized = []
        normal = []
        for indi in sorted_indis:
            mem_needed = (indi.estimated_memory_mb or 2048) + self.memory_margin
            # Oversized = cannot fit on any GPU
            can_fit = any(mem_needed <= gpu_mem for gpu_mem in self.gpu_total_memory.values())
            if not can_fit:
                oversized.append(indi)
                self.log.warn(f'{indi.id} marked as oversized ({mem_needed:.0f} MB exceeds all GPU memory)')
            else:
                normal.append(indi)
        
        # Bin-pack normal models using First Fit Decreasing (no model count limit)
        batches = []  # List of {gpu_id: [individuals]}
        
        while normal:
            # Create a new batch - pack as many as fit on each GPU
            batch = {}
            batch_memory = {gpu_id: 0 for gpu_id in gpu_ids}
            remaining = []
            
            for indi in normal:
                mem_needed = (indi.estimated_memory_mb or 2048) + self.memory_margin
                
                # Try to place on GPU with least remaining capacity that can still fit it (best fit)
                best_gpu = None
                best_remaining = float('inf')
                
                for gpu_id in gpu_ids:
                    available = self.gpu_total_memory[gpu_id] - batch_memory[gpu_id]
                    if available >= mem_needed and available < best_remaining:
                        best_gpu = gpu_id
                        best_remaining = available
                
                if best_gpu is not None:
                    if best_gpu not in batch:
                        batch[best_gpu] = []
                    batch[best_gpu].append(indi)
                    batch_memory[best_gpu] += mem_needed
                    self.log.debug(f'Placed {indi.id} ({mem_needed:.0f} MB) on GPU {best_gpu}, remaining: {self.gpu_total_memory[best_gpu] - batch_memory[best_gpu]:.0f} MB')
                else:
                    remaining.append(indi)
            
            if batch:
                batches.append(batch)
                for gpu_id, indis in batch.items():
                    # Calculate total memory
                    total_mem = sum((i.estimated_memory_mb or 2048) + self.memory_margin for i in indis)
                    self.log.info(f'Batch {len(batches)} - GPU {gpu_id}: {len(indis)} model(s) [{", ".join(i.id for i in indis)}] using {total_mem:.0f}/{self.gpu_total_memory[gpu_id]:.0f} MB')
            
            normal = remaining
            
            # Safety: if we couldn't place anything, move to oversized
            if normal and not batch:
                self.log.warn(f'Could not place {normal[0].id}, treating as oversized')
                oversized.append(normal.pop(0))
        
        # Sort batches by total model count (most models first for better utilization)
        batches.sort(key=lambda b: sum(len(indis) for indis in b.values()), reverse=True)
        
        self.log.info(f'Schedule created: {len(batches)} batch(es), {len(oversized)} oversized model(s)')
        return batches, oversized
    
    def get_available_slot(self, required_memory):
        """
        Find a GPU with sufficient memory based on our tracking.
        
        Args:
            required_memory: Memory required in MB (estimated per model + margin)
        
        Returns:
            tuple: (gpu_id, available_memory_mb) or (None, None)
        """
        self.log.debug(f'Looking for GPU with {required_memory} MB free memory')
        
        # Use our own memory tracking to find available GPU
        best_gpu = None
        best_available = -1
        
        for gpu_id, total_mem in self.gpu_total_memory.items():
            used_mem = self.gpu_used_memory.get(gpu_id, 0)
            available = total_mem - used_mem
            
            if available >= required_memory and available > best_available:
                best_gpu = gpu_id
                best_available = available
        
        if best_gpu is not None:
            self.log.info(f'Found available GPU {best_gpu} with {best_available:.0f} MB free memory')
            return (best_gpu, best_available)
        
        self.log.debug('No GPU available with sufficient resources')
        return (None, None)
    
    def start_job(self, individual, gpu_id):
        """
        Start a training job on the specified GPU.
        
        Args:
            individual: Individual to train
            gpu_id: GPU ID to use
            
        Returns:
            Process object or None if failed
        """
        # Calculate required memory for this individual
        required_memory = (individual.estimated_memory_mb or 2048) + self.memory_margin
        
        # Verify memory one more time before starting
        if not GPUTools.has_sufficient_memory(gpu_id, required_memory):
            self.log.warn(f'GPU {gpu_id} no longer has sufficient memory for {individual.id} (needs {required_memory:.0f} MB)')
            return None
        
        try:
            file_name = individual.id
            module_name = 'scripts.%s' % (file_name)
            
            # Handle module reloading
            if module_name in sys.modules.keys():
                self.log.info(f'Module:{module_name} has been loaded, delete it')
                del sys.modules[module_name]
            
            _module = importlib.import_module(module_name)
            _class = getattr(_module, 'RunModel')
            cls_obj = _class()
            
            p = Process(target=cls_obj.do_work, args=(str(gpu_id), file_name,))
            p.start()
            
            # Track running job with memory usage
            with self.lock:
                if gpu_id not in self.running_jobs:
                    self.running_jobs[gpu_id] = []
                self.running_jobs[gpu_id].append((p, individual.id, required_memory))
                
                # Track memory usage
                if gpu_id not in self.gpu_used_memory:
                    self.gpu_used_memory[gpu_id] = 0
                self.gpu_used_memory[gpu_id] += required_memory
            
            self.log.info(f'Started {individual.id} on GPU {gpu_id} (PID: {p.pid}, {required_memory:.0f} MB)')
            return p
            
        except Exception as e:
            self.log.error(f'Failed to start {individual.id} on GPU {gpu_id}: {e}')
            return None
    
    def monitor_and_schedule(self):
        """
        Check for finished processes and schedule next jobs from queue.
        Returns True if any scheduling occurred, False otherwise.
        """
        scheduled = False
        
        with self.lock:
            # Check all running processes
            for gpu_id in list(self.running_jobs.keys()):
                jobs = self.running_jobs[gpu_id]
                finished_jobs = []
                
                for job_tuple in jobs:
                    process, individual_id, mem_used = job_tuple
                    if not process.is_alive():
                        finished_jobs.append(job_tuple)
                        self.log.info(f'Process for {individual_id} on GPU {gpu_id} finished (freed {mem_used:.0f} MB)')
                        # Free memory tracking
                        if gpu_id in self.gpu_used_memory:
                            self.gpu_used_memory[gpu_id] -= mem_used
                
                # Remove finished jobs
                for job in finished_jobs:
                    jobs.remove(job)
                
                # Update running jobs
                if jobs:
                    self.running_jobs[gpu_id] = jobs
                else:
                    del self.running_jobs[gpu_id]
                    if gpu_id in self.gpu_used_memory:
                        self.gpu_used_memory[gpu_id] = 0
            
            # Try to schedule new jobs
            while not self.queue.empty():
                try:
                    individual = self.queue.get_nowait()
                    required_memory = (individual.estimated_memory_mb or 2048) + self.memory_margin
                    
                    gpu_id, free_memory = self.get_available_slot(required_memory)
                    if gpu_id is None:
                        # Put back in queue, no available GPUs for this model
                        self.queue.put(individual)
                        break
                    
                    if self.start_job(individual, gpu_id):
                        scheduled = True
                    else:
                        # Put back in queue if failed to start
                        self.queue.put(individual)
                        break
                except queue.Empty:
                    break
        
        return scheduled
    
    def terminate_all(self):
        """Terminate all running processes."""
        with self.lock:
            for gpu_id, jobs in list(self.running_jobs.items()):
                for job_tuple in jobs:
                    process, individual_id, _ = job_tuple
                    if process.is_alive():
                        self.log.info(f'Terminating {individual_id} on GPU {gpu_id}')
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
            self.running_jobs.clear()
            self.gpu_used_memory.clear()
            # Clear queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
    
    def wait_for_all(self):
        """Wait until all jobs in queue are processed and all processes complete."""
        self.log.info('Waiting for all GPU jobs to complete...')
        
        while not self.interrupted:
            # Check if queue is empty and no processes running
            with self.lock:
                queue_empty = self.queue.empty()
                no_running = len(self.running_jobs) == 0
            
            if queue_empty and no_running:
                break
            
            # Monitor and schedule
            self.monitor_and_schedule()
            
            # Sleep before next check
            time.sleep(self.poll_interval)
        
        if self.interrupted:
            self.log.warn('Jobs interrupted by user')
        else:
            self.log.info('All GPU jobs completed')
    
    def run_batch(self, batch):
        """
        Run a batch of individuals on GPUs simultaneously.
        
        Args:
            batch: dict {gpu_id: [individuals]}
        """
        # Start all jobs in batch
        for gpu_id, individuals in batch.items():
            for indi in individuals:
                self.start_job(indi, gpu_id)
        
        # Wait for all to complete
        self.wait_for_current_jobs()
    
    def wait_for_current_jobs(self):
        """Wait for all currently running jobs to complete (no new scheduling)."""
        while not self.interrupted:
            with self.lock:
                no_running = len(self.running_jobs) == 0
            
            if no_running:
                break
            
            # Just check for finished processes, don't schedule new ones
            with self.lock:
                for gpu_id in list(self.running_jobs.keys()):
                    jobs = self.running_jobs[gpu_id]
                    finished_jobs = []
                    
                    for job_tuple in jobs:
                        process, individual_id, mem_used = job_tuple
                        if not process.is_alive():
                            finished_jobs.append(job_tuple)
                            self.log.info(f'Process for {individual_id} on GPU {gpu_id} finished')
                            if gpu_id in self.gpu_used_memory:
                                self.gpu_used_memory[gpu_id] -= mem_used
                    
                    for job in finished_jobs:
                        jobs.remove(job)
                    
                    if jobs:
                        self.running_jobs[gpu_id] = jobs
                    else:
                        del self.running_jobs[gpu_id]
                        if gpu_id in self.gpu_used_memory:
                            self.gpu_used_memory[gpu_id] = 0
            
            time.sleep(self.poll_interval)
    
    def run_oversized(self, individuals):
        """
        Run oversized individuals one-per-GPU.
        
        Args:
            individuals: List of oversized individuals
        """
        gpu_ids = list(self.gpu_total_memory.keys())
        
        for indi in individuals:
            if self.interrupted:
                break
            
            # Wait for a free GPU
            while not self.interrupted:
                # Refresh GPU memory
                self.gpu_total_memory = self._get_all_gpu_memory()
                
                required_memory = (indi.estimated_memory_mb or 2048) + self.memory_margin
                
                # Find GPU with enough memory and no running jobs
                for gpu_id in gpu_ids:
                    with self.lock:
                        running_count = len(self.running_jobs.get(gpu_id, []))
                    
                    if running_count == 0:
                        free_mem = self.gpu_total_memory.get(gpu_id, 0)
                        if free_mem >= required_memory:
                            self.log.info(f'Running oversized {indi.id} exclusively on GPU {gpu_id}')
                            self.start_job(indi, gpu_id)
                            break
                else:
                    # No GPU available, wait
                    time.sleep(self.poll_interval)
                    self.wait_for_current_jobs()
                    continue
                break  # Started successfully


class FitnessEvaluate(object):

    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log

    def generate_to_python_file(self):
        self.log.info('Begin to generate python files')
        for indi in self.individuals:
            Utils.generate_pytorch_file(indi)
        self.log.info('Finish the generation of python files')
    
    def estimate_memory_requirements(self):
        """
        Estimate GPU memory requirements for each individual by loading
        the model on CPU and analyzing its structure.
        """
        self.log.info('Estimating memory requirements for individuals...')
        
        for indi in self.individuals:
            if indi.estimated_memory_mb is not None:
                self.log.info(f'{indi.id}: Using cached memory estimate {indi.estimated_memory_mb:.0f} MB')
                continue
            
            try:
                file_name = indi.id
                module_name = 'scripts.%s' % (file_name)
                
                # Handle module reloading
                if module_name in sys.modules.keys():
                    del sys.modules[module_name]
                
                _module = importlib.import_module(module_name)
                
                # Get the model class and instantiate on CPU
                model_class = getattr(_module, 'EvoCNNModel')
                model = model_class()
                
                # Estimate memory
                memory_mb = estimate_model_memory(model, batch_size=128, input_size=(3, 32, 32))
                indi.estimated_memory_mb = memory_mb
                
                self.log.info(f'{indi.id}: Estimated memory {memory_mb:.0f} MB')
                
                # Clean up
                del model
                
            except Exception as e:
                self.log.warn(f'{indi.id}: Failed to estimate memory ({e}), using default 2048 MB')
                indi.estimated_memory_mb = 2048
        
        self.log.info('Memory estimation complete')

    def evaluate(self):
        """
        Evaluate fitness using optimal bin-packing GPU scheduler.
        Groups models that fit together, runs oversized models one-per-GPU.
        """
        # Step 1: Check cache for all individuals
        self.log.info('Query fitness from cache')
        _map = Utils.load_cache_data()
        _count = 0
        for indi in self.individuals:
            _key, _str = indi.uuid()
            if _key in _map:
                _count += 1
                _acc = _map[_key]
                self.log.info('Hit the cache for %s, key:%s, acc:%.5f, assigned_acc:%.5f'%(indi.id, _key, float(_acc), indi.acc))
                indi.acc = float(_acc)
        self.log.info('Total hit %d individuals for fitness'%(_count))

        # Step 2: Get uncached individuals and estimate memory
        uncached_individuals = [indi for indi in self.individuals if indi.acc < 0]
        
        # Save cached fitness to file
        for indi in self.individuals:
            if indi.acc >= 0:
                file_name = indi.id
                self.log.info('%s has inherited the fitness as %.5f, no need to evaluate'%(file_name, indi.acc))
                f = open('./populations/after_%s.txt'%(file_name[4:6]), 'a+')
                f.write('%s=%.5f\n'%(file_name, indi.acc))
                f.flush()
                f.close()

        if not uncached_individuals:
            self.log.info('No individuals need evaluation')
            Utils.save_fitness_to_cache(self.individuals)
            return
        
        # Estimate memory for uncached individuals
        self.estimate_memory_requirements()
        
        # Step 3: Create optimal schedule using bin-packing
        queue_manager = GPUQueueManager(self.log)
        batches, oversized = queue_manager.create_optimal_schedule(uncached_individuals)
        
        self.log.info(f'=== Optimal Schedule ===')
        self.log.info(f'Normal batches: {len(batches)}')
        self.log.info(f'Oversized models (run 1-per-GPU): {len(oversized)}')
        
        # Step 4: Run batches (grouped models)
        for i, batch in enumerate(batches):
            if queue_manager.interrupted:
                break
            self.log.info(f'=== Running batch {i+1}/{len(batches)} ===')
            for gpu_id, indis in batch.items():
                self.log.info(f'  GPU {gpu_id}: {[ind.id for ind in indis]}')
            queue_manager.run_batch(batch)
        
        # Step 5: Run oversized models one-per-GPU
        if oversized and not queue_manager.interrupted:
            self.log.info(f'=== Running {len(oversized)} oversized model(s) (1-per-GPU) ===')
            queue_manager.run_oversized(oversized)
            queue_manager.wait_for_current_jobs()  # Wait for last oversized to finish
        
        # Step 6: Load fitness results
            file_name = './populations/after_%s.txt'%(self.individuals[0].id[4:6])
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            fitness_map = {}
            for line in f:
                if len(line.strip()) > 0:
                    line = line.strip().split('=')
                    fitness_map[line[0]] = float(line[1])
            f.close()
            
            # Wait a bit for any final writes
            max_wait = 60  # Maximum wait time in seconds
            wait_interval = 5
            waited = 0
            
            for indi in self.individuals:
                if indi.acc == -1:
                    if indi.id not in fitness_map:
                        # Wait for fitness result with timeout
                        while indi.id not in fitness_map and waited < max_wait:
                            self.log.warn('Fitness for %s not found, waiting %d seconds...'%(indi.id, wait_interval))
                            time.sleep(wait_interval)
                            waited += wait_interval
                            
                            # Re-read file
                            if os.path.exists(file_name):
                                f = open(file_name, 'r')
                                for line in f:
                                    if len(line.strip()) > 0:
                                        parts = line.strip().split('=')
                                        if len(parts) == 2:
                                            fitness_map[parts[0]] = float(parts[1])
                                f.close()
                        
                        if indi.id not in fitness_map:
                            self.log.error('Fitness for %s still not found after waiting'%(indi.id))
                            continue
                    
                    indi.acc = fitness_map[indi.id]
        else:
            self.log.warn('Fitness file %s does not exist yet'%(file_name))

        # Step 7: Save fitness to cache
        Utils.save_fitness_to_cache(self.individuals)
