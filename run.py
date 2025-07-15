import os
import re
import glob
import traceback
import json
import pickle
import argparse
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import string


def load_task_data(task_file):
    """Load task data from pickle and JSON files."""
    task_path = Path(task_file)
    
    # Load common info from JSON
    json_path = task_path.with_suffix('.json')
    with open(json_path, 'r') as f:
        common_info = json.load(f)
    
    # Load individual tasks from pickle
    with open(task_path, 'rb') as f:
        tasks = pickle.load(f)
    
    return common_info, tasks


def create_resource_queue(hardware_params, manager):
    """Create a queue of computing resources."""
    num_gpus = hardware_params['NUM_GPUS']
    num_proc = hardware_params['NUM_PROC']
    num_tasks = num_gpus // num_proc
    
    cpu_count = os.cpu_count()
    cpus_per_task = cpu_count // num_tasks
    
    base_port = 29540  # Default master port
    
    # Create managed queue with maxsize=num_tasks
    resource_queue = manager.Queue(maxsize=num_tasks)
    
    for i in range(num_tasks):
        # CUDA devices allocation
        start_gpu = i * num_proc
        end_gpu = start_gpu + num_proc
        cuda_devices = ','.join(str(gpu) for gpu in range(start_gpu, end_gpu))
        
        # CPU affinity allocation
        start_cpu = i * cpus_per_task
        end_cpu = start_cpu + cpus_per_task - 1
        if i == num_tasks - 1:  # Last task gets remaining CPUs
            end_cpu = cpu_count - 1
        cpu_affinity = f"{start_cpu}-{end_cpu}"
        
        # Master port allocation
        master_port = base_port + i
        
        resource = {
            'CUDA_DEVICES': cuda_devices,
            'CPU_AFFINITY': cpu_affinity,
            'MASTER_PORT': master_port
        }
        resource_queue.put(resource)
    
    return resource_queue, num_tasks


def format_params_for_command(params):
    """Format parameters for command line."""
    cmd_parts = []
    for key, value in params.items():
        cmd_parts.append(f"--{key}")
        if isinstance(value, (list, tuple)):
            cmd_parts.extend(str(v) for v in value)
        else:
            cmd_parts.append(str(value))
    return cmd_parts


def check_skip_pattern(output_dir, skip_pattern):
    """Check if output directory contains files matching skip pattern."""
    if not os.path.exists(output_dir):
        return False
    
    try:
        pattern = re.compile(skip_pattern)
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if pattern.match(file):
                    return True
    except re.error:
        print(f"Invalid regex pattern: {skip_pattern}")
        return False
    
    return False


def check_checkpoint_files(output_dir):
    """Check if output directory contains checkpoint files (*.pth)."""
    if not os.path.exists(output_dir):
        return False
    
    pth_files = glob.glob(os.path.join(output_dir, "**", "*.pth"), recursive=True)
    return len(pth_files) > 0


def should_skip_task(output_dir, skip_exist, skip_pattern):
    """Determine if task should be skipped and return skip mode."""
    if skip_exist >= 1 and check_skip_pattern(output_dir, skip_pattern):
        return 'skip', None
    elif skip_exist >= 2 and check_checkpoint_files(output_dir):
        return 'resume', output_dir
    else:
        return 'run', None


def execute_task_with_resource(args):
    """Execute a single training task with resource management."""
    task, common_info, hardware_params, resource_queue, skip_exist, skip_pattern = args
    
    resource = None
    try:
        # Set task parameters
        task_params = common_info['TASK_PARAMS'].copy()
        task_params['EXP_IDX'] = task['EXP_IDX']
        if 'OUTPUT_DIR' in task_params:
            task_params['OUTPUT_DIR'] = task_params['OUTPUT_DIR'].format(**task_params)
        else:
            task_params['OUTPUT_DIR'] = f'{task_params["OUTPUT_ROOT"]}/{task_params["MODEL_IDX"]}.{task_params["EXP_IDX"]}'
        
        # Check if task should be skipped
        skip_mode, resume_dir = should_skip_task(task_params['OUTPUT_DIR'], skip_exist, skip_pattern)
        
        if skip_mode == 'skip':
            print(f"Task {task['EXP_IDX']} skipped (output exists)")
            return {
                'exp_idx': task['EXP_IDX'],
                'returncode': 0,
                'skipped': True,
                'stdout': '',
                'stderr': ''
            }
        
        # Acquire resource from queue
        resource = resource_queue.get()
        
        task_params.update({
            'CUDA_DEVICES': resource['CUDA_DEVICES'],
            'CPU_AFFINITY': resource['CPU_AFFINITY'],
            'MASTER_PORT': resource['MASTER_PORT']
        })

        # Prepare environment variables
        env = os.environ.copy()
        
        # Update environment with task parameters
        for key, value in task_params.items():
            env[key] = str(value)
        
        # Update environment with hardware parameters
        for key, value in hardware_params.items():
            if key not in ['NUM_GPUS']:  # Skip NUM_GPUS as it's not a command parameter
                env[key] = str(value)
        
        # Prepare command line parameters
        fixed_params = format_params_for_command(common_info['FIXED_PARAMS'])
        search_params = format_params_for_command(task['SEARCH_PARAMS'])
        hw_params = format_params_for_command({k: v for k, v in hardware_params.items() 
                                              if k not in ['NUM_GPUS', 'NUM_PROC', 'NUM_TASKS']})
        
        # Add resume parameter if needed
        if skip_mode == 'resume':
            fixed_params.extend(['--resume', resume_dir])
            print(f"Task {task['EXP_IDX']} resuming from {resume_dir}")
        else:
            print(f"Task {task['EXP_IDX']} starting fresh")
        
        # Create training script with substituted variables
        train_script = string.Template(common_info['TRAIN_SCRIPT']).safe_substitute(env)
        
        # Replace parameter placeholders in script
        train_script = train_script.replace('$FIXED_PARAMS', ' '.join(fixed_params))
        train_script = train_script.replace('$HARDWARE_PARAMS', ' '.join(hw_params))
        train_script = train_script.replace('$SEARCH_PARAMS', ' '.join(search_params))
        
        # Execute the script
        result = subprocess.run(
            train_script,
            shell=True,
            env=env,
            capture_output=True,
            text=True
        )
        
        status_msg = "resumed and completed" if skip_mode == 'resume' else "completed"
        print(f"Task {task['EXP_IDX']} {status_msg} with return code {result.returncode}")
        if result.returncode != 0:
            print(f"Task {task['EXP_IDX']} stderr: {result.stderr}")
        
        return {
            'exp_idx': task['EXP_IDX'],
            'returncode': result.returncode,
            'resumed': skip_mode == 'resume',
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        print(f"Task {task['EXP_IDX']} failed with exception: {e}")
        traceback.print_exc()
        return {
            'exp_idx': task['EXP_IDX'],
            'returncode': -1,
            'error': str(e)
        }
    finally:
        # Always release resource back to queue
        if resource is not None:
            resource_queue.put(resource)


def execute_task(task_info):
    """Execute a single training task."""
    return execute_task_with_resource(task_info)


def run_tasks(task_file, hardware, skip_exist=0, skip_pattern='.*'):
    """Run hyperparameter search tasks."""
    # Load task data
    common_info, tasks = load_task_data(task_file)
    
    # Get hardware parameters
    hardware_params = common_info['HARDWARE_PARAMS'][hardware]
    
    # Create multiprocessing manager
    manager = mp.Manager()
    
    # Create resource queue
    resource_queue, num_tasks = create_resource_queue(hardware_params, manager)
    
    print(f"Running {len(tasks)} tasks on {hardware} with {num_tasks} parallel workers")
    print(f"Skip mode: {skip_exist}, Skip pattern: {skip_pattern}")
    
    # Prepare task arguments with resource queue
    task_args = []
    for task in tasks:
        task_args.append((task, common_info, hardware_params, resource_queue, skip_exist, skip_pattern))
    
    # Execute tasks
    with ProcessPoolExecutor(max_workers=num_tasks) as executor:
        results = list(executor.map(execute_task, task_args))
    
    # Print summary
    successful = sum(1 for r in results if r['returncode'] == 0)
    skipped = sum(1 for r in results if r.get('skipped', False))
    resumed = sum(1 for r in results if r.get('resumed', False))
    
    print(f"Completed {len(tasks)} tasks: {successful} successful, {len(tasks) - successful} failed")
    print(f"Tasks skipped: {skipped}, Tasks resumed: {resumed}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter search tasks')
    parser.add_argument('task_file', help='Path to task pickle file')
    parser.add_argument('--hardware', required=True, help='Hardware configuration to use')
    parser.add_argument('--skip_exist', type=int, default=0, choices=[0, 1, 2],
                       help='Skip existing tasks: 0=run all, 1=skip if pattern match, 2=skip or resume')
    parser.add_argument('--skip_pattern', default='.*',
                       help='Regex pattern to match files for skipping (default: .*)')
    
    args = parser.parse_args()
    
    run_tasks(args.task_file, args.hardware, args.skip_exist, args.skip_pattern)


if __name__ == '__main__':
    main()
