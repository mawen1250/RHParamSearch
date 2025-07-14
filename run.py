import os
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


def create_resource_queue(hardware_params):
    """Create a queue of computing resources."""
    num_gpus = hardware_params['NUM_GPUS']
    num_proc = hardware_params['NUM_PROC']
    num_tasks = num_gpus // num_proc
    
    cpu_count = os.cpu_count()
    cpus_per_task = cpu_count // num_tasks
    
    base_port = 29540  # Default master port
    
    resources = []
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
        
        resources.append({
            'CUDA_DEVICES': cuda_devices,
            'CPU_AFFINITY': cpu_affinity,
            'MASTER_PORT': master_port
        })
    
    return resources, num_tasks


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


def execute_task_with_resource(args):
    """Execute a single training task with resource management."""
    task, common_info, hardware_params, resources, task_index = args
    
    # Get resource for this task
    resource = resources[task_index % len(resources)]
    
    # Prepare environment variables
    env = os.environ.copy()
    
    # Set task parameters
    task_params = common_info['TASK_PARAMS'].copy()
    task_params.update({
        'EXP_IDX': task['EXP_IDX'],
        'CUDA_DEVICES': resource['CUDA_DEVICES'],
        'CPU_AFFINITY': resource['CPU_AFFINITY'],
        'MASTER_PORT': resource['MASTER_PORT']
    })
    
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
    
    # Create training script with substituted variables
    train_script = string.Template(common_info['TRAIN_SCRIPT']).safe_substitute(env)
    
    # Replace parameter placeholders in script
    train_script = train_script.replace('$FIXED_PARAMS', ' '.join(fixed_params))
    train_script = train_script.replace('$HARDWARE_PARAMS', ' '.join(hw_params))
    train_script = train_script.replace('$SEARCH_PARAMS', ' '.join(search_params))
    
    # Execute the script
    try:
        result = subprocess.run(
            train_script,
            shell=True,
            env=env,
            capture_output=True,
            text=True
        )
        
        print(f"Task {task['EXP_IDX']} completed with return code {result.returncode}")
        if result.returncode != 0:
            print(f"Task {task['EXP_IDX']} stderr: {result.stderr}")
        
        return {
            'exp_idx': task['EXP_IDX'],
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        print(f"Task {task['EXP_IDX']} failed with exception: {e}")
        return {
            'exp_idx': task['EXP_IDX'],
            'returncode': -1,
            'error': str(e)
        }


def execute_task(task_info):
    """Execute a single training task."""
    return execute_task_with_resource(task_info)


def run_tasks(task_file, hardware):
    """Run hyperparameter search tasks."""
    # Load task data
    common_info, tasks = load_task_data(task_file)
    
    # Get hardware parameters
    hardware_params = common_info['HARDWARE_PARAMS'][hardware]
    
    # Create resource queue
    resources, num_tasks = create_resource_queue(hardware_params)
    
    print(f"Running {len(tasks)} tasks on {hardware} with {num_tasks} parallel workers")
    
    # Prepare task arguments with resource allocation
    task_args = []
    for i, task in enumerate(tasks):
        task_args.append((task, common_info, hardware_params, resources, i))
    
    # Execute tasks
    with ProcessPoolExecutor(max_workers=num_tasks) as executor:
        results = list(executor.map(execute_task, task_args))
    
    # Print summary
    successful = sum(1 for r in results if r['returncode'] == 0)
    print(f"Completed {len(tasks)} tasks: {successful} successful, {len(tasks) - successful} failed")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter search tasks')
    parser.add_argument('task_file', help='Path to task pickle file')
    parser.add_argument('--hardware', required=True, help='Hardware configuration to use')
    
    args = parser.parse_args()
    
    run_tasks(args.task_file, args.hardware)


if __name__ == '__main__':
    main()
