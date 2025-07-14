import os
import json
import pickle
import csv
import argparse
import importlib.util
import numpy as np
from pathlib import Path


def load_config(config_path):
    """Load configuration from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def sample_search_params(search_params, seed=None):
    """Sample parameters from search space."""
    if seed is not None:
        np.random.seed(seed)
    
    sampled = {}
    for key, value in search_params.items():
        if isinstance(value, dict):
            # Dict format: parameter -> weight mapping
            params = list(value.keys())
            weights = np.array(list(value.values()))
            weights = weights / weights.sum()  # Normalize weights
            # Use integer indices for choice, then select the actual parameter
            idx = np.random.choice(len(params), p=weights)
            sampled[key] = params[idx]
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            # Check if it's a list of lists/tuples (for nargs parameters)
            if len(value) > 0 and isinstance(value[0], (list, tuple)):
                idx = np.random.choice(len(value))
                sampled[key] = value[idx]
            else:
                # Convert to 1D array for np.random.choice
                value_list = list(value)
                if len(value_list) > 0:
                    # Handle numpy arrays and other types
                    if isinstance(value, np.ndarray):
                        sampled[key] = np.random.choice(value)
                    else:
                        idx = np.random.choice(len(value_list))
                        sampled[key] = value_list[idx]
                else:
                    sampled[key] = None
        else:
            sampled[key] = value
    
    return sampled


def format_param_value(value):
    """Format parameter value for command line."""
    if isinstance(value, (list, tuple)):
        return ' '.join(map(str, value))
    return str(value)


def generate_tasks(config_path, task_name=None, exp_idx=None, num_exps=128, seed=None):
    """Generate hyperparameter search tasks."""
    config = load_config(config_path)
    
    # Set task name
    if task_name is None:
        task_name = Path(config_path).stem
    
    # Set initial experiment index
    if exp_idx is None:
        exp_idx = int(config.TASK_PARAMS['EXP_IDX'])
    
    # Set random seed
    if seed is None:
        seed = exp_idx
    
    # Create tasks directory
    tasks_dir = Path('tasks')
    tasks_dir.mkdir(exist_ok=True)
    
    # Generate task filename prefix
    task_prefix = f"{task_name}_{exp_idx:04d}_{exp_idx + num_exps - 1:04d}"
    
    # Generate common task info (JSON)
    common_info = {
        'TRAIN_SCRIPT': config.TRAIN_SCRIPT,
        'TASK_PARAMS': config.TASK_PARAMS,
        'HARDWARE_PARAMS': config.HARDWARE_PARAMS,
        'FIXED_PARAMS': config.FIXED_PARAMS
    }
    
    json_path = tasks_dir / f"{task_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(common_info, f, indent=2)
    
    # Generate individual task parameters
    tasks = []
    task_params_list = []
    
    for i in range(num_exps):
        current_exp_idx = exp_idx + i
        current_seed = seed + i
        
        # Sample search parameters
        sampled_params = sample_search_params(config.SEARCH_PARAMS, current_seed)
        
        # Create task
        task = {
            'EXP_IDX': f"{current_exp_idx:04d}",
            'SEARCH_PARAMS': sampled_params
        }
        
        tasks.append(task)
        
        # Prepare for CSV
        csv_row = {'EXP_IDX': task['EXP_IDX']}
        csv_row.update(sampled_params)
        task_params_list.append(csv_row)
    
    # Save tasks as pickle
    pkl_path = tasks_dir / f"{task_prefix}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(tasks, f)
    
    # Save task parameters as CSV
    csv_path = tasks_dir / f"{task_prefix}.csv"
    if task_params_list:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=task_params_list[0].keys())
            writer.writeheader()
            writer.writerows(task_params_list)
    
    print(f"Generated {num_exps} tasks:")
    print(f"  Common info: {json_path}")
    print(f"  Task data: {pkl_path}")
    print(f"  Parameters: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate hyperparameter search tasks')
    parser.add_argument('config', help='Path to configuration file')
    parser.add_argument('--task_name', help='Task name (default: config filename)')
    parser.add_argument('--exp_idx', type=int, help='Initial experiment index')
    parser.add_argument('--num_exps', type=int, default=128, help='Number of experiments')
    parser.add_argument('--seed', type=int, help='Random seed (default: initial exp_idx)')
    
    args = parser.parse_args()
    
    generate_tasks(
        config_path=args.config,
        task_name=args.task_name,
        exp_idx=args.exp_idx,
        num_exps=args.num_exps,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
