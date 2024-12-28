import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def execute_command(command, log_file):
    """Execute a shell command and log the output to a file."""
    with open(log_file, 'w') as log:
        process = subprocess.run(command, shell=True, stdout=log, stderr=log, text=True)
    return command, process.returncode

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create noisy labels')
    parser.add_argument('--dataset', default='pathmnist', help='data set under test', type=str)
    args = parser.parse_args()

    accuracies = [80, 85, 90, 95]
    commands = [
        f"python /home/dsi/rotemnizhar/dev/python_scripts/noisy_training/general_noisy_labels.py --dataset {args.dataset} --acc {acc}"
        for acc in accuracies
    ]
    commands.append(f"python /home/dsi/rotemnizhar/dev/python_scripts/noisy_training/feature_map.py --dataset {args.dataset}")

    # Run commands in parallel
    log_files = [f"{args.dataset}_command_{i}.log" for i in range(len(commands))]  # Generate unique log file names
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        future_to_command = {executor.submit(execute_command, cmd, log): cmd for cmd, log in zip(commands, log_files)}

        for future in as_completed(future_to_command):
            cmd = future_to_command[future]
            try:
                command, returncode = future.result()
                print(f"Command '{command}' completed with return code {returncode}")
            except Exception as e:
                print(f"Command '{cmd}' generated an exception: {e}")