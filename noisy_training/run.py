import subprocess
import itertools
import os

# Define lists of models, datasets, and accuracies
models = ['densenet121']
datasets = ['bloodmnist']
accuracies = [80, 85, 90, 95]

# Path to the Python script you want to run
script_path = '/home/dsi/rotemnizhar/dev/python_scripts/noisy_training/train.py'

# Generate all combinations of models, datasets, and accuracies
combinations = list(itertools.product(models, datasets, accuracies))

# Iterate over each combination and run the script
for model, dataset, acc in combinations:
    # Log file name for this combination
    log_file = f"{model}_{dataset}_{acc}_log.txt"
    
    # Build the command
    command = [
        'python', script_path,
        '--model_name', model,
        '--dataset', dataset,
        '--acc', str(acc)
    ]
    
    print(f"Running: {' '.join(command)}")
    
    # Open log file for writing
    with open(log_file, 'w') as log:
        # Run the script with the generated command and write output/errors to the log file
        result = subprocess.run(command, stdout=log, stderr=log, text=True)
    
    # Check the log file content for review (optional)
    with open(log_file, 'r') as log:
        print(f"Log file content for combination {model}, {dataset}, {acc}:\n")
        print(log.read())
    
    # Delete the log file after processing
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"Log file {log_file} deleted.")
    else:
        print(f"Log file {log_file} not found!")

print("All combinations processed.")