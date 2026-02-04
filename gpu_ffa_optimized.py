import os
import random
import numpy as np
import torch
import time

# Set random seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment data structure
experiment_data = {
    "hyperparam_tuning_num_samples": {
        "synthetic_dataset": {
            "num_samples": [],
            "cpu_runtime": [],
            "gpu_runtime": [],
            "metrics": {"speedup_factor": []},
        }
    }
}


# Generate synthetic periodic dataset
def generate_synthetic_data(num_samples=2**18, period=1024, noise_level=0.5):
    signal = np.zeros(num_samples)
    for i in range(0, num_samples, period):
        signal[i] = 1.0  # Impulse every 'period'
    noise = np.random.normal(0, noise_level, num_samples)
    return signal + noise


# Fast Folding Algorithm (CPU implementation)
def ffa_cpu(data, max_trial_period=2048):
    folded_signals = []
    for trial_period in range(2, max_trial_period + 1):
        folded = np.zeros(trial_period)
        for i, val in enumerate(data):
            folded[i % trial_period] += val
        folded_signals.append(folded)
    return folded_signals


# Fast Folding Algorithm (GPU implementation)
def ffa_gpu(data, max_trial_period=2048):
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    folded_signals = []
    for trial_period in range(2, max_trial_period + 1):
        indices = torch.arange(data_tensor.size(0), device=device) % trial_period
        folded = torch.zeros(trial_period, device=device).scatter_add_(
            0, indices, data_tensor
        )
        folded_signals.append(folded.cpu().numpy())
    return folded_signals


# Hyperparameter tuning for num_samples
sample_sizes = [2**16, 2**17, 2**18, 2**19]  # Different dataset sizes
for num_samples in sample_sizes:
    synthetic_data = generate_synthetic_data(num_samples=num_samples)
    experiment_data["hyperparam_tuning_num_samples"]["synthetic_dataset"][
        "num_samples"
    ].append(num_samples)

    # Measure runtime for CPU implementation
    start_cpu = time.time()
    ffa_cpu(synthetic_data)
    cpu_runtime = time.time() - start_cpu
    experiment_data["hyperparam_tuning_num_samples"]["synthetic_dataset"][
        "cpu_runtime"
    ].append(cpu_runtime)

    # Measure runtime for GPU implementation
    start_gpu = time.time()
    ffa_gpu(synthetic_data)
    gpu_runtime = time.time() - start_gpu
    experiment_data["hyperparam_tuning_num_samples"]["synthetic_dataset"][
        "gpu_runtime"
    ].append(gpu_runtime)

    # Calculate speedup factor
    speedup_factor = cpu_runtime / gpu_runtime
    experiment_data["hyperparam_tuning_num_samples"]["synthetic_dataset"]["metrics"][
        "speedup_factor"
    ].append(speedup_factor)

    print(
        f"num_samples={num_samples}, CPU runtime={cpu_runtime:.4f}s, GPU runtime={gpu_runtime:.4f}s, Speedup={speedup_factor:.4f}"
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
