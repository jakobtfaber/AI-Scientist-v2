import os
import numpy as np
import time
import torch

# Setting up the working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define experimental data storage
experiment_data = {
    "synthetic_pulsar": {
        "metrics": {"speedup_factor": None},
        "data": {"input": None, "cpu_output": None, "gpu_output": None},
    }
}


# Create synthetic data
def generate_synthetic_data(size=1024):
    return np.sin(np.linspace(0, 10 * np.pi, size)) + np.random.normal(0, 0.1, size)


synthetic_data = generate_synthetic_data(size=1024)
experiment_data["synthetic_pulsar"]["data"]["input"] = synthetic_data
np.save(os.path.join(working_dir, "synthetic_data.npy"), synthetic_data)


# CPU-based Fast Folding Algorithm (simplified version for benchmarking)
def ffa_cpu(data, n_bins):
    folded = np.zeros((n_bins, len(data) // n_bins))
    for i in range(n_bins):
        folded[i] = data[i::n_bins][: folded.shape[1]]
    return folded.mean(axis=0)


# GPU-based Fast Folding Algorithm (simplified with PyTorch)
def ffa_gpu(data, n_bins):
    data_gpu = torch.tensor(data, device=device)
    folded_gpu = torch.zeros((n_bins, len(data) // n_bins), device=device)
    for i in range(n_bins):
        folded_gpu[i] = data_gpu[i::n_bins][: folded_gpu.shape[1]]
    return folded_gpu.mean(dim=0).cpu().numpy()


# CPU Benchmark
start_cpu = time.time()
cpu_result = ffa_cpu(synthetic_data, n_bins=16)
cpu_time = time.time() - start_cpu
experiment_data["synthetic_pulsar"]["data"]["cpu_output"] = cpu_result

# GPU Benchmark
synthetic_data_tensor = torch.tensor(synthetic_data, device=device)
start_gpu = time.time()
gpu_result = ffa_gpu(synthetic_data, n_bins=16)
gpu_time = time.time() - start_gpu
experiment_data["synthetic_pulsar"]["data"]["gpu_output"] = gpu_result

# Compute speedup factor
speedup_factor = cpu_time / gpu_time
experiment_data["synthetic_pulsar"]["metrics"]["speedup_factor"] = speedup_factor

# Save experiment results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Print summary
print(
    f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s, Speedup Factor: {speedup_factor:.2f}"
)
