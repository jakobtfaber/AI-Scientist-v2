"""Load and parse experimental results for writeup validation"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def load_experiment_data(base_folder: str) -> Dict[str, Any]:
    """
    Load all experiment data from a folder
    
    Returns unified dict with:
    - experiment_data: Results from .npy files
    - metadata: final_info.json if present
    - plots_data: plots.pkl if present
    
    Args:
        base_folder: Path to experiment directory
        
    Returns:
        Dictionary containing all experimental results
    """
    base_path = Path(base_folder)
    result = {
        "experiment_data": {},
        "metadata": {},
        "plots_data": None,
        "experiment_code": None
    }
    
    # Load .npy experiment data
    npy_files = list(base_path.glob("*.npy")) + list(base_path.glob("**/*.npy"))
    for npy_file in npy_files:
        try:
            data = np.load(npy_file, allow_pickle=True)
            # Convert to dict if it's a structured array
            if isinstance(data, np.ndarray) and data.dtype.names:
                data = {name: data[name] for name in data.dtype.names}
            elif isinstance(data, np.ndarray) and data.shape == ():
                # 0-d array containing dict
                data = data.item()
            
            result["experiment_data"][npy_file.stem] = data
            logger.info(f"Loaded experiment data from {npy_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load {npy_file}: {e}")
    
    # Load final_info.json
    final_info_path = base_path / "final_info.json"
    if final_info_path.exists():
        try:
            with open(final_info_path) as f:
                result["metadata"] = json.load(f)
            logger.info("Loaded final_info.json")
        except Exception as e:
            logger.warning(f"Failed to load final_info.json: {e}")
    
    # Load plots.pkl
    plots_path = base_path / "plots.pkl"
    if plots_path.exists():
        try:
            with open(plots_path, "rb") as f:
                result["plots_data"] = pickle.load(f)
            logger.info("Loaded plots.pkl")
        except Exception as e:
            logger.warning(f"Failed to load plots.pkl: {e}")
    
    # Load experiment code for reference
    code_files = list(base_path.glob("**/experiment_code.py"))
    if code_files:
        try:
            with open(code_files[0]) as f:
                result["experiment_code"] = f.read()
            logger.info(f"Loaded experiment code from {code_files[0]}")
        except Exception as e:
            logger.warning(f"Failed to load experiment code: {e}")
    
    return result


def format_data_for_prompt(data: Dict[str, Any], max_tokens: int = 2000) -> str:
    """
    Format experimental data for LLM prompt with token budget
    
    Prioritizes:
    1. Key metrics (speedup, runtime, accuracy)
    2. Summary statistics
    3. Truncated raw arrays if space permits
    
    Args:
        data: Dictionary of experimental results
        max_tokens: Approximate token budget (~4 chars/token)
        
    Returns:
        Formatted string suitable for prompt inclusion
    """
    max_chars = max_tokens * 4
    lines = []
    
    def add_section(title: str, content: str):
        section = f"\n## {title}\n{content}"
        if sum(len(l) for l in lines) + len(section) < max_chars:
            lines.append(section)
            return True
        return False
    
    # Extract experiment data
    exp_data = data.get("experiment_data", {})
    
    # Priority 1: Key metrics
    metrics = extract_key_metrics(exp_data)
    if metrics:
        content = json.dumps(metrics, indent=2)
        add_section("Key Metrics", content)
    
    # Priority 2: Summary statistics
    summaries = compute_summary_statistics(exp_data)
    if summaries:
        content = json.dumps(summaries, indent=2)
        add_section("Summary Statistics", content)
    
    # Priority 3: Metadata
    metadata = data.get("metadata", {})
    if metadata:
        # Filter to important fields
        important_fields = ["status", "num_runs", "timestamp", "config"]
        filtered_meta = {k: v for k, v in metadata.items() if k in important_fields}
        if filtered_meta:
            content = json.dumps(filtered_meta, indent=2)
            add_section("Experiment Metadata", content)
    
    # Priority 4: Raw data (truncated if needed)
    if sum(len(l) for l in lines) < max_chars * 0.8:
        raw_summary = summarize_raw_data(exp_data, max_items=10)
        if raw_summary:
            add_section("Raw Data (Sample)", raw_summary)
    
    return "\n".join(lines)


def extract_key_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics like speedup, runtime, accuracy from nested data"""
    metrics = {}
    
    def extract_recursive(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                
                # Check if this is a metric
                metric_keywords = [
                    "speedup", "runtime", "accuracy", "loss", "time",
                    "score", "metric", "performance", "throughput"
                ]
                
                if any(kw in key.lower() for kw in metric_keywords):
                    if isinstance(value, (int, float)):
                        metrics[new_prefix] = value
                    elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                        metrics[new_prefix] = {
                            "values": value[:10],  # Truncate
                            "mean": float(np.mean(value)),
                            "std": float(np.std(value))
                        }
                else:
                    extract_recursive(value, new_prefix)
        elif isinstance(obj, (list, np.ndarray)):
            if len(obj) > 0 and isinstance(obj[0], dict):
                extract_recursive(obj[0], f"{prefix}[0]")
    
    extract_recursive(data)
    return metrics


def compute_summary_statistics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary stats for numerical arrays"""
    summaries = {}
    
    def summarize_recursive(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                summarize_recursive(value, new_prefix)
        elif isinstance(obj, (list, np.ndarray)):
            try:
                arr = np.array(obj)
                if arr.dtype.kind in ('f', 'i', 'u'):  # float or int
                    summaries[prefix] = {
                        "count": len(arr),
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr))
                    }
            except:
                pass
    
    summarize_recursive(data)
    return summaries


def summarize_raw_data(data: Dict[str, Any], max_items: int = 10) -> str:
    """Create human-readable summary of raw data"""
    lines = []
    
    def format_value(v):
        if isinstance(v, (list, np.ndarray)):
            if len(v) > max_items:
                return f"[{', '.join(map(str, v[:max_items]))} ... ({len(v)} items)]"
            return str(v)
        elif isinstance(v, dict):
            return "{...}"  # Don't expand nested dicts
        return str(v)
    
    def summarize_recursive(obj, prefix="", depth=0):
        if depth > 3:  # Limit recursion
            return
        
        if isinstance(obj, dict):
            for key, value in list(obj.items())[:max_items]:
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    lines.append(f"{new_prefix}: {{...}}")
                    summarize_recursive(value, new_prefix, depth + 1)
                else:
                    lines.append(f"{new_prefix}: {format_value(value)}")
    
    summarize_recursive(data)
    return "\n".join(lines[:50])  # Limit output


def load_experiment_results_by_name(base_folder: str, experiment_name: str) -> Optional[Dict[str, Any]]:
    """
    Load specific experiment results by name
    
    Args:
        base_folder: Base experiment directory
        experiment_name: Name of specific experiment (e.g., 'impact_of_data_sampling_distribution')
        
    Returns:
        Experiment data if found, None otherwise
    """
    data = load_experiment_data(base_folder)
    exp_data = data.get("experiment_data", {})
    
    # Try exact match
    if experiment_name in exp_data:
        return exp_data[experiment_name]
    
    # Try partial match
    for key, value in exp_data.items():
        if experiment_name.lower() in key.lower():
            return value
    
    logger.warning(f"Experiment '{experiment_name}' not found in {list(exp_data.keys())}")
    return None
