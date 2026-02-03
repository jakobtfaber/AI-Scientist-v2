#!/usr/bin/env python3
"""
Standalone script to rerun Stage 5 (Writeup) for a completed experiment.
This script calls perform_writeup directly with the specified models.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_scientist.perform_icbinb_writeup import perform_writeup

def main():
    # Configuration
    experiment_dir = "/data/ai-tools/SakanaAI/AI-Scientist-v2/experiments/2026-02-02_23-55-13_gpu_ffa_pulsar_attempt_0"
    
    # Model configuration - using Gemini instead of o1-preview
    small_model = "gpt-4o-2024-05-13"  # For citation gathering
    big_model = "gemini-2.0-flash"  # For writeup generation - stable Gemini model

    
    print("=" * 80)
    print("RERUNNING STAGE 5: WRITEUP")
    print("=" * 80)
    print(f"Experiment: {experiment_dir}")
    print(f"Small model (citations): {small_model}")
    print(f"Big model (writeup): {big_model}")
    print("=" * 80)
    print()
    
    # Ensure GEMINI_API_KEY is set
    if "GEMINI_API_KEY" not in os.environ:
        print("ERROR: GEMINI_API_KEY environment variable not set!")
        print("Please run: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Run the writeup
    success = perform_writeup(
        base_folder=experiment_dir,
        citations_text=None,  # Will be gathered automatically
        no_writing=False,
        num_cite_rounds=20,
        small_model=small_model,
        big_model=big_model,
        n_writeup_reflections=3,
        page_limit=4,
    )
    
    if success:
        print("\n" + "=" * 80)
        print("✓ WRITEUP COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Check the experiment directory for the final PDF:")
        print(f"  {experiment_dir}")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ WRITEUP FAILED")
        print("=" * 80)
        print("Check the logs above for error details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
