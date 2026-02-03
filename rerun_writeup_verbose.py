#!/usr/bin/env python3
"""
Standalone script to rerun Stage 5 (Writeup) with verbose error handling
"""

import os
import sys
import traceback

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set the Gemini API key
os.environ["GEMINI_API_KEY"] = "AIzaSyAPSRI-HHsFBwk6cwSYd_NSSJ16ROiKYFI"

from ai_scientist.perform_icbinb_writeup import perform_writeup

def main():
    # Configuration
    experiment_dir = "/data/ai-tools/SakanaAI/AI-Scientist-v2/experiments/2026-02-02_23-55-13_gpu_ffa_pulsar_attempt_0"
    
    # Model configuration
    small_model = "gpt-4o-2024-05-13"  
    big_model = "gemini-3-pro-preview"  # Gemini 3 Pro - better for large prompts

    
    print("=" * 80)
    print("RERUNNING STAGE 5: WRITEUP (VERBOSE MODE)")
    print("=" * 80)
    print(f"Experiment: {experiment_dir}")
    print(f"Small model (citations): {small_model}")
    print(f"Big model (writeup): {big_model}")
    print(f"GEMINI_API_KEY: {os.environ.get('GEMINI_API_KEY', 'NOT SET')[:20]}...")
    print("=" * 80)
    print()
    
    try:
        # Run the writeup
        print("Starting perform_writeup...")
        success = perform_writeup(
            base_folder=experiment_dir,
            citations_text=None,
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
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("✗ WRITEUP FAILED (returned False)")
            print("=" * 80)
            sys.exit(1)
            
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ WRITEUP FAILED WITH EXCEPTION: {type(e).__name__}")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
