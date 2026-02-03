#!/usr/bin/env python3
"""
Direct call to perform_writeup to capture exceptions
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["GEMINI_API_KEY"] = "AIzaSyAPSRI-HHsFBwk6cwSYd_NSSJ16ROiKYFI"

from ai_scientist.perform_icbinb_writeup import perform_writeup

exp_dir = "/data/ai-tools/SakanaAI/AI-Scientist-v2/experiments/2026-02-02_23-55-13_gpu_ffa_pulsar_attempt_0"

try:
    result = perform_writeup(
        folder_name=exp_dir,
        small_client_model="gpt-4o-2024-05-13",
        big_client_model="gemini-3-pro-preview",
        n_writeup_reflections=3,
        vlm_model="gpt-4o-2024-05-13",  # Used for PDF image review
        page_limit=6,
    )
    
    print(f"\n{'='*80}")
    if result:
        print("✅ WRITEUP SUCCEEDED")
    else:
        print("✗ WRITEUP RETURNED FALSE")
    print(f"{'='*80}")
    
except Exception as e:
    print(f"\n{'='*80}")
    print("EXCEPTION CAUGHT:")
    print(traceback.format_exc())
    print(f"{'='*80}")
