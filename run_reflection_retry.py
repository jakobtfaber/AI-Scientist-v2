import os
import sys

# Ensure compiled dependencies are used if needed
sys.path.insert(0, '/data/ai-tools/SakanaAI/AI-Scientist-v2')

# Set API key
os.environ['GEMINI_API_KEY'] = 'AIzaSyAPSRI-HHsFBwk6cwSYd_NSSJ16ROiKYFI'

from ai_scientist.perform_icbinb_writeup import perform_writeup

print('[bold green]Starting Writeup with Reflection (Script Run)...[/bold green]', flush=True)

try:
    result = perform_writeup(
        base_folder='/data/ai-tools/SakanaAI/AI-Scientist-v2/experiments/2026-02-02_23-55-13_gpu_ffa_pulsar_attempt_0',
        small_model="gemini-2.5-flash",
        big_model="gemini-2.5-pro",
        n_writeup_reflections=3,
        page_limit=6,
    )
    print(f'\n{"="*80}')
    print(f'FINAL RESULT: {result}')
    print(f'{"="*80}')
except Exception as e:
    print(f"CRITICAL FAILURE IN SCRIPT: {e}")
    import traceback
    traceback.print_exc()
