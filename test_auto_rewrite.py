#!/usr/bin/env python3
"""
Test automated claim rewriting on a paper with validation failures
"""

import sys
import logging
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ai_scientist.data_loader import load_experiment_data
from ai_scientist.validators.consistency_checker import ConsistencyChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

def test_auto_rewrite(experiment_dir: str):
    """Test auto-rewrite feature on an experiment"""
    
    print(f"Testing auto-rewrite on: {experiment_dir}")
    print("=" * 80)
    
    # Find LaTeX file
    exp_path = Path(experiment_dir)
    latex_files = list(exp_path.glob("latex/*.tex")) + list(exp_path.glob("*.tex"))
    latex_files = [f for f in latex_files if f.name not in ["iclr2025_conference.tex", "template.tex"]]
    
    if not latex_files:
        print("ERROR: No LaTeX file found in experiment directory")
        return
    
    latex_file = latex_files[0]
    print(f"\n1. Loading LaTeX file: {latex_file.name}")
    
    # Load experiment data
    print(f"\n2. Loading experimental data...")
    try:
        exp_data = load_experiment_data(experiment_dir)
        print(f"   Loaded {len(exp_data.get('files', []))} data files")
    except Exception as e:
        print(f"ERROR: Failed to load experiment data: {e}")
        return
    
    # Initialize checker
    print(f"\n3. Running validation...")
    figures_dir = exp_path / "figures"
    checker = ConsistencyChecker(
        latex_path=str(latex_file),
        data=exp_data,
        figures_dir=str(figures_dir) if figures_dir.exists() else None
    )
    
    # Run validation with auto-fix
    print(f"\n4. Running validation with auto-fix enabled...")
    report, corrected_latex = checker.run_all_checks_with_autofix(
        latex_path=None,  # Don't save yet
        apply_fixes=True
    )
    
    # Display results
    print("\n" + "=" * 80)
    print(report.summary())
    print("=" * 80)
    
    if corrected_latex:
        print(f"\n✓ Auto-rewrite produced corrected LaTeX ({len(corrected_latex)} chars)")
        
        # Save corrected version
        corrected_path = latex_file.parent / f"{latex_file.stem}_autocorrected.tex"
        with open(corrected_path, 'w') as f:
            f.write(corrected_latex)
        print(f"✓ Saved corrected version to: {corrected_path}")
        
        # Run validation again on corrected version
        print(f"\n5. Re-validating corrected LaTeX...")
        checker_v2 = ConsistencyChecker(
            latex_path=str(corrected_path),
            data=exp_data,
            figures_dir=str(figures_dir) if figures_dir.exists() else None
        )
        report_v2 = checker_v2.run_all_checks()
        
        print("\n" + "=" * 80)
        print("AFTER AUTO-FIX:")
        print(report_v2.summary())
        print("=" * 80)
        
        # Compare
        improvement = report.failed_claims - report_v2.failed_claims
        if improvement > 0:
            print(f"\n🎯 AUTO-FIX SUCCESS: Fixed {improvement} claims!")
        else:
            print(f"\n⚠️  No improvement from auto-fix")
    else:
        print(f"\n�� No failed claims to auto-fix, or auto-fix was not applied")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_auto_rewrite.py <experiment_dir>")
        print("\nExample:")
        print("  python test_auto_rewrite.py experiments/2026-02-02_23-55-13_gpu_ffa_pulsar_attempt_0")
        sys.exit(1)
    
    test_auto_rewrite(sys.argv[1])
