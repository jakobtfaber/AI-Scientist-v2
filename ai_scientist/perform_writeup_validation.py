"""
Validation integration helper for perform_writeup.py

This module provides the validation logic to be called after initial writeup generation.
"""

import traceback
from typing import Dict, Any


def run_validation_checks(
    base_folder: str,
    writeup_file: str,
    updated_latex_code: str,
    figures_dir: str,
    aggregator_path: str,
    big_model: str
) -> str:
    """
    Run consistency checks and multi-agent review on generated writeup.
    
    Args:
        base_folder: Experiment base folder
        writeup_file: Path to LaTeX file
        updated_latex_code: Generated LaTeX content
        figures_dir: Path to figures directory
        aggregator_path: Path to aggregator script
        big_model: Model name for review pipeline
        
    Returns:
        validation_feedback: String with validation findings to inject into reflection prompt
    """
    from ai_scientist.data_loader import load_experiment_data
    from ai_scientist.validators.consistency_checker import ConsistencyChecker
    from ai_scientist.reviewers.review_pipeline import ReviewPipeline
    import os.path as osp
    
    print("[VALIDATION] Running consistency and review checks...")
    validation_feedback = ""
    
    try:
        # Load experimental data
        exp_data = load_experiment_data(base_folder)
        
        # 1. Consistency check
        checker = ConsistencyChecker(
            latex_path=writeup_file,
            data=exp_data,
            figures_dir=figures_dir
        )
        validation_report_obj = checker.run_all_checks()
        
        if validation_report_obj.has_critical_errors():
            print(f"[VALIDATION] ⚠️  Found {len(validation_report_obj.critical_issues)} critical consistency issues!")
            critical_summary = "\n".join([
                f"  - {result.claim.text[:80]}... (Reason: {result.reason})"
                for result in validation_report_obj.critical_issues
            ])
            validation_feedback += f"\n\n**CRITICAL CONSISTENCY ISSUES:**\n{critical_summary}\n"
        else:
            print("[VALIDATION] ✓ No critical consistency issues found")
        
        # 2. Multi-agent review (only if relatively clean)
        if len(validation_report_obj.critical_issues) <= 2:
            print("[VALIDATION] Running multi-agent review...")
            pipeline = ReviewPipeline(model=big_model)
            aggregated_review = pipeline.review_paper(
                paper_draft=updated_latex_code,
                data=exp_data,
                code_path=aggregator_path if osp.exists(aggregator_path) else None,
                figures_dir=figures_dir
            )
            
            if aggregated_review.has_critical_issues():
                print(f"[VALIDATION] ⚠️  Multi-agent review found {len(aggregated_review.critical_issues)} critical issues!")
                review_summary = aggregated_review.summary()
                validation_feedback += f"\n\n**MULTI-AGENT REVIEW FINDINGS:**\n{review_summary[:1000]}...\n"
            else:
                print("[VALIDATION] ✓ Multi-agent review passed")
        
    except Exception:
        print("[VALIDATION] Exception during validation (continuing anyway):")
        print(traceback.format_exc())
    
    return validation_feedback
