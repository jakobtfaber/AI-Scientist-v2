"""Test/demo script for new validation system"""

import logging
from pathlib import Path

from ai_scientist.data_loader import load_experiment_data, format_data_for_prompt
from ai_scientist.validators.consistency_checker import ConsistencyChecker
from ai_scientist.reviewers.review_pipeline import ReviewPipeline
from ai_scientist.linters.experiment_linter import ExperimentLinter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_validation_system(experiment_dir: str):
    """
    Test the new validation system on an experiment
    
    Args:
        experiment_dir: Path to experiment directory
    """
    logger.info(f"Testing validation system on: {experiment_dir}")
    logger.info("=" * 80)
    
    # Step 1: Load experimental data
    logger.info("\n1. Loading experimental data...")
    data = load_experiment_data(experiment_dir)
    logger.info(f"   Loaded {len(data['experiment_data'])} data files")
    
    # Step 2: Format data for prompts
    logger.info("\n2. Formatting data for LLM prompts...")
    formatted = format_data_for_prompt(data, max_tokens=500)
    logger.info(f"   Formatted data ({len(formatted)} chars):")
    print(formatted[:500] + "...")
    
    # Step 3: Lint experiment code
    logger.info("\n3. Linting experiment code...")
    code_files = list(Path(experiment_dir).glob("**/*experiment*.py"))
    if code_files:
        linter = ExperimentLinter()
        lint_report = linter.lint(str(code_files[0]))
        print(lint_report.summary())
    else:
        logger.warning("   No experiment code found to lint")
    
    # Step 4: Consistency check (if LaTeX exists)
    logger.info("\n4. Running consistency checks...")
    latex_files = list(Path(experiment_dir).glob("**/*.tex"))
    if latex_files:
        checker = ConsistencyChecker(
            latex_path=str(latex_files[0]),
            data=data,
            figures_dir=str(Path(experiment_dir) / "figures")
        )
        
        claims = checker.extract_claims()
        logger.info(f"   Found {len(claims)} claims in paper")
        
        if claims:
            validation_report = checker.run_all_checks()
            print(validation_report.summary())
    else:
        logger.warning("   No LaTeX file found for consistency check")
    
    # Step 5: Multi-agent review (if LaTeX exists)
    logger.info("\n5. Running multi-agent review...")
    if latex_files:
        with open(latex_files[0]) as f:
            paper_draft = f.read()
        
        pipeline = ReviewPipeline(model="gemini-2.5-flash")  # Use flash for demo
        aggregated_review = pipeline.review_paper(
            paper_draft=paper_draft,
            data=data,
            code_path=str(code_files[0]) if code_files else None,
            figures_dir=str(Path(experiment_dir) / "figures")
        )
        
        print(aggregated_review.summary())
    else:
        logger.warning("   No LaTeX file found for review")
    
    logger.info("\n" + "=" * 80)
    logger.info("Validation system test complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_validation.py <experiment_directory>")
        print("\nExample:")
        print("  python test_validation.py /data/ai-tools/SakanaAI/AI-Scientist-v2/experiments/2026-02-02_23-55-13_gpu_ffa_pulsar_attempt_0")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    test_validation_system(experiment_dir)
