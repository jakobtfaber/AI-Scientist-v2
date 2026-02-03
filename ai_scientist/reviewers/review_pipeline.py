"""Orchestrate multi-agent review pipeline"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

from ai_scientist.reviewers.base_reviewer import ReviewReport
from ai_scientist.reviewers.statistics_reviewer import StatisticsReviewer
from ai_scientist.reviewers.methods_reviewer import MethodsReviewer
from ai_scientist.reviewers.adversarial_reviewer import AdversarialReviewer

logger = logging.getLogger(__name__)


@dataclass
class AggregatedReview:
    """Aggregated results from all reviewers"""
    all_reviews: List[ReviewReport]
    critical_issues: List[ReviewReport] = field(default_factory=list)
    major_issues: List[ReviewReport] = field(default_factory=list)
    recommendation: str = "accept"
    
    def has_critical_issues(self) -> bool:
        return len(self.critical_issues) > 0
    
    def summary(self) -> str:
        lines = [
            "=== Aggregated Review Summary ===",
            f"Total reviewers: {len(self.all_reviews)}",
            f"Overall recommendation: {self.recommendation.upper()}",
            ""
        ]
        
        for review in self.all_reviews:
            lines.append(f"{review.reviewer_name}:")
            lines.append(f"  {review.issue_summary()}")
            lines.append(f"  Recommendation: {review.recommendation}")
            lines.append("")
        
        if self.critical_issues:
            lines.append("CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            for review in self.critical_issues:
                for issue in review.issues:
                    if issue.severity == "critical":
                        lines.append(f"  - [{review.reviewer_name}]  {issue.issue}")
        
        return "\n".join(lines)


class ReviewPipeline:
    """Orchestrate multi-agent review process"""
    
    def __init__(self, model: str = "gemini-3-pro-preview"):
        """
        Initialize review pipeline
        
        Args:
            model: LLM model to use for all reviewers
        """
        self.reviewers = [
            StatisticsReviewer(model=model),
            MethodsReviewer(model=model),
            AdversarialReviewer(model=model)
        ]
        logger.info(f"Initialized review pipeline with {len(self.reviewers)} reviewers")
    
    def review_paper(
        self,
        paper_draft: str,
        data: Dict[str, Any],
        code_path: str = None,
        figures_dir: str = None
    ) -> AggregatedReview:
        """
        Run all reviewers and aggregate results
        
        Args:
            paper_draft: LaTeX content of paper
            data: Experimental data dictionary
            code_path: Path to experiment code
            figures_dir: Path to figures directory
            
        Returns:
            AggregatedReview with all findings
        """
        logger.info("Starting multi-agent review...")
        
        reviews = []
        
        for reviewer in self.reviewers:
            logger.info(f"Running {reviewer.name} review...")
            
            try:
                review = reviewer.review(
                    paper_draft=paper_draft,
                    data=data,
                    code_path=code_path,
                    figures_dir=figures_dir
                )
                reviews.append(review)
                logger.info(f"  {reviewer.name}: {review.issue_summary()}")
                
            except Exception as e:
                logger.error(f"  {reviewer.name} failed: {e}")
                # Continue with other reviewers
        
        # Aggregate results
        critical_reviews = [r for r in reviews if r.has_critical_issues()]
        major_reviews = [r for r in reviews if r.has_major_issues()]
        
        # Determine overall recommendation (most conservative wins)
        recommendations = [r.recommendation for r in reviews]
        if "reject" in recommendations:
            overall_recommendation = "reject"
        elif "major_revision" in recommendations:
            overall_recommendation = "major_revision"
        elif "minor_revision" in recommendations:
            overall_recommendation = "minor_revision"
        else:
            overall_recommendation = "accept"
        
        aggregated = AggregatedReview(
            all_reviews=reviews,
            critical_issues=critical_reviews,
            major_issues=major_reviews,
            recommendation=overall_recommendation
        )
        
        logger.info(f"Review complete: {aggregated.recommendation.upper()}")
        
        return aggregated
