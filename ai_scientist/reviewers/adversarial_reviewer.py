"""Adversarial reviewer - actively searches for contradictions"""

from typing import Dict, Any
from ai_scientist.reviewers.base_reviewer import BaseReviewer, ReviewReport, ReviewIssue
from ai_scientist.prompt_templates import ADVERSARIAL_REVIEW_PROMPT
import re


class AdversarialReviewer(BaseReviewer):
    """Hostile reviewer actively searching for errors"""
    
    def review(
        self,
        paper_draft: str,
        data: Dict[str, Any],
        figures_dir: str = None,
        **kwargs
    ) -> ReviewReport:
        """
        Adversarially review paper for ANY contradictions
        
        Focuses on:
        - Figure-text contradictions
        - Data-text mismatches
        - Internal inconsistencies
        """
        # List figures if available
        figure_descriptions = "N/A"
        if figures_dir:
            from pathlib import Path
            figures = list(Path(figures_dir).glob("*.png"))
            figure_descriptions = "\n".join([f.name for f in figures])
        
        prompt = ADVERSARIAL_REVIEW_PROMPT.format(
            paper_text=paper_draft[:10000],
            experiment_data=str(data.get("experiment_data", {}))[:5000],
            figure_descriptions=figure_descriptions
        )
        
        response = self.llm_call(prompt)
        
        # Parse issues from structured output
        issues = self._parse_adversarial_issues(response)
        
        summary = (
            f"Adversarial review complete. "
            f"Found {len(issues)} potential issues requiring attention."
        )
        
        return self.format_report(issues, summary)
    
    def _parse_adversarial_issues(self, response: str) -> list:
        """Parse adversarial review response"""
        issues = []
        
        # Pattern: ISSUE: ... SEVERITY: ... LOCATION: ... EVIDENCE: ...
        pattern = r'ISSUE:\s*(.+?)\s*SEVERITY:\s*(\w+)\s*LOCATION:\s*(.+?)\s*EVIDENCE:\s*(.+?)(?=ISSUE:|IMPACT:|$)'
        
        for match in re.finditer(pattern, response, re.DOTALL | re.IGNORECASE):
            issues.append(ReviewIssue(
                issue=match.group(1).strip(),
                severity=match.group(2).strip().lower(),
                location=match.group(3).strip(),
                evidence=match.group(4).strip()[:500]  # Truncate evidence
            ))
        
        return issues
