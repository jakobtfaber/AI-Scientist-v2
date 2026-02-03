"""Statistics reviewer - validates statistical claims"""

from typing import Dict, Any
from ai_scientist.reviewers.base_reviewer import BaseReviewer, ReviewReport, ReviewIssue
from ai_scientist.prompt_templates import STATISTICS_REVIEW_PROMPT
import json


class StatisticsReviewer(BaseReviewer):
    """Validate statistical claims and methodology"""
    
    def review(
        self,
        paper_draft: str,
        data: Dict[str, Any],
        **kwargs
    ) -> ReviewReport:
        """
        Review paper for statistical rigor
        
        Checks:
        - Independence claims (CV < 0.2)
        - Significance tests
        - Error bars
- Statistical validity
        """
        prompt = STATISTICS_REVIEW_PROMPT.format(
            paper_text=paper_draft[:10000],  # Truncate if too long
            experiment_data=json.dumps(data.get("experiment_data", {}), indent=2)[:5000]
        )
        
        response = self.llm_call(prompt)
        
        # Parse issues from response
        issues = self._parse_issues_from_response(response)
        
        summary = f"Reviewed statistical claims. Found {len(issues)} issues."
        
        return self.format_report(issues, summary)
    
    def _parse_issues_from_response(self, response: str) -> list:
        """Parse LLM response into ReviewIssue objects"""
        issues = []
        
        # Try JSON parsing
        json_blocks = self.parse_json_response(response)
        
        for block in json_blocks:
            if isinstance(block, dict):
                issues.append(ReviewIssue(
                    issue=block.get("issue", "Unknown issue"),
                    severity=block.get("severity", "minor"),
                    location=block.get("claim", "Unknown location"),
                    evidence=str(block.get("data_check", {})),
                    fix=block.get("correct_approach")
                ))
            elif isinstance(block, list):
                for item in block:
                    if isinstance(item, dict):
                        issues.append(ReviewIssue(
                            issue=item.get("issue", "Unknown issue"),
                            severity=item.get("severity", "minor"),
                            location=item.get("claim", "Unknown location"),
                            evidence=str(item.get("data_check", {})),
                            fix=item.get("correct_approach")
                        ))
        
        return issues
