"""Methods reviewer - checks experimental design"""

from typing import Dict, Any
from ai_scientist.reviewers.base_reviewer import BaseReviewer, ReviewReport, ReviewIssue
from ai_scientist.prompt_templates import METHODS_REVIEW_PROMPT
import json


class MethodsReviewer(BaseReviewer):
    """Review experimental methodology and design"""
    
    def review(
        self,
        paper_draft: str,
        data: Dict[str, Any],
        code_path: str = None,
        **kwargs
    ) -> ReviewReport:
        """
        Review experimental methods
        
        Checks:
        - Timing methodology
        - Confounding variables
        - Sample sizes
        - Reproducibility
        """
        # Extract methods section from paper
        methods_text = self._extract_methods_section(paper_draft)
        
        # Load experiment code if available
        experiment_code = data.get("experiment_code", "")
        if code_path:
            try:
                with open(code_path) as f:
                    experiment_code = f.read()
            except:
                pass
        
        prompt = METHODS_REVIEW_PROMPT.format(
            methods_text=methods_text[:5000],
            experiment_code=experiment_code[:5000]
        )
        
        response = self.llm_call(prompt)
        
        # Parse issues
        issues = self._parse_methods_issues(response)
        
        summary = f"Reviewed experimental methods. Found {len(issues)} methodological concerns."
        
        return self.format_report(issues, summary)
    
    def _extract_methods_section(self, paper: str) -> str:
        """Extract methods/experiments section from paper"""
        import re
        
        # Find section between \section{Experiments} and next \section
        match = re.search(
            r'\\section\{(Experiments?|Methods?|Methodology)\}(.+?)(?=\\section|\Z)',
            paper,
            re.DOTALL | re.IGNORECASE
        )
        
        if match:
            return match.group(2)
        
        # Fallback: return full paper if section not found
        return paper
    
    def _parse_methods_issues(self, response: str) -> list:
        """Parse methods review response"""
        issues = []
        
        # Try JSON parsing
        json_blocks = self.parse_json_response(response)
        
        for block in json_blocks:
            if isinstance(block, dict):
                issues.append(ReviewIssue(
                    issue=block.get("flaw", "Methodological issue"),
                    severity=block.get("severity", "minor"),
                    location=block.get("location", "Unknown"),
                    evidence="",
                    fix=block.get("fix")
                ))
            elif isinstance(block, list):
                for item in block:
                    if isinstance(item, dict):
                        issues.append(ReviewIssue(
                            issue=item.get("flaw", "Methodological issue"),
                            severity=item.get("severity", "minor"),
                            location=item.get("location", "Unknown"),
                            evidence="",
                            fix=item.get("fix")
                        ))
        
        return issues
