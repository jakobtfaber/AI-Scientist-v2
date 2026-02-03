"""Base class for specialized reviewers"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReviewIssue:
    """Single issue found during review"""
    issue: str
    severity: str  # "critical", "major", "minor"
    location: str
    evidence: str
    fix: Optional[str] = None
    
    def __str__(self):
        return f"[{self.severity.upper()}] {self.issue} @ {self.location}"


@dataclass
class ReviewReport:
    """Report from a single reviewer"""
    reviewer_name: str
    issues: List[ReviewIssue]
    summary: str
    recommendation: str  # "accept", "minor_revision", "major_revision", "reject"
    
    def has_critical_issues(self) -> bool:
        return any(i. severity == "critical" for i in self.issues)
    
    def has_major_issues(self) -> bool:
        return any(i.severity in ["critical", "major"] for i in self.issues)
    
    def issue_summary(self) -> str:
        critical = sum(1 for i in self.issues if i.severity == "critical")
        major = sum(1 for i in self.issues if i.severity == "major")
        minor = sum(1 for i in self.issues if i.severity == "minor")
        
        return f"{critical} critical, {major} major, {minor} minor issues"
    
    def __str__(self):
        lines = [
            f"=== {self.reviewer_name} Review ===",
            f"Recommendation: {self.recommendation.upper()}",
            f"Issues: {self.issue_summary()}",
            f"\nSummary: {self.summary}"
        ]
        
        if self.issues:
            lines.append("\nDetailed Issues:")
            for issue in self.issues:
                lines.append(f"  {issue}")
        
        return "\n".join(lines)


class BaseReviewer(ABC):
    """Base class for specialized reviewers"""
    
    def __init__(self, model: str = "gemini-3-pro-preview"):
        """
        Initialize reviewer
        
        Args:
            model: LLM model to use for reviews
        """
        self.model = model
        self.name = self.__class__.__name__.replace("Reviewer", "")
    
    @abstractmethod
    def review(
        self,
        paper_draft: str,
        data: Dict[str, Any],
        **kwargs
    ) -> ReviewReport:
        """
        Generate review of paper
        
        Args:
            paper_draft: LaTeX content of paper
            data: Experimental data dictionary
            **kwargs: Additional reviewer-specific arguments
            
        Returns:
            ReviewReport with findings
        """
        pass
    
    def llm_call(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Call LLM for review analysis
        
        Args:
            prompt: Review prompt
            system_message: Optional system message
            
        Returns:
            LLM response
        """
        from ai_scientist.llm import get_response_from_llm, create_client
        
        client, model = create_client(self.model)
        
        response, _ = get_response_from_llm(
            prompt=prompt,
            client=client,
            model=model,
            system_message=system_message or "You are a rigorous scientific peer reviewer.",
            print_debug=False
        )
        
        return response
    
    def parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse JSON-formatted LLM response
        
        Args:
            response: LLM response potentially containing JSON
            
        Returns:
            Parsed JSON objects
        """
        # Try to extract JSON blocks
        json_blocks = []
        
        # Pattern 1: ```json ... ```
        import re
        for match in re.finditer(r'```json\s*(\[.*?\]|{.*?})\s*```', response, re.DOTALL):
            try:
                json_blocks.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass
        
        # Pattern 2: Raw JSON array/object
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                json_blocks.extend(parsed)
            else:
                json_blocks.append(parsed)
        except json.JSONDecodeError:
            pass
        
        return json_blocks
    
    def format_report(
        self,
        issues: List[ReviewIssue],
        summary: str
    ) -> ReviewReport:
        """
        Create standardized review report
        
        Args:
            issues: List of issues found
            summary: Overall summary text
            
        Returns:
            Formatted ReviewReport
        """
        # Determine recommendation based on issues
        critical_count = sum(1 for i in issues if i.severity == "critical")
        major_count = sum(1 for i in issues if i.severity == "major")
        
        if critical_count > 0:
            recommendation = "reject"
        elif major_count > 3:
            recommendation = "major_revision"
        elif major_count > 0:
            recommendation = "minor_revision"
        else:
            recommendation = "accept"
        
        return ReviewReport(
            reviewer_name=self.name,
            issues=issues,
            summary=summary,
            recommendation=recommendation
        )
