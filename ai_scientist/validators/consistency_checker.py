"""Validate paper writeup against experimental data"""

import json
import re
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """Represents a quantitative claim from the paper"""
    text: str
    claim_type: str  # "independence", "speedup", "statistical", "measurement"
    location: str  # Section/line reference
    variables: List[str] = field(default_factory=list)
    metric: Optional[str] = None
    expected_value: Optional[float] = None
    
    def __str__(self):
        return f"Claim({self.claim_type}): '{self.text[:50]}...' at {self.location}"


@dataclass
class ValidationResult:
    """Result of validating a single claim"""
    claim: Claim
    passed: bool
    severity: str  # "none", "minor", "major", "critical"
    reason: str
    data_evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_correction: Optional[str] = None
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} [{self.severity.upper()}] {self.claim.text[:40]}..."


@dataclass
class ValidationReport:
    """Aggregated validation results"""
    results: List[ValidationResult]
    total_claims: int
    passed_claims: int
    failed_claims: int
    critical_issues: List[ValidationResult] = field(default_factory=list)
    major_issues: List[ValidationResult] = field(default_factory=list)
    
    def all_passed(self) -> bool:
        return self.failed_claims == 0
    
    def has_critical_errors(self) -> bool:
        return len(self.critical_issues) > 0
    
    def summary(self) -> str:
        lines = [
            f"Validation Report:",
            f"  Total claims checked: {self.total_claims}",
            f"  Passed: {self.passed_claims}",
            f"  Failed: {self.failed_claims}",
            f"  Critical issues: {len(self.critical_issues)}",
            f"  Major issues: {len(self.major_issues)}"
        ]
        
        if self.critical_issues:
            lines.append("\nCritical Issues:")
            for issue in self.critical_issues:
                lines.append(f"  - {issue.claim.text}")
                lines.append(f"    Reason: {issue.reason}")
        
        return "\n".join(lines)


class ConsistencyChecker:
    """Validate writeup against experimental data"""
    
    def __init__(
        self,
        latex_path: str,
        data: Dict[str, Any],
        figures_dir: Optional[str] = None
    ):
        """
        Initialize consistency checker
        
        Args:
            latex_path: Path to LaTeX file
            data: Experimental data dictionary (from data_loader)
            figures_dir: Optional path to figures directory
        """
        self.latex_path = Path(latex_path)
        self.data = data
        self.figures_dir = Path(figures_dir) if figures_dir else None
        
        # Load LaTeX content
        if self.latex_path.exists():
            with open(self.latex_path) as f:
                self.latex_content = f.read()
        else:
            logger.warning(f"LaTeX file not found: {latex_path}")
            self.latex_content = ""
    
    def extract_claims(self) -> List[Claim]:
        """
        Extract quantitative claims from LaTeX
        
        Returns:
            List of Claims found in the paper
        """
        claims = []
        
        # Pattern 1: Independence claims
        independence_patterns = [
            r"independent of ([a-zA-Z\s]+)",
            r"invariant to ([a-zA-Z\s]+)",
            r"unaffected by ([a-zA-Z\s]+)",
            r"does not depend on ([a-zA-Z\s]+)"
        ]
        
        for pattern in independence_patterns:
            for match in re.finditer(pattern, self.latex_content, re.IGNORECASE):
                claims.append(Claim(
                    text=self._extract_sentence(match.start()),
                    claim_type="independence",
                    location=self._find_location(match.start()),
                    variables=[match.group(1).strip()]
                ))
        
        # Pattern 2: Speedup claims
        speedup_patterns = [
            r"speedup.*?(\d+\.?\d*)×",
            r"(\d+\.?\d*)×\s*faster",
            r"(\d+\.?\d*)×\s*speedup"
        ]
        
        for pattern in speedup_patterns:
            for match in re.finditer(pattern, self.latex_content, re.IGNORECASE):
                try:
                    speedup_value = float(match.group(1))
                    claims.append(Claim(
                        text=self._extract_sentence(match.start()),
                        claim_type="speedup",
                        location=self._find_location(match.start()),
                        expected_value=speedup_value
                    ))
                except (ValueError, IndexError):
                    pass
        
        # Pattern 3: Statistical claims
        stats_patterns = [
            r"p\s*[<>=]\s*0?\.\d+",
            r"significantly",
            r"correlation",
            r"mean.*?(\d+\.?\d*)",
            r"standard deviation.*?(\d+\.?\d*)"
        ]
        
        for pattern in stats_patterns:
            for match in re.finditer(pattern, self.latex_content, re.IGNORECASE):
                claims.append(Claim(
                    text=self._extract_sentence(match.start()),
                    claim_type="statistical",
                    location=self._find_location(match.start())
                ))
        
        logger.info(f"Extracted {len(claims)} claims from paper")
        return claims
    
    def validate_claim(self, claim: Claim) -> ValidationResult:
        """
        Validate a single claim against data
        
        Args:
            claim: Claim to validate
            
        Returns:
            ValidationResult with pass/fail and evidence
        """
        if claim.claim_type == "independence":
            return self.check_independence(claim)
        elif claim.claim_type == "speedup":
            return self.check_speedup(claim)
        elif claim.claim_type == "statistical":
            return self.check_statistical(claim)
        else:
            # Unknown type - pass by default
            return ValidationResult(
                claim=claim,
                passed=True,
                severity="none",
                reason="Unknown claim type, skipped validation"
            )
    
    def check_independence(self, claim: Claim) -> ValidationResult:
        """
        Check independence claim by computing coefficient of variation
        
        Uses threshold: CV < 0.2 for independence
        """
        variable = claim.variables[0] if claim.variables else None
        
        if not variable:
            return ValidationResult(
                claim=claim,
                passed=False,
                severity="minor",
                reason="Could not extract variable from claim"
            )
        
        # Extract relevant data
        values = self._find_values_for_variable(variable)
        
        if not values or len(values) < 2:
            return ValidationResult(
                claim=claim,
                passed=False,
                severity="minor",
                reason=f"Insufficient data for variable '{variable}'",
                data_evidence={"values": values}
            )
        
        # Compute coefficient of variation
        values_array = np.array(list(values.values()))
        mean = np.mean(values_array)
        std = np.std(values_array)
        cv = std / mean if mean != 0 else float('inf')
        
        # Threshold for independence: CV < 0.2
        passed = cv < 0.2
        
        severity = "none" if passed else ("critical" if cv > 0.5 else "major")
        
        reason = (
            f"CV = {cv:.3f} {'< 0.2 (independent)' if passed else '>= 0.2 (dependent)'}"
        )
        
        if not passed:
            # Suggest correction
            values_str = ", ".join([f"{k}={v:.2f}" for k, v in values.items()])
            suggested = (
                f"The {claim.metric or 'metric'} shows dependence on {variable} "
                f"with CV={cv:.2f} ({values_str})"
            )
        else:
            suggested = None
        
        return ValidationResult(
            claim=claim,
            passed=passed,
            severity=severity,
            reason=reason,
            data_evidence={"values": values, "cv": cv, "mean": mean, "std": std},
            suggested_correction=suggested
        )
    
    def check_speedup(self, claim: Claim) -> ValidationResult:
        """Check speedup claim against actual measured speedup"""
        
        claimed_speedup = claim.expected_value
        
        if not claimed_speedup:
            return ValidationResult(
                claim=claim,
                passed=False,
                severity="minor",
                reason="Could not extract speedup value from claim"
            )
        
        # Find actual speedup in data
        actual_speedup = self._find_speedup_in_data()
        
        if actual_speedup is None:
            return ValidationResult(
                claim=claim,
                passed=False,
                severity="minor",
                reason="Could not find speedup metric in experimental data",
                data_evidence={}
            )
        
        # Allow 10% tolerance
        tolerance = 0.1
        relative_error = abs(actual_speedup - claimed_speedup) / actual_speedup
        passed = relative_error < tolerance
        
        severity = "none" if passed else ("major" if relative_error > 0.5 else "minor")
        
        reason = (
            f"Claimed {claimed_speedup}× vs actual {actual_speedup:.1f}× "
            f"(error: {relative_error*100:.1f}%)"
        )
        
        return ValidationResult(
            claim=claim,
            passed=passed,
            severity=severity,
            reason=reason,
            data_evidence={"claimed": claimed_speedup, "actual": actual_speedup},
            suggested_correction=f"speedup of {actual_speedup:.1f}×" if not passed else None
        )
    
    def check_statistical(self, claim: Claim) -> ValidationResult:
        """Check statistical claims (placeholder for now)"""
        # TODO: Implement statistical validation
        return ValidationResult(
            claim=claim,
            passed=True,
            severity="none",
            reason="Statistical validation not yet implemented"
        )
    
    def run_all_checks(self) -> ValidationReport:
        """
        Run all validators and compile report
        
        Returns:
            ValidationReport with all results
        """
        claims = self.extract_claims()
        results = []
        
        for claim in claims:
            result = self.validate_claim(claim)
            results.append(result)
            
            # Log failures
            if not result.passed:
                logger.warning(f"Validation failed: {result}")
        
        # Aggregate results
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        critical = [r for r in results if r.severity == "critical"]
        major = [r for r in results if r.severity == "major"]
        
        report = ValidationReport(
            results=results,
            total_claims=len(claims),
            passed_claims=len(passed),
            failed_claims=len(failed),
            critical_issues=critical,
            major_issues=major
        )
        
        logger.info(f"Validation complete: {report.summary()}")
        
        return report
    
    def auto_correct_claims(
        self,
        latex_content: str,
        failed_validations: List[ValidationResult],
        max_fixes: int = 5
    ) -> str:
        """
        Automatically rewrite failed claims in LaTeX
        
        Args:
            latex_content: Original LaTeX content
            failed_validations: List of failed validation results
            max_fixes: Maximum number of claims to auto-fix
            
        Returns:
            Updated LaTeX content with corrections
        """
        from ai_scientist.prompt_templates import REWRITE_CLAIM_PROMPT
        from ai_scientist.llm import get_response_from_llm, create_client
        
        corrected_latex = latex_content
        fixes_applied = 0
        
        # Sort by severity (critical first)
        severity_order = {"critical": 0, "major": 1, "minor": 2}
        sorted_validations = sorted(
            failed_validations,
            key=lambda v: severity_order.get(v.severity, 3)
        )
        
        for validation in sorted_validations[:max_fixes]:
            if validation.severity in ["critical", "major"]:
                logger.info(f"Auto-fixing {validation.severity} claim: {validation.claim.text[:50]}...")
                
                try:
                    # Prepare data evidence as string
                    evidence_str = json.dumps(validation.data_evidence, indent=2)
                    
                    # Generate rewrite prompt
                    prompt = REWRITE_CLAIM_PROMPT.format(
                        claim=validation.claim.text,
                        data=evidence_str,
                        reason=validation.reason
                    )
                    
                    # Call LLM to rewrite claim
                    client, model = create_client("gemini-2.5-flash")
                    rewritten, _ = get_response_from_llm(
                        prompt=prompt,
                        client=client,
                        model=model,
                        system_message="You are a scientific writing assistant. Rewrite claims to accurately reflect experimental data.",
                        print_debug=False
                    )
                    
                    # Extract rewritten claim (remove markdown markers if present)
                    rewritten_clean = rewritten.strip().strip('`"\'')
                    
                    # Replace in LaTeX
                    if validation.claim.text in corrected_latex:
                        corrected_latex = corrected_latex.replace(
                            validation.claim.text,
                            rewritten_clean,
                            1  # Only replace first occurrence
                        )
                        fixes_applied += 1
                        logger.info(f"✓ Fixed claim: '{rewritten_clean[:60]}...'")
                    else:
                        logger.warning(f"Could not find exact claim text in LaTeX for replacement")
                        
                except Exception as e:
                    logger.error(f"Failed to auto-fix claim: {e}")
                    continue
        
        logger.info(f"Applied {fixes_applied} automated claim fixes")
        return corrected_latex
    
    def run_all_checks_with_autofix(
        self,
        latex_path: Optional[str] = None,
        apply_fixes: bool = True
    ) -> tuple[ValidationReport, Optional[str]]:
        """
        Run validation and optionally apply auto-fixes
        
        Args:
            latex_path: Optional path to save corrected LaTeX
            apply_fixes: Whether to apply automatic fixes
            
        Returns:
            Tuple of (ValidationReport, corrected_latex_content or None)
        """
        # Run normal validation
        report = self.run_all_checks()
        
        if not apply_fixes or not report.failed_claims:
            return report, None
        
        # Apply auto-fixes
        failed = [r for r in report.results if not r.passed]
        corrected_latex = self.auto_correct_claims(
            latex_content=self.latex_content,
            failed_validations=failed
        )
        
        # Optionally save corrected version
        if latex_path:
            with open(latex_path, 'w') as f:
                f.write(corrected_latex)
            logger.info(f"Saved corrected LaTeX to {latex_path}")
        
        return report, corrected_latex
    
    # Helper methods
    
    def _extract_sentence(self, position: int) -> str:
        """Extract full sentence containing position"""
        # Find sentence boundaries
        start = self.latex_content.rfind('.', 0, position) + 1
        end = self.latex_content.find('.', position) + 1
        
        sentence = self.latex_content[start:end].strip()
        return sentence[:200]  # Truncate very long sentences
    
    def _find_location(self, position: int) -> str:
        """Find section/line reference for position"""
        # Count \\section commands before position
        sections = re.findall(r'\\section\{([^}]+)\}', self.latex_content[:position])
        
        if sections:
            return f"Section: {sections[-1]}"
        
        # Fall back to line number
        line_num = self.latex_content[:position].count('\n') + 1
        return f"Line {line_num}"
    
    def _find_values_for_variable(self, variable: str) -> Dict[str, float]:
        """Find data values for a given variable"""
        variable_lower = variable.lower()
        values = {}
        
        def search_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if variable_lower in key.lower():
                        if isinstance(value, dict):
                            # Extract runtime/speedup metrics
                            for metric in ["gpu_runtime", "cpu_runtime", "speedup_factor", "runtime"]:
                                if metric in value:
                                    values[key] = float(value[metric])
                        elif isinstance(value, (int, float)):
                            values[key] = float(value)
                    
                    # Recurse
                    search_recursive(value, f"{prefix}.{key}" if prefix else key)
        
        search_recursive(self.data.get("experiment_data", {}))
        
        return values
    
    def _find_speedup_in_data(self) -> Optional[float]:
        """Find speedup factor in experimental data"""
        
        def search_recursive(obj):
            if isinstance(obj, dict):
                # Direct speedup key
                if "speedup_factor" in obj:
                    val = obj["speedup_factor"]
                    if isinstance(val, (int, float)):
                        return float(val)
                    elif isinstance(val, list):
                        return float(np.mean(val))
                
                # Compute from cpu/gpu runtime
                if "cpu_runtime" in obj and "gpu_runtime" in obj:
                    cpu = float(obj["cpu_runtime"]) if isinstance(obj["cpu_runtime"], (int, float)) else float(np.mean(obj["cpu_runtime"]))
                    gpu = float(obj["gpu_runtime"]) if isinstance(obj["gpu_runtime"], (int, float)) else float(np.mean(obj["gpu_runtime"]))
                    if gpu > 0:
                        return cpu / gpu
                
                # Recurse
                for value in obj.values():
                    result = search_recursive(value)
                    if result is not None:
                        return result
            
            return None
        
        return search_recursive(self.data.get("experiment_data", {}))


class ConsistencyError(Exception):
    """Raised when critical consistency issues are found"""
    pass
