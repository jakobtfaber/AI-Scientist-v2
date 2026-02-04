"""Lightweight data quality validation for experiment stages

This module provides fast, stage-level validation checks to catch
data quality issues early during BFTS tree search, before writeup phase.

Features:
- Data completeness checks
- Statistical validity (sample size, variance)
- Sanity checks (positive values, reasonable ranges)
- Reproducibility checks
- Perplexity-powered reasoning for suspicious results
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"  # Blocks progress
    MAJOR = "major"        # Should fix before continuing
    WARNING = "warning"    # Log but continue
    INFO = "info"         # Informational


@dataclass
class ValidationIssue:
    """Represents a data quality issue"""
    severity: Severity
    check: str
    message: str
    details: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'severity': self.severity.value,
            'check': self.check,
            'message': self.message,
            'details': self.details
        }


class DataQualityChecker:
    """Fast validation checks for experimental data"""
    
    def __init__(
        self,
        min_samples: int = 3,
        max_cv: float = 0.2,
        use_perplexity_validation: bool = False
    ):
        """
        Args:
            min_samples: Minimum samples for statistical validity
            max_cv: Maximum coefficient of variation (std/mean)
            use_perplexity_validation: Use Perplexity for suspicious results
        """
        self.min_samples = min_samples
        self.max_cv = max_cv
        self.use_perplexity = use_perplexity_validation
        self.issues: List[ValidationIssue] = []
    
    def validate(self, experiment_data: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Run all validation checks on experiment data
        
        Args:
            experiment_data: Dict with metrics like:
                {
                    'speedup_factor': [1.5, 1.6, 1.55],
                    'cpu_runtime': [10.0, 10.2, 10.1],
                    'gpu_runtime': [6.5, 6.4, 6.6],
                    'random_seed': 42
                }
        
        Returns:
            List of ValidationIssue objects
        """
        self.issues = []
        
        # Run all checks
        self._check_completeness(experiment_data)
        self._check_statistical_validity(experiment_data)
        self._check_sanity(experiment_data)
        self._check_reproducibility(experiment_data)
        
        # Optional: Perplexity validation for suspicious results
        if self.use_perplexity:
            self._validate_with_perplexity(experiment_data)
        
        return self.issues
    
    def _check_completeness(self, data: dict) -> None:
        """Check for missing required metrics"""
        # Define required metrics based on experiment type
        # For GPU experiments, expect speedup metrics
        if 'gpu_runtime' in data or 'cpu_runtime' in data:
            required = ['speedup_factor', 'cpu_runtime', 'gpu_runtime']
            missing = [k for k in required if k not in data]
            
            if missing:
                self.issues.append(ValidationIssue(
                    severity=Severity.MAJOR,
                    check='completeness',
                    message=f"Missing required metrics: {missing}",
                    details="GPU experiments should report speedup, CPU time, and GPU time"
                ))
    
    def _check_statistical_validity(self, data: dict) -> None:
        """Check sample sizes and variance"""
        for metric_name in ['cpu_runtime', 'gpu_runtime', 'speedup_factor']:
            if metric_name not in data:
                continue
            
            values = data[metric_name]
            
            # Convert to list if single value
            if not isinstance(values, (list, tuple, np.ndarray)):
                values = [values]
            
            # Check sample size
            if len(values) < self.min_samples:
                self.issues.append(ValidationIssue(
                    severity=Severity.MAJOR,
                    check='statistical_validity',
                    message=f"{metric_name}: {len(values)} samples (need >= {self.min_samples})",
                    details="Multiple runs required for statistical validity"
                ))
            
            # Check coefficient of variation (only if multiple samples)
            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if mean_val != 0:
                    cv = std_val / mean_val
                    
                    if cv > self.max_cv:
                        self.issues.append(ValidationIssue(
                            severity=Severity.WARNING,
                            check='statistical_validity',
                            message=f"{metric_name}: High variance (CV={cv:.2f}, max={self.max_cv})",
                            details="High variance may indicate unstable measurements"
                        ))
    
    def _check_sanity(self, data: dict) -> None:
        """Sanity checks for reasonable values"""
        # Check speedup is positive
        if 'speedup_factor' in data:
            speedup = data['speedup_factor']
            
            # Get single value if list
            if isinstance(speedup, (list, tuple, np.ndarray)):
                speedup = np.mean(speedup)
            
            if speedup <= 0:
                self.issues.append(ValidationIssue(
                    severity=Severity.CRITICAL,
                    check='sanity',
                    message=f"Speedup <= 0 ({speedup})",
                    details="Speedup must be positive (GPU_time < CPU_time)"
                ))
            
            # Flag suspiciously high speedups
            elif speedup > 10000:
                self.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    check='sanity',
                    message=f"Suspiciously high speedup: {speedup:.1f}×",
                    details="Verify timing measurements are correct. 10000× is extremely rare."
                ))
            
            elif speedup > 1000:
                self.issues.append(ValidationIssue(
                    severity=Severity.INFO,
                    check='sanity',
                    message=f"Very high speedup: {speedup:.1f}×",
                    details="1000× speedup is excellent but worth double-checking"
                ))
        
        # Check GPU is actually faster than CPU
        if 'cpu_runtime' in data and 'gpu_runtime' in data:
            cpu_time = data['cpu_runtime']
            gpu_time = data['gpu_runtime']
            
            # Get means if lists
            if isinstance(cpu_time, (list, tuple, np.ndarray)):
                cpu_time = np.mean(cpu_time)
            if isinstance(gpu_time, (list, tuple, np.ndarray)):
                gpu_time = np.mean(gpu_time)
            
            if gpu_time >= cpu_time:
                self.issues.append(ValidationIssue(
                    severity=Severity.MAJOR,
                    check='sanity',
                    message=f"GPU ({gpu_time:.4f}s) not faster than CPU ({cpu_time:.4f}s)",
                    details="GPU should be faster than CPU for this to be worthwhile"
                ))
    
    def _check_reproducibility(self, data: dict) -> None:
        """Check for reproducibility indicators"""
        # Check for random seed
        if 'random_seed' not in data and 'seeds' not in data and 'seed' not in data:
            self.issues.append(ValidationIssue(
                severity=Severity.WARNING,
                check='reproducibility',
                message="No random seed recorded",
                details="Results may not be reproducible without seed information"
            ))
    
    def _validate_with_perplexity(self, data: dict) -> None:
        """Use Perplexity reasoning to validate suspicious results"""
        if 'speedup_factor' not in data:
            return
        
        speedup = data['speedup_factor']
        if isinstance(speedup, (list, tuple, np.ndarray)):
            speedup = np.mean(speedup)
        
        # Only validate if speedup is high
        if speedup > 1000:
            try:
                from ai_scientist.research.perplexity_integration import validate_speedup_with_reasoning
                
                is_realistic, reasoning = validate_speedup_with_reasoning(
                    speedup=speedup,
                    algorithm="Fast Folding Algorithm",
                    implementation="PyTorch GPU vs NumPy CPU"
                )
                
                if not is_realistic:
                    self.issues.append(ValidationIssue(
                        severity=Severity.MAJOR,
                        check='perplexity_validation',
                        message=f"Perplexity flagged {speedup:.1f}× as potentially unrealistic",
                        details=reasoning
                    ))
                else:
                    logger.info(f"Perplexity validated {speedup:.1f}× speedup as realistic")
                    
            except Exception as e:
                logger.warning(f"Perplexity validation failed: {e}")
    
    def has_critical_issues(self) -> bool:
        """Check if any critical issues found"""
        return any(i.severity == Severity.CRITICAL for i in self.issues)
    
    def has_major_issues(self) -> bool:
        """Check if any major issues found"""
        return any(i.severity == Severity.MAJOR for i in self.issues)
    
    def get_summary(self) -> str:
        """Get summary of validation results"""
        if not self.issues:
            return "✓ All data quality checks passed"
        
        by_severity = {
            Severity.CRITICAL: 0,
            Severity.MAJOR: 0,
            Severity.WARNING: 0,
            Severity.INFO: 0
        }
        
        for issue in self.issues:
            by_severity[issue.severity] += 1
        
        parts = []
        if by_severity[Severity.CRITICAL]:
            parts.append(f"❌ {by_severity[Severity.CRITICAL]} CRITICAL")
        if by_severity[Severity.MAJOR]:
            parts.append(f"⚠️  {by_severity[Severity.MAJOR]} MAJOR")
        if by_severity[Severity.WARNING]:
            parts.append(f"⚡ {by_severity[Severity.WARNING]} WARNING")
        if by_severity[Severity.INFO]:
            parts.append(f"ℹ️  {by_severity[Severity.INFO]} INFO")
        
        return ", ".join(parts)
