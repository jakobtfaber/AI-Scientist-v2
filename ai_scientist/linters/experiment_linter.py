"""Static analysis for experiment code - detect common pitfalls"""

import ast
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class LintIssue:
    """Single linting issue"""
    severity: str  # "error", "warning", "info"
    message: str
    line: int
    suggestion: str = ""
    
    def __str__(self):
        return f"[{self.severity.upper()}] Line {self.line}: {self.message}"


@dataclass
class LintReport:
    """Aggregated linting results"""
    issues: List[LintIssue]
    file_path: str
    
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)
    
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)
    
    def summary(self) -> str:
        errors = sum(1 for i in self.issues if i.severity == "error")
        warnings = sum(1 for i in self.issues if i.severity == "warning")
        info = sum(1 for i in self.issues if i.severity == "info")
        
        lines = [
            f"Lint Report for {self.file_path}",
            f"  Errors: {errors}",
            f"  Warnings: {warnings}",
            f"  Info: {info}"
        ]
        
        if self.issues:
            lines.append("\nIssues:")
            for issue in self.issues:
                lines.append(f"  {issue}")
                if issue.suggestion:
                    lines.append(f"    → {issue.suggestion}")
        
        return "\n".join(lines)


class ExperimentLinter:
    """Lint experiment code for common pitfalls"""
    
    def __init__(self):
        """Initialize linter with default rules"""
        self.rules = [
            TimingBlockRule(),
            TensorConversionRule(),
            RandomSeedRule()
        ]
    
    def lint(self, code_path: str) -> LintReport:
        """
        Run all linting rules
        
        Args:
            code_path: Path to Python experiment code
            
        Returns:
            LintReport with all issues
        """
        code_path = Path(code_path)
        
        if not code_path.exists():
            return LintReport(
                issues=[LintIssue("error", f"File not found: {code_path}", 0)],
                file_path=str(code_path)
            )
        
        try:
            with open(code_path) as f:
                code = f.read()
            
            tree = ast.parse(code)
            
        except SyntaxError as e:
            return LintReport(
                issues=[LintIssue("error", f"Syntax error: {e}", e.lineno or 0)],
                file_path=str(code_path)
            )
        
        # Run all rules
        issues = []
        for rule in self.rules:
            try:
                rule_issues = rule.check(tree, code)
                issues.extend(rule_issues)
            except Exception as e:
                logger.error(f"Rule {rule.__class__.__name__} failed: {e}")
        
        return LintReport(issues=issues, file_path=str(code_path))


class LintRule:
    """Base class for linting rules"""
    
    def check(self, tree: ast.AST, code: str) -> List[LintIssue]:
        """
        Check for issues in AST
        
        Args:
            tree: Parsed AST
            code: Original source code
            
        Returns:
            List of issues found
        """
        raise NotImplementedError


class TimingBlockRule(LintRule):
    """Detect data generation inside timing blocks"""
    
    def check(self, tree: ast.AST, code: str) -> List[LintIssue]:
        issues = []
        
        # Find timing blocks: pattern is time.time()
        timing_blocks = self._find_timing_blocks(tree)
        
        for start_node, end_node in timing_blocks:
            # Check if timing block contains data generation
            if self._contains_data_generation(tree, start_node, end_node):
                issues.append(LintIssue(
                    severity="warning",
                    message="Data generation detected inside timing block",
                    line=start_node.lineno,
                    suggestion="Move data generation (generate_*, np.random.*, torch.rand*) before timing starts"
                ))
        
        return issues
    
    def _find_timing_blocks(self, tree: ast.AST) -> List:
        """Find start/end pairs of timing blocks"""
        timing_pairs = []
        
        # Simple heuristic: look for time.time() assignments
        time_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    if self._is_time_call(node.value):
                        time_calls.append(node)
        
        # Pair them up (assumes start/end pattern)
        for i in range(0, len(time_calls) - 1, 2):
            if i + 1 < len(time_calls):
                timing_pairs.append((time_calls[i], time_calls[i + 1]))
        
        return timing_pairs
    
    def _is_time_call(self, node: ast.Call) -> bool:
        """Check if node is time.time() call"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "time":
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id == "time"
        return False
    
    def _contains_data_generation(self, tree, start_node, end_node) -> bool:
        """Check if code between start and end contains data generation"""
        # This is a simplified heuristic
        # In practice, would need more sophisticated control flow analysis
        
        start_line = start_node.lineno
        end_line = end_node.lineno
        
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                if start_line < node.lineno < end_line:
                    if isinstance(node, ast.Call):
                        # Check for data generation functions
                        func_name = self._get_func_name(node)
                        if func_name:
                            gen_keywords = ["generate", "random", "rand", "randn", "normal", "uniform"]
                            if any(kw in func_name.lower() for kw in gen_keywords):
                                return True
        
        return False
    
    def _get_func_name(self, call_node: ast.Call) -> str:
        """Extract function name from Call node"""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return ""


class TensorConversionRule(LintRule):
    """Detect repeated torch.tensor() conversions in loops"""
    
    def check(self, tree: ast.AST, code: str) -> List[LintIssue]:
        issues = []
        
        # Find for loops
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if loop body contains torch.tensor
                if self._contains_tensor_conversion(node):
                    issues.append(LintIssue(
                        severity="warning",
                        message="torch.tensor() conversion inside loop - inefficient",
                        line=node.lineno,
                        suggestion="Move tensor conversion outside the loop to avoid repeated overhead"
                    ))
        
        return issues
    
    def _contains_tensor_conversion(self, loop_node: ast.For) -> bool:
        """Check if loop contains torch.tensor() calls"""
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Call):
                # Check for torch.tensor(...)
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "tensor":
                        if isinstance(node.func.value, ast.Name):
                            if node.func.value.id == "torch":
                                return True
        return False


class RandomSeedRule(LintRule):
    """Check for missing random seeds"""
    
    def check(self, tree: ast.AST, code: str) -> List[LintIssue]:
        issues = []
        
        # Check if code uses random but doesn't set seed
        uses_random = False
        sets_seed = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_func_name(node)
                
                # Check for random usage
                if "random" in func_name.lower() or "rand" in func_name.lower():
                    uses_random = True
                
                # Check for seed setting
                if "seed" in func_name.lower():
                    sets_seed = True
        
        if uses_random and not sets_seed:
            issues.append(LintIssue(
                severity="info",
                message="Random number generation detected but no seed set",
                line=1,
                suggestion="Add np.random.seed() or torch.manual_seed() for reproducibility"
            ))
        
        return issues
    
    def _get_func_name(self, call_node: ast.Call) -> str:
        """Extract function name from Call node"""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return ""


class LintError(Exception):
    """Raised when linting finds critical errors"""
    pass
