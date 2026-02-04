"""Perplexity integration for idea refinement and result validation

Provides helper functions to use Perplexity MCP server for:
1. Research to enhance ideas before experiments
2. Reasoning to validate suspicious experimental results
"""

import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def research_idea(idea: Dict[str, Any]) -> str:
    """
    Use Perplexity research to enhance an idea with latest literature
    
    Args:
        idea: Idea dict with Title, Hypothesis, etc.
    
    Returns:
        Research findings as markdown string
    """
    try:
        # Import MCP tools (they're available in the environment)
        from mcp_perplexity_perplexity_research import perplexity_research
        
        query = f"""
Research the current state-of-the-art for: {idea['Title']}

Hypothesis: {idea.get('Short Hypothesis', idea.get('Abstract', ''))}

Please find:
1. Recent papers (2023-2026) on this topic
2. Existing implementations and their performance
3. Common approaches and optimizations
4. Known challenges and limitations
5. Realistic performance expectations

Focus on concrete, quantitative information with citations.
"""
        
        result = perplexity_research(messages=[
            {"role": "user", "content": query}
        ])
        
        logger.info(f"Perplexity research completed for idea: {idea['Name']}")
        return result
        
    except ImportError:
        logger.warning("Perplexity MCP server not available")
        return ""
    except Exception as e:
        logger.error(f"Perplexity research failed: {e}")
        return ""


def validate_speedup_with_reasoning(
    speedup: float,
    algorithm: str,
    implementation: str
) -> Tuple[bool, str]:
    """
    Use Perplexity reasoning to validate if a speedup is realistic
    
    Args:
        speedup: Measured speedup factor (e.g., 956)
        algorithm: Algorithm name (e.g., "Fast Folding Algorithm")
        implementation: Details (e.g., "PyTorch GPU vs NumPy CPU")
    
    Returns:
        (is_realistic, reasoning_text)
    """
    try:
        from mcp_perplexity_perplexity_reason import perplexity_reason
        
        query = f"""
We measured a {speedup:.1f}× speedup for GPU vs CPU implementation.

Context:
- Algorithm: {algorithm}
- Implementation: {implementation}
- Speedup: {speedup:.1f}×

Is this speedup realistic? Consider:
1. Theoretical speedup limits for this type of algorithm
2. Memory bandwidth vs compute constraints
3. Typical GPU vs CPU speedups for similar workloads
4. Potential measurement errors or bugs

Provide clear YES/NO answer with reasoning and citations.
Format your answer starting with either "REALISTIC:" or "UNREALISTIC:"
"""
        
        result = perplexity_reason(messages=[
            {"role": "user", "content": query}
        ])
        
        # Parse response
        is_realistic = not ("UNREALISTIC:" in result or "unrealistic" in result.lower()[:200])
        
        logger.info(f"Perplexity validation: {speedup:.1f}× is {'realistic' if is_realistic else 'unrealistic'}")
        return is_realistic, result
        
    except ImportError:
        logger.warning("Perplexity MCP server not available")
        return True, "Perplexity not available"
    except Exception as e:
        logger.error(f"Perplexity reasoning failed: {e}")
        return True, f"Validation error: {e}"


def refine_idea_with_research(idea: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance an idea with Perplexity research before running experiments
    
    Args:
        idea: Original idea dict
    
    Returns:
        Enhanced idea with research context
    """
    research = research_idea(idea)
    
    if research:
        idea['perplexity_research'] = research
        idea['research_enhanced'] = True
        logger.info(f"Enhanced idea '{idea['Name']}' with Perplexity research")
    
    return idea
