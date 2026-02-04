"""
Perplexity Integration for AI Scientist

Provides research and reasoning capabilities using the Perplexity API SDK.

Features:
- research_idea(): Literature review before experiments
- validate_speedup_with_reasoning(): AI validation of suspicious results
- refine_idea_with_research(): Enhance ideas with SOTA knowledge

Setup:
    pip install perplexityai
    export PERPLEXITY_API_KEY="your_api_key_here"
"""

import logging
import os
from typing import Dict, Any, Tuple

logger = logging.getLogger("ai-scientist")


def get_perplexity_client():
    """
    Get initialized Perplexity client.
    
    Loads API key from ~/.perplexity_key file or PERPLEXITY_API_KEY env var.
    
    Returns:
        Perplexity client instance or None if unavailable
    """
    try:
        from perplexity import Perplexity
        from pathlib import Path
        
        # Try loading API key from file first
        api_key = None
        key_file = Path.home() / ".perplexity_key"
        
        if key_file.exists():
            try:
                api_key = key_file.read_text().strip()
                logger.debug(f"Loaded Perplexity API key from {key_file}")
            except Exception as e:
                logger.warning(f"Failed to read API key from {key_file}: {e}")
        
        # Fall back to environment variable
        if not api_key:
            api_key = os.getenv("PERPLEXITY_API_KEY")
            if api_key:
                logger.debug("Loaded Perplexity API key from PERPLEXITY_API_KEY env var")
        
        if not api_key:
            logger.warning("Perplexity API key not found. Set PERPLEXITY_API_KEY or create ~/.perplexity_key")
            return None
        
        return Perplexity(api_key=api_key)
    
    except ImportError:
        logger.warning("perplexityai package not installed. Install with: pip install perplexityai")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Perplexity client: {e}")
        return None


def research_idea(idea: Dict[str, Any]) -> str:
    """
    Research an idea using Perplexity to gather state-of-the-art knowledge.
    
    Args:
        idea: Dictionary containing idea details (Title, Name, Experiment, etc.)
        
    Returns:
        Research summary string, or empty string if unavailable
        
    Example:
        >>> idea = {"Title": "GPU Fast Folding Algorithm", "Name": "gpu_ffa"}
        >>> research = research_idea(idea)
        >>> print(research[:200])
        "Recent literature on GPU-accelerated pulsar search shows..."
    """
    client = get_perplexity_client()
    if not client:
        return ""
    
    try:
        # Build research query
        query = f"""
Research the current state-of-the-art for: {idea['Title']}

Please provide:
1. Recent implementations and libraries (2023-2024)
2. Typical performance benchmarks and speedup ranges
3. Known challenges and limitations
4. Best practices for implementation
5. Relevant recent papers (if any)

Focus on practical implementation details rather than theory.
"""
        
        # Call Perplexity Chat Completions API
        logger.info(f"Researching idea: {idea.get('Name', 'unnamed')}")
        
        completion = client.chat.completions.create(
            model="sonar-pro",  # Web-grounded model for research
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        research_text = completion.choices[0].message.content
        logger.info(f"✅ Perplexity research completed ({len(research_text)} chars)")
        
        # Log citations if available
        if hasattr(completion, 'citations') and completion.citations:
            logger.info(f"📚 Found {len(completion.citations)} citations")
            for i, citation in enumerate(completion.citations[:5], 1):
                logger.debug(f"  [{i}] {citation}")
        
        return research_text
        
    except Exception as e:
        logger.error(f"Perplexity research failed: {e}")
        return ""


def validate_speedup_with_reasoning(
    speedup: float,
    algorithm: str,
    implementation: str
) -> Tuple[bool, str]:
    """
    Use Perplexity reasoning to validate if a measured speedup is realistic.
    
    Args:
        speedup: Measured speedup factor (e.g., 956.4)
        algorithm: Algorithm name (e.g., "Fast Folding Algorithm")
        implementation: Implementation description (e.g., "GPU-accelerated PyTorch")
        
    Returns:
        Tuple of (is_realistic: bool, reasoning: str)
        
    Example:
        >>> is_ok, reason = validate_speedup_with_reasoning(956.4, "FFA", "GPU PyTorch")
        >>> print(f"Realistic: {is_ok}")
        >>> print(reason[:200])
    """
    client = get_perplexity_client()
    if not client:
        return True, "Perplexity not available"
    
    try:
        # Build reasoning query
        query = f"""
We measured a {speedup:.1f}× speedup for GPU vs CPU implementation.

Algorithm: {algorithm}
Implementation: {implementation}

Please analyze if this speedup is realistic by considering:
1. Theoretical limits for this type of algorithm on GPUs
2. Typical speedup ranges reported in literature
3. Possible measurement errors or timing issues
4. Whether this could be due to algorithmic improvements

Start your response with either:
- "REALISTIC:" if the speedup is plausible
- "UNREALISTIC:" if the speedup is highly suspicious

Then provide detailed reasoning.
"""
        
        logger.info(f"🤔 Validating {speedup:.1f}× speedup with Perplexity reasoning...")
        
        # Use sonar model for reasoning (it has web access for verification)
        completion = client.chat.completions.create(
            model="sonar",  # Fast model for reasoning
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        reasoning_text = completion.choices[0].message.content
        
        # Determine if realistic based on response
        is_realistic = "UNREALISTIC:" not in reasoning_text[:50]
        
        verdict = "realistic" if is_realistic else "unrealistic"
        logger.info(f"{'✅' if is_realistic else '❌'} Perplexity validation: {speedup:.1f}× is {verdict}")
        
        return is_realistic, reasoning_text
        
    except Exception as e:
        logger.error(f"Perplexity reasoning failed: {e}")
        return True, f"Validation error: {e}"


def refine_idea_with_research(idea: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance an idea with Perplexity research before experiments.
    
    Args:
        idea: Idea dictionary
        
    Returns:
        Enhanced idea with 'perplexity_research' field added
        
    Example:
        >>> idea = {"Title": "GPU FFA", "Name": "gpu_ffa"}
        >>> enriched = refine_idea_with_research(idea)
        >>> print('perplexity_research' in enriched)
        True
    """
    research = research_idea(idea)
    
    if research:
        idea['perplexity_research'] = research
        logger.info(f"✅ Idea '{idea.get('Name', 'unnamed')}' enriched with {len(research)} chars of research")
    else:
        logger.warning(f"⚠️  No research available for idea '{idea.get('Name', 'unnamed')}'")
    
    return idea


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test 1: Research
    print("\n" + "="*70)
    print("TEST: Perplexity Research")
    print("="*70 + "\n")
    
    test_idea = {
        "Title": "GPU-Accelerated Fast Folding Algorithm for Pulsar Search",
        "Name": "gpu_ffa_pulsar",
        "Experiment": "Implement FFA on GPU using PyTorch"
    }
    
    research_result = research_idea(test_idea)
    if research_result:
        print(f"✅ Research successful ({len(research_result)} chars)")
        print(f"\nFirst 500 chars:\n{research_result[:500]}...\n")
    else:
        print("❌ Research failed (check API key)\n")
    
    # Test 2: Reasoning
    print("\n" + "="*70)
    print("TEST: Perplexity Reasoning")
    print("="*70 + "\n")
    
    is_realistic, reasoning = validate_speedup_with_reasoning(
        speedup=956.4,
        algorithm="Fast Folding Algorithm",
        implementation="GPU-accelerated pulsar search using PyTorch"
    )
    
    if reasoning != "Perplexity not available":
        print(f"Realistic: {is_realistic}")
        print(f"\nReasoning:\n{reasoning[:500]}...\n")
    else:
        print("❌ Reasoning failed (check API key)\n")
    
    # Test 3: Idea refinement
    print("\n" + "="*70)
    print("TEST: Idea Refinement")
    print("="*70 + "\n")
    
    enriched_idea = refine_idea_with_research(test_idea.copy())
    if 'perplexity_research' in enriched_idea:
        print(f"✅ Idea enriched with {len(enriched_idea['perplexity_research'])} chars\n")
    else:
        print("❌ Idea enrichment failed\n")
