#!/usr/bin/env python3
"""
Test script for Perplexity Research and Reasoning integration

Tests both the research and reasoning functions from perplexity_integration.py
"""

import json
import sys


def test_research():
    """Test Perplexity research on GPU FFA idea"""
    print("\n" + "="*70)
    print("TEST 1: Perplexity Research Integration")
    print("="*70 + "\n")
    
    try:
        from ai_scientist.research.perplexity_integration import research_idea
        
        # Load the GPU FFA idea
        with open("ai_scientist/ideas/gpu_ffa.json", "r") as f:
            ideas = json.load(f)
        
        gpu_ffa_idea = ideas[0]
        print(f"📋 Idea: {gpu_ffa_idea['Title']}")
        print(f"📝 Name: {gpu_ffa_idea['Name']}\n")
        
        # Run research
        print("🔬 Running Perplexity research...")
        research_results = research_idea(gpu_ffa_idea)
        
        if research_results:
            print("\n✅ Research completed successfully!")
            print(f"📊 Research length: {len(research_results)} characters")
            print(f"\nFirst 500 chars:\n{'-'*70}")
            print(research_results[:500] + "...")
            print("-"*70)
            return True
        else:
            print("❌ Research returned empty results")
            print("⚠️  Perplexity MCP server may not be available")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("⚠️  perplexity_integration module not found")
        return False
    except Exception as e:
        print(f"❌ Research failed: {e}")
        return False


def test_reasoning():
    """Test Perplexity reasoning on suspicious speedup"""
    print("\n" + "="*70)
    print("TEST 2: Perplexity Reasoning Integration")
    print("="*70 + "\n")
    
    try:
        from ai_scientist.research.perplexity_integration import validate_speedup_with_reasoning
        
        # Test with the 956× speedup from our experiment
        speedup = 956.4
        algorithm = "Fast Folding Algorithm (FFA)"
        implementation = "GPU-accelerated pulsar search using PyTorch"
        
        print(f"🔢 Testing speedup: {speedup:.1f}×")
        print(f"🧮 Algorithm: {algorithm}")
        print(f"💻 Implementation: {implementation}\n")
        
        print("🤔 Running Perplexity reasoning validation...")
        is_realistic, reasoning = validate_speedup_with_reasoning(
            speedup, algorithm, implementation
        )
        
        if reasoning and reasoning != "Perplexity not available":
            print(f"\n✅ Validation completed!")
            print(f"📊 Is realistic: {is_realistic}")
            print(f"\nReasoning ({len(reasoning)} chars):\n{'-'*70}")
            print(reasoning[:500] + "...")
            print("-"*70)
            return True
        else:
            print("❌ Reasoning validation unavailable")
            print("⚠️  Perplexity MCP server may not be available")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Reasoning failed: {e}")
        return False


def test_data_quality_with_reasoning():
    """Test DataQualityChecker with Perplexity reasoning enabled"""
    print("\n" + "="*70)
    print("TEST 3: DataQualityChecker with Perplexity Reasoning")
    print("="*70 + "\n")
    
    try:
        from ai_scientist.validators.data_quality_checker import DataQualityChecker
        
        # Simulated experiment data with suspicious speedup
        exp_data = {
            'speedup_factor': [956.4, 950.2, 962.1],
            'cpu_runtime': [10.5, 10.3, 10.6],
            'gpu_runtime': [0.011, 0.0108, 0.0112],
            'random_seed': 42
        }
        
        print("📊 Test data:")
        print(f"  Speedup: {exp_data['speedup_factor']}")
        print(f"  CPU runtime: {exp_data['cpu_runtime']}")
        print(f"  GPU runtime: {exp_data['gpu_runtime']}\n")
        
        # Run validation WITH Perplexity
        print("🔬 Running DataQualityChecker with Perplexity enabled...")
        checker = DataQualityChecker(
            min_samples=3,
            max_cv=0.2,
            use_perplexity_validation=True  # Enable!
        )
        
        issues = checker.validate(exp_data)
        
        print(f"\n✅ Validation completed!")
        print(f"📊 {checker.get_summary()}\n")
        
        if issues:
            for issue in issues:
                severity_emoji = {
                    'critical': '🔴',
                    'major': '🟡',
                    'warning': '⚡',
                    'info': '💡'
                }
                emoji = severity_emoji.get(issue.severity.value, '❓')
                print(f"{emoji} [{issue.severity.value.upper()}] {issue.check}: {issue.message}")
        else:
            print("✅ All checks passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 PERPLEXITY INTEGRATION TEST SUITE")
    print("="*70)
    
    results = {
        'research': test_research(),
        'reasoning': test_reasoning(),
        'data_quality': test_data_quality_with_reasoning()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED (Perplexity MCP server may not be available)")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
