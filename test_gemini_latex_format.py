#!/usr/bin/env python3
"""
Minimal test to see how Gemini formats LaTeX responses
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["GEMINI_API_KEY"] = "AIzaSyAPSRI-HHsFBwk6cwSYd_NSSJ16ROiKYFI"

from ai_scientist.llm import get_response_from_llm, create_client

# Create Gemini client
client, model = create_client("gemini-3-pro-preview")

# Simulate the writeup prompt (simplified)
prompt = """You are an AI research assistant. Your task is to write a complete LaTeX document for a research paper.

Please generate a COMPLETE LaTeX document using the ICLR 2025 template. The document should include:
- Title, abstract, introduction, methods, results, and conclusion
- Must be wrapped in ```latex code blocks

Topic: GPU Acceleration Performance Study

Please generate the complete LaTeX code now."""

system_msg = "You are an expert AI research assistant."

print("Sending prompt to gemini-3-pro-preview...")
print(f"Prompt length: {len(prompt)} chars\n")

response, _ = get_response_from_llm(
    prompt=prompt,
    client=client,
    model=model,
    system_message=system_msg,
    msg_history=[],
    temperature=0.75,
    print_debug=False
)

print(f"\n{'='*80}")
print(f"RESPONSE LENGTH: {len(response)} chars")
print(f"{'='*80}\n")

# Check for latex code blocks
if "```latex" in response:
    print("✓ Response contains ```latex code blocks")
else:
    print("✗ Response does NOT contain ```latex code blocks")
    
if "```" in response:
    print("✓ Response contains generic ``` code blocks")
else:
    print("✗ Response has NO code blocks at all")

# Show first and last parts
print(f"\nFIRST 500 CHARS:\n{'-'*80}\n{response[:500]}\n")
print(f"\nLAST 500 CHARS:\n{'-'*80}\n{response[-500:]}\n")

# Save to file
with open("/data/ai-tools/SakanaAI/AI-Scientist-v2/gemini_latex_test_response.txt", "w") as f:
    f.write(response)
print("Full response saved to: gemini_latex_test_response.txt")
