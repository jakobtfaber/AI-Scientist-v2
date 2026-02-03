#!/usr/bin/env python3
"""Test if gemini-3-pro-preview can handle a large prompt via OpenAI compatibility layer"""
import os
os.environ["GEMINI_API_KEY"] = "AIzaSyAPSRI-HHsFBwk6cwSYd_NSSJ16ROiKYFI"

import openai

print("Testing gemini-3-pro-preview with a moderately large prompt...")

client = openai.OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Create a moderately large prompt (not full writeup, but substantial)
large_prompt = """
You are writing a technical paper about GPU acceleration. 

Here is experiment data:
""" + ("- Experiment detail line\n" * 200) + """

Based on this data, write a 2-paragraph abstract for a technical paper.
Return the abstract in plain text.
"""

try:
    print(f"Prompt size: {len(large_prompt)} characters")
    print("Sending request to gemini-3-pro-preview...")
    
    response = client.chat.completions.create(
        model="gemini-3-pro-preview",
        messages=[
            {"role": "system", "content": "You are a helpful technical writing assistant."},
            {"role": "user", "content": large_prompt}
        ],
        max_tokens=500,
        temperature=0.7,
    )
    
    content = response.choices[0].message.content
    print(f"\n✓ SUCCESS! Received {len(content)} characters")
    print(f"\nResponse preview:\n{content[:300]}...")
    
except Exception as e:
    print(f"\n✗ FAILED: {type(e).__name__}: {e}")
