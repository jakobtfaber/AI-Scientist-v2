#!/usr/bin/env python3
"""Debug Gemini response to see what's actually being returned"""
import os
os.environ["GEMINI_API_KEY"] = "AIzaSyAPSRI-HHsFBwk6cwSYd_NSSJ16ROiKYFI"

import openai
import json

print("Testing gemini-3-pro-preview with detailed response inspection...")

client = openai.OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=300.0,
    max_retries=2,
)

# Try a simple prompt first
simple_prompt = "Write a 2-sentence abstract about GPU acceleration."

try:
    print(f"\n1. Testing SIMPLE prompt ({len(simple_prompt)} chars)...")
    response = client.chat.completions.create(
        model="gemini-3-pro-preview",
        messages=[
            {" role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": simple_prompt}
        ],
        max_tokens=8192,
        temperature=0.7,
    )
    
    print(f"Response type: {type(response)}")
    print(f"Response object: {response}")
    print(f"Choices: {response.choices}")
    print(f"Choices[0]: {response.choices[0]}")
    print(f"Message: {response.choices[0].message}")
    print(f"Content: {response.choices[0].message.content}")
    print(f"Content type: {type(response.choices[0].message.content)}")
    
    if response.choices[0].message.content:
        print(f"\n✓ SIMPLE SUCCESS: {response.choices[0].message.content[:100]}")
    else:
        print(f"\n✗ SIMPLE FAILED: Content is None or empty")
        
except Exception as e:
    print(f"\n✗ SIMPLE FAILED: {type(e).__name__}: {e}")

# Now try a larger prompt
large_prompt = "Write a detailed abstract about GPU acceleration.\n\nContext:\n" + ("- Detail line\n" * 200)

try:
    print(f"\n2. Testing LARGE prompt ({len(large_prompt)} chars)...")
    response = client.chat.completions.create(
        model="gemini-3-pro-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": large_prompt}
        ],
        max_tokens=8192,
        temperature=0.7,
    )
    
    print(f"Response type: {type(response)}")
    print(f"Choices length: {len(response.choices) if response.choices else 'None'}")
    if response.choices and len(response.choices) > 0:
        print(f"Message: {response.choices[0].message}")
        print(f"Content type: {type(response.choices[0].message.content)}")
        print(f"Content value: {repr(response.choices[0].message.content)}")
        
        if response.choices[0].message.content:
            print(f"\n✓ LARGE SUCCESS: {response.choices[0].message.content[:100]}")
        else:
            print(f"\n✗ LARGE FAILED: Content is None or empty!")
    else:
        print("✗ No choices in response!")
        
except Exception as e:
    print(f"\n✗ LARGE FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
