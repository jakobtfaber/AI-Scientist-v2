#!/usr/bin/env python3
"""Quick test of Gemini API to diagnose issues"""
import os
os.environ["GEMINI_API_KEY"] = "AIzaSyAPSRI-HHsFBwk6cwSYd_NSSJ16ROiKYFI"

# Test 1: Using OpenAI compatibility layer (what AI Scientist uses)
print("=" * 60)
print("TEST 1: OpenAI Compatibility Layer")
print("=" * 60)
try:
    import openai
    client = openai.OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'API TEST SUCCESSFUL' if you can read this."}
        ],
        max_tokens=50,
    )
    print("✓ SUCCESS!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")

# Test 2: List models via native Google SDK
print("\n" + "=" * 60)
print("TEST 2: Native Google GenAI SDK - List Models")
print("=" * 60)
try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    models = list(genai.list_models())
    print(f"✓ Found {len(models)} models")
    print("\nAvailable Gemini models:")
    for m in models:
        if "gemini" in m.name.lower():
            print(f"  - {m.name}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")

# Test 3: Native Google SDK generation
print("\n" + "=" * 60)
print("TEST 3: Native Google GenAI SDK - Generate Content")
print("=" * 60)
try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content("Say 'NATIVE SDK TEST SUCCESSFUL'")
    print("✓ SUCCESS!")
    print(f"Response: {response.text[:100]}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
