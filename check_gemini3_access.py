import os
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI


def main():
    print("Checking access to gemini-3-pro-preview...")

    # Load key from ~/.gemini_key
    gemini_key = None
    gemini_key_path = Path.home() / ".gemini_key"
    if gemini_key_path.exists():
        try:
            with open(gemini_key_path, "r") as f:
                content = f.read().strip()
                content = content.strip("\"'")
                if content:
                    gemini_key = content
                    print(f"Loaded key from {gemini_key_path}")
        except Exception as e:
            print(f"Error reading {gemini_key_path}: {e}")

    if not gemini_key:
        print("Key not found in ~/.gemini_key, checking environment...")
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not gemini_key:
        print("❌ No API key found.")
        return

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview", google_api_key=gemini_key, temperature=0.7
        )
        response = llm.invoke("Hello, verify access.")
        print(f"✅ Access confirmed. Response: {response.content}")
    except Exception as e:
        print(f"❌ Access failed: {e}")


if __name__ == "__main__":
    main()

