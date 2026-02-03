#!/usr/bin/env python3
"""Script to add fallback LaTeX extraction for initial writeup generation"""
import re

file_path = "/data/ai-tools/SakanaAI/AI-Scientist-v2/ai_scientist/perform_icbinb_writeup.py"

with open(file_path, "r") as f:
    content = f.read()

# Find and replace the initial latex extraction section
old_pattern = r'(print\(f"\{\'=\'\*80\}\\n"\)\s+)\s+latex_code_match = re\.search\(r"```latex\(.*?\)```", response, re\.DOTALL\)\s+if not latex_code_match:\s+return False\s+updated_latex_code = latex_code_match\.group\(1\)\.strip\(\)'

new_code = '''print(f"{\'=\'*80}\\n")

        # Extract LaTeX from initial writeup response
        # Try ```latex blocks first (expected format)
        latex_code_match = re.search(r"```latex(.*?)```", response, re.DOTALL)
        if latex_code_match:
            updated_latex_code = latex_code_match.group(1).strip()
        else:
            # Gemini may not wrap in ```latex blocks
            # Try generic code block
            generic_match = re.search(r"```(.*?)```", response, re.DOTALL)
            if generic_match:
                content_text = generic_match.group(1).strip()
                # Remove possible language marker from start
                lines = content_text.split("\\n")
                if lines and lines[0].strip().lower() in ("tex", "latex", "plaintex"):
                    content_text = "\\n".join(lines[1:])
                updated_latex_code = content_text.strip()
                print(f"[yellow]Extracted from generic code block in initial generation[/yellow]")
            elif "\\\\documentclass" in response or "\\\\begin{document}" in response:
                # Last resort: whole response is LaTeX
                updated_latex_code = response.strip()
                print(f"[yellow]Using full response as LaTeX (no code blocks) in initial generation[/yellow]")
            else:
                print(f"[red]No valid LaTeX found in initial generation.[/red]")
                return False'''

# Simple replacement - find line with "latex_code_match = re.search" and replace the next 4 lines
lines = content.split("\n")
result = []
i = 0
while i < len(lines):
    if "latex_code_match = re.search(r\"```latex" in lines[i]:
        # Found the target line - check if it's the initial one (not reflection)
        # Look backwards to confirm it's after "print(f"{\'=\'\*80}\\n")"
        is_initial = False
        for j in range(max(0, i-5), i):
            if "print(f\"{\'=\'*80" in lines[j]:
                is_initial = True
                break
        
        if is_initial and i+3 < len(lines) and "if not latex_code_match:" in lines[i+1]:
            # This is the initial one - replace it
            result.append(lines[i-1])  # Keep the print line
            result.append("")
            result.append("        # Extract LaTeX from initial writeup response")
            result.append("        # Try ```latex blocks first (expected format)")
            result.append("        latex_code_match = re.search(r\"```latex(.*?)```\", response, re.DOTALL)")
            result.append("        if latex_code_match:")
            result.append("            updated_latex_code = latex_code_match.group(1).strip()")
            result.append("        else:")
            result.append("            # Gemini may not wrap in ```latex blocks")
            result.append("            # Try generic code block")
            result.append("            generic_match = re.search(r\"```(.*?)```\", response, re.DOTALL)")
            result.append("            if generic_match:")
            result.append("                content_text = generic_match.group(1).strip()")
            result.append("                # Remove possible language marker from start")
            result.append("                lines_split = content_text.split(\"\\n\")")
            result.append("                if lines_split and lines_split[0].strip().lower() in (\"tex\", \"latex\", \"plaintex\"):")
            result.append("                    content_text = \"\\n\".join(lines_split[1:])")
            result.append("                updated_latex_code = content_text.strip()")
            result.append("                print(f\"[yellow]Extracted from generic code block in initial generation[/yellow]\")")
            result.append("            elif \"\\\\documentclass\" in response or \"\\\\begin{document}\" in response:")
            result.append("                # Last resort: whole response is LaTeX")
            result.append("                updated_latex_code = response.strip()")
            result.append("                print(f\"[yellow]Using full response as LaTeX (no code blocks) in initial generation[/yellow]\")")
            result.append("            else:")
            result.append("                print(f\"[red]No valid LaTeX found in initial generation.[/red]\")")
            result.append("                return False")
            i += 4  # Skip the old lines
            continue
    
    result.append(lines[i])
    i += 1

with open(file_path, "w") as f:
    f.write("\n".join(result))

print("Fix applied successfully!")
