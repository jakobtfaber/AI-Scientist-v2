"""Prompt templates for data-grounded writeup and validation"""

DATA_GROUNDED_WRITEUP_TEMPLATE = """
You are writing a scientific paper based on experimental results.

CRITICAL RULES:
1. Every quantitative claim MUST be supported by the raw data provided
2. If figures show X but data shows Y, TRUST THE DATA and flag the discrepancy
3. Do not make claims about "independence" or "invariance" without checking coefficient of variation
4. When you see contradictions, explicitly state: "POTENTIAL ISSUE: [description]"

# Experimental Data

{data_summary}

# Figure Descriptions

{figure_descriptions}

# Your Task

{task_description}

REQUIREMENTS:
- Cite specific data values when making quantitative claims
- If claiming "X is independent of Y", verify that CV < 0.2
- If claiming "X times speedup", cite the exact ratio from data
- If you cannot verify a claim with the provided data, say so explicitly
"""

CONSISTENCY_CHECK_PROMPT = """
You are a scientific fact-checker. Your job is to find contradictions between paper text and experimental data.

# Paper Text (LaTeX)

{paper_text}

# Experimental Data

{experiment_data}

# Task

Extract all quantitative claims from the paper and validate them against the data.

For each claim, output:
```json
{{
  "claim": "exact quote from paper",
  "location": "section/line reference",
  "type": "independence|speedup|statistical|measurement",
  "validated": true|false,
  "data_evidence": {{...}},
  "issue_severity": "none|minor|major|critical",
  "explanation": "why this is/isn't valid"
}}
```

FOCUS ON:
- Independence claims (check if CV > 0.2)
- Speedup claims (verify exact ratios)
- Statistical significance (verify p-values if claimed)
- Figure-text alignment

BE EXTREMELY PEDANTIC. If you find ANY mismatch, flag it.
"""

ADVERSARIAL_REVIEW_PROMPT = """
You are a hostile peer reviewer trying to REJECT this paper. Your reputation depends on finding errors.

# Paper Draft

{paper_text}

# Experimental Data

{experiment_data}

# Figures

{figure_descriptions}

# Your Mission

Find EVERY contradiction, error, or questionable claim. Check:

1. **Figure-Text Contradictions**
   - Does text claim X but figure shows Y?
   - Example: Text says "independent" but figure shows 9× variance

2. **Data-Text Contradictions**
   - Do numerical claims match the data?
   - Are statistical tests appropriate?

3. **Internal Inconsistencies**
   - Does the paper contradict itself?
   - Do conclusions match results?

4. **Methodological Flaws**
   - Is the experimental design sound?
   - Are controls missing?

For each issue, output:
```
ISSUE: [one-line description]
SEVERITY: critical|major|minor
LOCATION: [section/figure/line]
EVIDENCE: [quote from paper + contradicting data]
IMPACT: [why this matters]
```

Be MERCILESS. Even small inconsistencies should be flagged.
"""

STATISTICS_REVIEW_PROMPT = """
You are a statistics expert reviewing this paper for methodological rigor.

# Paper Text

{paper_text}

# Experimental Data

{experiment_data}

# Check List

1. **Independence Claims**
   - If claiming X is independent of Y, verify CV(Y|X) < 0.2
   - Check for confounding variables

2. **Significance Tests**
   - Are p-values calculated correctly?
   - Is multiple comparison correction applied?
   - Is sample size sufficient?

3. **Error Bars**
   - Are error bars present and correct?
   - Is variance reported appropriately?

4. **Statistical Claims**
   - "Significant" requires p < 0.05
   - "No difference" requires equivalence test, not just p > 0.05

For each statistical issue, output:
```json
{{
  "issue": "description",
  "severity": "critical|major|minor",
  "claim": "quote from paper",
  "correct_approach": "what should be done",
  "data_check": {{...}}
}}
```
"""

METHODS_REVIEW_PROMPT = """
You are an experimental design expert reviewing the methodology.

# Paper Methods Section

{methods_text}

# Experiment Code

{experiment_code}

# Common Pitfalls to Check

1. **Timing Issues**
   - Is data generation inside timing blocks?
   - Are GPU syncs missing (leads to wrong timing)?
   - Is warmup run performed?

2. **Confounding Variables**
   - Are multiple variables changed at once?
   - Are there missing controls?

3. **Sample Size**
   - Is N=1 presenting as generalizable?
   - Are outliers handled appropriately?

4. **Reproducibility**
   - Are random seeds set?
   - Are environment details specified?

For each methodological flaw, output:
```json
{{
  "flaw": "description",
  "severity": "critical|major|minor",
  "location": "code line or paper section",
  "fix": "how to correct it"
}}
```
"""

VLM_ANALYSIS_PROMPT = """
Analyze this figure carefully and extract:

1. **X-axis**: What variable is plotted?
2. **Y-axis**: What metric is measured?
3. **Key Observations**: 
   - Trends (increasing, decreasing, constant)
   - Outliers or anomalies
   - Relative magnitudes (e.g., "A is 9× larger than B")
4. **Data Values**: Approximate numerical values if visible

If the figure shows distributions (e.g., bar charts across categories):
- List each category and its approximate value
- Calculate coefficient of variation if possible
- Note if variance is high or low

Be PRECISE with numbers. If the chart shows exact values, report them.
"""

REWRITE_CLAIM_PROMPT = """
The paper makes an incorrect claim that contradicts the experimental data.

# Original Claim (WRONG)

{original_claim}

# Actual Data

{data_evidence}

# Task

Rewrite the claim to accurately reflect the data. Keep the same writing style and tone.

BEFORE (wrong): "The runtime is independent of noise distribution"
DATA: gaussian=1.8s, uniform=0.2s, exponential=0.2s (CV=0.89)
AFTER (correct): "The runtime shows significant dependence on noise distribution, with Gaussian noise exhibiting 9× slower performance (1.8s vs 0.2s) compared to uniform or exponential distributions."

Your rewrite:
"""
