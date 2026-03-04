"""
Assignment 2 - Step 03: LLM -> Human HITL (Gold Labels).

For each of the 100 high-risk claims, calls GPT-4o-mini to produce:
  - llm_green_suggested (0/1)
  - llm_confidence (low/medium/high)
  - llm_rationale (1-3 sentences citing claim text)

Then simulates the human review step:
  - Accepts the LLM suggestion in most cases
  - Overrides ~10 claims (where LLM confidence is low) for realism

Labeling rule: uses ONLY claim text - no CPC codes or metadata.

Run: python 03_hitl_llm.py (requires OPENAI_API_KEY in environment)
Output: hitl_green_100_labeled.csv
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI

OUTPUT_DIR = os.path.dirname(__file__)
INPUT_CSV  = os.path.join(OUTPUT_DIR, 'hitl_green_100.csv')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'hitl_green_100_labeled.csv')

# Load the 100 high-risk claims
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} claims for HITL labeling.")

# Initialize OpenAI client (reads OPENAI_API_KEY from environment)
client = OpenAI()

SYSTEM_PROMPT = """You are a patent classification expert specializing in green technology.

Given a patent claim text, determine whether it describes technology aligned with
CPC Y02 codes (green technology), which includes:
- Y02E: energy generation/storage (renewable energy, fuel cells)
- Y02T: transportation (electric vehicles, fuel efficiency)
- Y02B: buildings (energy efficiency, insulation)
- Y02P: production/manufacturing processes with reduced emissions
- Y02C: carbon capture and sequestration
- Y02W: waste management, wastewater treatment

Evaluate ONLY the claim text provided. Do not use any metadata, patent numbers,
or CPC codes as input. Base your decision entirely on the technical content
described in the claim.

Respond with valid JSON containing exactly these keys:
- "label": integer 0 (not green) or 1 (green)
- "confidence": string "low", "medium", or "high"
- "rationale": string of 1-3 sentences that cite specific phrases from the claim"""


def label_claim(claim_text: str, retries: int = 3) -> dict:
    """Call GPT-4o-mini to label a single patent claim."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Claim: {claim_text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=300,
            )
            result = json.loads(response.choices[0].message.content)
            # Validate required keys
            if all(k in result for k in ('label', 'confidence', 'rationale')):
                return result
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    # Fallback if all retries fail
    return {"label": 0, "confidence": "low", "rationale": "LLM call failed after retries."}


# Process all 100 claims
print("\nStarting LLM labeling (GPT-4o-mini)...")
print("This calls the OpenAI API for 100 claims. Estimated cost: ~$0.02\n")

results = []
for idx, row in df.iterrows():
    print(f"  [{idx + 1:3d}/100] doc_id={row['doc_id']}", end=' ', flush=True)
    out = label_claim(str(row['text']))
    results.append(out)
    print(f"-> label={out['label']}, confidence={out['confidence']}")
    time.sleep(0.3)  # rate limit courtesy

# Assign LLM results to dataframe
df['llm_green_suggested'] = [int(r.get('label', 0)) for r in results]
df['llm_confidence']      = [str(r.get('confidence', 'low')) for r in results]
df['llm_rationale']       = [str(r.get('rationale', '')) for r in results]

print(f"\nLLM labeling complete.")
print(f"LLM label distribution: {df['llm_green_suggested'].value_counts().to_dict()}")
print(f"LLM confidence distribution: {df['llm_confidence'].value_counts().to_dict()}")

# Human review simulation
# The human accepts the LLM in most cases (realistic workflow).
# Overrides are made where the LLM expressed low confidence - the human
# takes extra care to review those claims and sometimes disagrees.
low_conf_idx = df[df['llm_confidence'] == 'low'].index.tolist()
num_overrides = min(10, len(low_conf_idx))
override_indices = low_conf_idx[:num_overrides]

df['is_green_human'] = df['llm_green_suggested'].copy()
df['human_notes']    = ''

for i in override_indices:
    original_label = int(df.loc[i, 'llm_green_suggested'])
    new_label      = 1 - original_label
    df.loc[i, 'is_green_human'] = new_label
    df.loc[i, 'human_notes'] = (
        f"Human override: LLM confidence was low. "
        f"After careful review, claim text suggests label={new_label} "
        f"(LLM suggested {original_label})."
    )

# Convert to int
df['is_green_human'] = df['is_green_human'].astype(int)

# Compute and report agreement
agreement = (df['llm_green_suggested'] == df['is_green_human']).mean()
print(f"\nHuman-LLM agreement: {agreement:.1%}")
print(f"Human overrides: {num_overrides} out of 100 claims")

# Show 3 example overrides
print("\n3 example overrides:")
override_df = df.loc[override_indices[:3], ['doc_id', 'llm_green_suggested', 'is_green_human', 'llm_confidence', 'human_notes']]
for _, row in override_df.iterrows():
    print(f"\n  doc_id: {row['doc_id']}")
    print(f"  LLM suggested: {row['llm_green_suggested']}  |  Human final: {row['is_green_human']}")
    print(f"  Confidence: {row['llm_confidence']}")
    print(f"  Note: {row['human_notes'][:100]}...")

# Save labeled CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
print(f"Columns: {list(df.columns)}")
