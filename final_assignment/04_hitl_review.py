"""
Final Assignment - Step 04: Interactive Exception-Based HITL Review.

Reads the raw MAS labels from final_labels_100_raw.csv and presents
any low-confidence claims for human review in the terminal.

Exception-based HITL means you only review claims where the Judge
agent assigned confidence="low". All other claims are accepted as-is.
If no claims are low-confidence, the script finishes immediately.

Run interactively on the HPC login node (not via SLURM):
    python final_assignment/04_hitl_review.py

Output: final_labels_100.csv (with is_green_gold column)
Next step: sbatch slurm_finetune_final.sh
"""

import os
import pandas as pd

OUTPUT_DIR = os.path.dirname(__file__)
RAW_CSV    = os.path.join(OUTPUT_DIR, 'final_labels_100_raw.csv')
FINAL_CSV  = os.path.join(OUTPUT_DIR, 'final_labels_100.csv')

if not os.path.isfile(RAW_CSV):
    raise FileNotFoundError(
        f"Raw MAS labels not found at {RAW_CSV}. "
        "Run 02_mas_qlora.py (sbatch slurm_mas_final.sh) first."
    )

df = pd.read_csv(RAW_CSV)
print(f"Loaded {len(df)} claims from {RAW_CSV}")
print(f"MAS confidence distribution: {df['final_mas_confidence'].value_counts().to_dict()}")

# Initialize gold labels from MAS output
df['is_green_gold']     = df['final_mas_label'].astype(int)
df['hitl_flag']         = False
df['human_notes_final'] = ''

# Identify claims needing human review
needs_review = df[df['final_mas_confidence'] == 'low'].index.tolist()
print(f"\nException-based HITL: {len(needs_review)}/100 claims flagged for human review.")

if len(needs_review) == 0:
    print("No low-confidence claims. All MAS labels accepted as gold labels.")
else:
    print(f"\nYou will now review {len(needs_review)} claim(s).")
    print("For each claim: press Enter to accept the MAS label, or type 0/1 to override.")
    print("=" * 70)

    overrides = 0
    for i, idx in enumerate(needs_review):
        row = df.loc[idx]

        print(f"\n[Claim {i + 1} of {len(needs_review)}]")
        print(f"doc_id:          {row['doc_id']}")
        print(f"Claim text:\n{str(row['text'])[:500]}")
        print(f"\nQLoRA verdict:   {'GREEN' if int(row.get('qlora_label', 0)) == 1 else 'NOT GREEN'} "
              f"(raw: '{row.get('qlora_output', '')}')")
        print(f"MAS label:       {int(row['final_mas_label'])} "
              f"({'green' if int(row['final_mas_label']) == 1 else 'not green'})")
        print(f"MAS confidence:  {row['final_mas_confidence']}")
        print(f"MAS rationale:   {row['final_mas_rationale']}")

        while True:
            response = input(
                f"\nAccept MAS label [{int(row['final_mas_label'])}]? "
                "Press Enter to accept, or type 0/1 to override: "
            ).strip()

            if response == '':
                df.loc[idx, 'human_notes_final'] = (
                    f"Human reviewed and accepted MAS label {int(row['final_mas_label'])}."
                )
                print(f"Accepted: label = {int(row['final_mas_label'])}")
                break

            elif response in ('0', '1'):
                new_label = int(response)
                df.loc[idx, 'is_green_gold']     = new_label
                df.loc[idx, 'hitl_flag']         = True
                df.loc[idx, 'human_notes_final'] = (
                    f"Human override: MAS={int(row['final_mas_label'])}, "
                    f"human={new_label}."
                )
                print(f"Override recorded: {int(row['final_mas_label'])} -> {new_label}")
                overrides += 1
                break

            else:
                print("Invalid input. Press Enter to accept, or type 0 or 1.")

    print("\n" + "=" * 70)
    print(f"HITL review complete.")
    print(f"  Claims reviewed: {len(needs_review)}")
    print(f"  Labels overridden: {overrides}")
    print(f"  Labels accepted as-is: {len(needs_review) - overrides}")

# Summary
gold_dist = df['is_green_gold'].value_counts().to_dict()
n_hitl    = df['hitl_flag'].sum()
print(f"\nFinal gold label distribution: {gold_dist}")
print(f"Claims that went through HITL: {n_hitl}/100")

df.to_csv(FINAL_CSV, index=False)
print(f"\nSaved: {FINAL_CSV}")
print("Next step: sbatch final_assignment/slurm_finetune_final.sh")
