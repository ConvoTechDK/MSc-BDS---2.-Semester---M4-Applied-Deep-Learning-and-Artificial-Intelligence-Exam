"""
Final Assignment - Step 02: Multi-Agent System with QLoRA-Informed Advocate.

Two-stage QLoRA integration:
  - Stage 1 (00_qlora_inference.py): the fine-tuned QLoRA adapter runs inference
    on all 100 claims and saves its predictions to qlora_predictions_100.csv.
  - Stage 2 (this script): the Advocate agent receives the QLoRA model's actual
    prediction and raw output as the starting point for its argument. This means
    the fine-tuned model genuinely drives the Advocate's reasoning rather than
    just influencing its system prompt.

Agent roles:
  - Advocate (GPT-4o-mini, informed by QLoRA prediction): argues FOR green
  - Skeptic  (GPT-4o-mini): argues AGAINST (looks for greenwashing)
  - Judge    (GPT-4o-mini): weighs both and outputs final JSON decision

Output: final_labels_100_raw.csv (raw MAS labels, before human HITL review)
Next step: run 04_hitl_review.py interactively to review low-confidence claims,
           then sbatch slurm_finetune_final.sh.

Run on HPC: sbatch slurm_mas_final.sh (no GPU required)
"""

import os
import json
import time
import pandas as pd
from textwrap import dedent
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

OUTPUT_DIR  = os.path.dirname(__file__)
QLORA_CSV   = os.path.join(OUTPUT_DIR, 'qlora_predictions_100.csv')
INPUT_CSV   = os.path.join(OUTPUT_DIR, '..', 'assignment_2', 'hitl_green_100.csv')
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, 'final_labels_100_raw.csv')

# Verify QLoRA predictions exist before starting
if not os.path.isfile(QLORA_CSV):
    raise FileNotFoundError(
        f"QLoRA predictions not found at {QLORA_CSV}. "
        "Run 00_qlora_inference.py (or sbatch slurm_qlora_inference.sh) first."
    )

# Load QLoRA predictions into a lookup dict: doc_id -> {label, output}
df_qlora = pd.read_csv(QLORA_CSV)
qlora_lookup = {
    str(row['doc_id']): {
        'label':  int(row['qlora_label']),
        'output': str(row['qlora_output']),
    }
    for _, row in df_qlora.iterrows()
}
print(f"Loaded QLoRA predictions for {len(qlora_lookup)} claims.")
print(f"QLoRA label distribution: {df_qlora['qlora_label'].value_counts().to_dict()}")

# Single LLM backend for all agents
gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

print("\nAll agents use GPT-4o-mini.")
print("Advocate is informed by real QLoRA model predictions (two-stage integration).")
print("Skeptic and Judge use GPT-4o-mini directly.")


# ---------------------------------------------------------------------------
# Agent and Task definitions
# ---------------------------------------------------------------------------

class PatentAgentsFinal:

    def advocate_agent(self):
        return Agent(
            role="QLoRA-Informed Patent Green Technology Advocate",
            backstory=dedent("""
                You are a patent classification specialist. Before each debate, you
                receive the assessment of a QLoRA language model that was fine-tuned
                on thousands of patent claims using 4-bit quantization and LoRA
                adapters. Your job is to take the QLoRA model's prediction and build
                the strongest possible argument to support it, citing specific phrases
                from the claim text.
            """),
            goal=dedent("""
                Given a patent claim and the QLoRA model's prediction, argue FOR
                the green classification if QLoRA says green, or present the
                QLoRA model's case honestly if it says not green. Always cite
                specific phrases from the claim.
            """),
            llm=gpt_llm,
            verbose=False,
        )

    def skeptic_agent(self):
        return Agent(
            role="Patent Classification Skeptic",
            backstory=dedent("""
                You are an independent patent auditor who identifies greenwashing
                and unsubstantiated green claims in patent applications.
            """),
            goal=dedent("""
                Argue against the green classification if the claim lacks sufficient
                evidence of direct environmental benefit under Y02 standards.
            """),
            llm=gpt_llm,
            verbose=False,
        )

    def judge_agent(self):
        return Agent(
            role="Patent Classification Judge",
            backstory=dedent("""
                You are the chief patent classification officer who makes final
                Y02 classification decisions based on both pro and con arguments.
            """),
            goal=dedent("""
                Weigh the Advocate and Skeptic arguments. Output ONLY valid JSON:
                {"label": 0 or 1, "confidence": "low"/"medium"/"high",
                 "rationale": "1-2 sentences"}
            """),
            llm=gpt_llm,
            verbose=False,
        )


class PatentTasksFinal:

    def advocate_task(self, agent, claim_text: str, qlora_label: int, qlora_output: str):
        qlora_verdict = "GREEN (label=1)" if qlora_label == 1 else "NOT GREEN (label=0)"
        return Task(
            description=dedent(f"""
                You have received the following pre-assessment from a QLoRA language
                model fine-tuned on patent claims:

                  QLoRA verdict:     {qlora_verdict}
                  QLoRA raw output:  "{qlora_output}"

                Now review the claim text below and build the strongest argument that
                supports the QLoRA model's verdict. Cite specific phrases from the claim.

                Claim:
                {claim_text[:600]}
            """),
            expected_output="2-4 sentences arguing for or against green classification, citing the claim.",
            agent=agent,
        )

    def skeptic_task(self, agent, claim_text: str):
        return Task(
            description=dedent(f"""
                Review this patent claim and argue AGAINST Y02 green classification.

                Claim: {claim_text[:600]}

                Look for vague claims, greenwashing, or missing direct environmental benefit.
            """),
            expected_output="2-4 sentences arguing against green classification.",
            agent=agent,
        )

    def judge_task(self, agent):
        return Task(
            description=dedent(f"""
                Weigh the Advocate and Skeptic arguments. Output ONLY valid JSON:
                {{"label": 0 or 1, "confidence": "low"/"medium"/"high",
                 "rationale": "1-2 sentences"}}

                No text outside the JSON object.
            """),
            expected_output='{"label": 1, "confidence": "high", "rationale": "..."}',
            agent=agent,
        )


def classify_claim_final(claim_text: str, qlora_label: int, qlora_output: str) -> dict:
    agent_factory = PatentAgentsFinal()
    task_factory  = PatentTasksFinal()

    advocate = agent_factory.advocate_agent()
    skeptic  = agent_factory.skeptic_agent()
    judge    = agent_factory.judge_agent()

    crew = Crew(
        agents=[advocate, skeptic, judge],
        tasks=[
            task_factory.advocate_task(advocate, claim_text, qlora_label, qlora_output),
            task_factory.skeptic_task(skeptic, claim_text),
            task_factory.judge_task(judge),
        ],
        verbose=False,
    )

    try:
        result = crew.kickoff()
        raw = getattr(result, 'raw', str(result))
        start = raw.rfind('{')
        end   = raw.rfind('}') + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
            if all(k in parsed for k in ('label', 'confidence', 'rationale')):
                return parsed
    except Exception as e:
        print(f"    Crew error: {e}")

    return {"label": 0, "confidence": "low", "rationale": "Agent debate error."}


# ---------------------------------------------------------------------------
# Main: classify all 100 high-risk claims
# ---------------------------------------------------------------------------

print(f"\nLoading 100 high-risk claims from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} claims.")
print("Estimated cost: ~$0.20 (Advocate + Skeptic + Judge on GPT-4o-mini)\n")

records = []
for idx, row in df.iterrows():
    doc_id = str(row['doc_id'])
    qlora  = qlora_lookup.get(doc_id, {'label': 0, 'output': 'no'})

    print(
        f"[{idx + 1:3d}/100] doc_id={doc_id} "
        f"qlora={'GREEN' if qlora['label'] == 1 else 'NOT GREEN'}",
        end=' ', flush=True,
    )

    out = classify_claim_final(str(row['text']), qlora['label'], qlora['output'])
    records.append({
        'doc_id':               doc_id,
        'text':                 row['text'],
        'p_green':              row.get('p_green', ''),
        'u':                    row.get('u', ''),
        'qlora_label':          qlora['label'],
        'qlora_output':         qlora['output'],
        'final_mas_label':      int(out.get('label', 0)),
        'final_mas_confidence': str(out.get('confidence', 'low')),
        'final_mas_rationale':  str(out.get('rationale', '')),
    })
    print(f"-> MAS label={out['label']}, confidence={out['confidence']}")
    time.sleep(0.3)

df_out = pd.DataFrame(records)

# Report agreement between QLoRA prediction and MAS final label
agreement = (df_out['qlora_label'] == df_out['final_mas_label']).mean()
conf_dist  = df_out['final_mas_confidence'].value_counts().to_dict()
low_count  = (df_out['final_mas_confidence'] == 'low').sum()

print(f"\nMAS label distribution:      {df_out['final_mas_label'].value_counts().to_dict()}")
print(f"MAS confidence distribution: {conf_dist}")
print(f"QLoRA vs MAS agreement:      {agreement:.1%}")
print(f"Low-confidence claims (will require HITL): {low_count}/100")

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved raw MAS labels: {OUTPUT_CSV}")
print("\nNext step: run interactively -> python final_assignment/04_hitl_review.py")
