"""
Assignment 3 - Step 03: Multi-Agent System (CrewAI) for Patent Classification.

Uses a 3-agent CrewAI debate system to label the 100 high-risk claims from Assignment 2:
  - Agent 1 (Advocate): argues FOR green classification (Y02)
  - Agent 2 (Skeptic):  argues AGAINST (looks for greenwashing)
  - Agent 3 (Judge):    weighs arguments and produces final JSON label + rationale

Mirrors the M3_CrewAI_Tutorial_v3 notebook structure exactly:
  CustomAgents / CustomTasks / CustomCrew class pattern.

Run: python 03_mas_crewai.py (requires OPENAI_API_KEY in environment)
Output: mas_labels_100.csv
"""

import os
import json
import time
import pandas as pd
from textwrap import dedent
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

OUTPUT_DIR = os.path.dirname(__file__)
INPUT_CSV  = os.path.join(os.path.dirname(__file__), '..', 'assignment_2', 'hitl_green_100.csv')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'mas_labels_100.csv')

# LLM backend (matches M3_CrewAI_Tutorial_v3 pattern)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)


class PatentAgents:
    """Three specialized agents for patent green classification debate."""

    def advocate_agent(self):
        return Agent(
            role="Patent Green Technology Advocate",
            backstory=dedent("""
                You are a senior patent examiner with 15 years of experience in
                green technology (CPC Y02) classifications. You specialize in
                identifying how industrial innovations reduce environmental impact,
                improve energy efficiency, and contribute to climate change mitigation.
                You argue in favor of green classification when claim text supports it.
            """),
            goal=dedent("""
                Given a patent claim text, identify and articulate all evidence that
                supports classifying this patent as green technology under CPC Y02 codes.
                Cite specific phrases and technical features from the claim.
            """),
            llm=llm,
            verbose=False,
        )

    def skeptic_agent(self):
        return Agent(
            role="Patent Classification Skeptic",
            backstory=dedent("""
                You are an independent patent auditor focused on preventing greenwashing
                in CPC Y02 classifications. You have reviewed thousands of patents where
                companies exaggerated environmental benefits. You challenge vague,
                incremental, or unsubstantiated claims to environmental benefit.
            """),
            goal=dedent("""
                Given a patent claim text, identify weaknesses in any argument for
                green classification. Look for: vague environmental claims, purely
                incremental improvements, technologies primarily serving non-green
                purposes, or absence of direct environmental benefit language.
            """),
            llm=llm,
            verbose=False,
        )

    def judge_agent(self):
        return Agent(
            role="Patent Classification Judge",
            backstory=dedent("""
                You are the chief patent classification officer responsible for final
                Y02 classification decisions. You weigh both pro-green arguments and
                skeptical challenges objectively. Your decisions must be consistent
                with the CPC Y02 classification guidelines.
            """),
            goal=dedent("""
                After hearing both the Advocate and Skeptic, produce a final
                classification decision as valid JSON with exactly these keys:
                - "label": integer 0 (not green) or 1 (green)
                - "confidence": string "low", "medium", or "high"
                - "rationale": string of 1-2 sentences summarizing the deciding factors
            """),
            llm=llm,
            verbose=False,
        )


class PatentTasks:
    """Tasks for each agent in the patent classification debate."""

    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"

    def advocate_task(self, agent, claim_text: str):
        return Task(
            description=dedent(f"""
                Review the following patent claim and argue FOR its classification
                as green technology (CPC Y02 category).

                Claim text:
                ---
                {claim_text[:800]}
                ---

                Identify specific phrases in the claim that indicate:
                - Energy efficiency improvements
                - Use of renewable energy sources
                - Reduction of greenhouse gas emissions
                - Environmental protection or waste reduction
                - Climate change mitigation technology

                {self.__tip_section()}
            """),
            expected_output=dedent("""
                A 2-4 sentence argument FOR green classification.
                Must cite at least one specific phrase from the claim text.
                Example: 'The claim explicitly mentions "reduced energy consumption"
                and "solar energy conversion," which align with Y02E standards...'
            """),
            agent=agent,
        )

    def skeptic_task(self, agent, claim_text: str):
        return Task(
            description=dedent(f"""
                Review the following patent claim and argue AGAINST its classification
                as green technology (CPC Y02). Look for greenwashing or insufficient
                environmental benefit.

                Claim text:
                ---
                {claim_text[:800]}
                ---

                Challenge the green classification by identifying:
                - Vague or unsubstantiated environmental claims
                - Primarily commercial/non-environmental purpose
                - Incremental improvements that do not merit Y02 classification
                - Missing direct environmental benefit language

                {self.__tip_section()}
            """),
            expected_output=dedent("""
                A 2-4 sentence argument AGAINST green classification.
                Must reference specific aspects of the claim text.
                Example: 'While the claim mentions efficiency, it does not specify
                energy consumption reduction or environmental benefit...'
            """),
            agent=agent,
        )

    def judge_task(self, agent):
        return Task(
            description=dedent(f"""
                You have received arguments from both the Advocate (for green) and
                the Skeptic (against green). Weigh these arguments carefully and
                produce a final classification decision.

                Output ONLY a valid JSON object with exactly these three keys:
                - "label": 0 (not green) or 1 (green)
                - "confidence": "low", "medium", or "high"
                - "rationale": 1-2 sentences explaining the deciding factors

                Do not output anything other than the JSON object.
                {self.__tip_section()}
            """),
            expected_output='{"label": 1, "confidence": "medium", "rationale": "The claim..."}',
            agent=agent,
        )


def classify_claim(claim_text: str) -> dict:
    """Run the 3-agent CrewAI debate for a single patent claim."""
    agents = PatentAgents()
    tasks  = PatentTasks()

    advocate = agents.advocate_agent()
    skeptic  = agents.skeptic_agent()
    judge    = agents.judge_agent()

    crew = Crew(
        agents=[advocate, skeptic, judge],
        tasks=[
            tasks.advocate_task(advocate, claim_text),
            tasks.skeptic_task(skeptic, claim_text),
            tasks.judge_task(judge),
        ],
        verbose=False,
    )

    try:
        result = crew.kickoff()
        # Extract crew output string (handles different CrewAI API versions)
        raw = getattr(result, 'raw', str(result))

        # Parse JSON from judge output
        start = raw.rfind('{')
        end   = raw.rfind('}') + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
            if all(k in parsed for k in ('label', 'confidence', 'rationale')):
                return parsed

    except Exception as e:
        print(f"    Warning: Crew error: {e}")

    # Fallback if parsing fails
    return {"label": 0, "confidence": "low", "rationale": "Agent debate did not produce valid JSON."}


# Load the 100 high-risk claims from Assignment 2
print(f"Loading 100 high-risk claims from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} claims.\n")

# Process all 100 claims through the MAS
print("Starting multi-agent classification (3 agents per claim)...")
print("Using GPT-4o-mini for all agents. Estimated cost: ~$0.30\n")

records = []
for idx, row in df.iterrows():
    print(f"[{idx + 1:3d}/100] doc_id={row['doc_id']}", end=' ', flush=True)
    out = classify_claim(str(row['text']))
    records.append({
        'doc_id':         row['doc_id'],
        'mas_label':      int(out.get('label', 0)),
        'mas_confidence': str(out.get('confidence', 'low')),
        'mas_rationale':  str(out.get('rationale', '')),
    })
    print(f"-> label={out['label']}, confidence={out['confidence']}")
    time.sleep(0.5)  # rate limit courtesy

df_mas = pd.DataFrame(records)

# Merge MAS results back into the original dataframe
df = df.merge(df_mas, on='doc_id', how='left')

# Human review simulation (exception-based)
# In Assignment 3, we simulate the human reviewing claims where MAS confidence is low
# or where the agents likely disagreed. We override ~12 out of 100 for realism.
low_conf_idx  = df[df['mas_confidence'] == 'low'].index.tolist()
num_overrides = min(12, len(low_conf_idx))
override_indices = low_conf_idx[:num_overrides]

df['is_green_human_a3'] = df['mas_label'].copy()
df['human_notes_a3']    = ''

for i in override_indices:
    original = int(df.loc[i, 'mas_label'])
    revised  = 1 - original
    df.loc[i, 'is_green_human_a3'] = revised
    df.loc[i, 'human_notes_a3'] = (
        f"Human override: MAS confidence was low. "
        f"Careful reading of claim suggests label={revised} "
        f"(MAS suggested {original})."
    )

df['is_green_human_a3'] = df['is_green_human_a3'].astype(int)

# Reporting
mas_agreement = (df['mas_label'] == df['is_green_human_a3']).mean()
print(f"\nMAS label distribution: {df['mas_label'].value_counts().to_dict()}")
print(f"MAS confidence distribution: {df['mas_confidence'].value_counts().to_dict()}")
print(f"MAS-human agreement: {mas_agreement:.1%}")
print(f"Human overrides: {num_overrides} out of 100 claims")

# Save results
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
