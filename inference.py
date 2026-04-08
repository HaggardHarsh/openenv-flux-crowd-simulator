"""
OpenEnv Inference Script
========================
Runs an LLM agent against the Crowd Management OpenEnv tasks.
Requirements: 
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN
"""

import os
import sys
import json

# Ensure we can import the environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crowd_env.environment import CrowdManagementEnv
from crowd_env.models import Action

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)


SYSTEM_PROMPT = """
You are an expert AI crowd management system.
You are managing a stadium with 6 interconnected zones.
Your goal is to maintain safe crowd densities (ideally < 2.0 ppm2, max < 3.5 ppm2).
If a zone reaches > 5.0 ppm2 (stampede), you immediately fail the episode.

You can take one of the following actions every step:
- {"action_type": "redirect", "source_zone": "A", "target_zone": "B"} (Moves flow from A to B)
- {"action_type": "gate_control", "source_zone": "A", "gate_index": 0, "gate_open": false} (Closes gate 0 in zone A)
- {"action_type": "gate_control", "source_zone": "A", "gate_index": 0, "gate_open": true} (Opens gate 0 in zone A)
- {"action_type": "alert", "source_zone": "A"} (Issues an alert to zone A, reducing panic and spread)
- {"action_type": "no_op"} (Take no action)

Examine the current observation. If there are zones with `risk_level` of "critical" or "elevated", take action.
1. Redirect from dense zones to less dense neighboring zones.
2. Issue alerts to dense zones.
3. If entry zones (A or E) are critical, close a gate. Otherwise, if safe, ensure gates are open to allow throughput.

You MUST respond with ONLY valid JSON containing your chosen action. No markdown formatting or extra text.
Example response:
{"action_type": "redirect", "source_zone": "C", "target_zone": "D"}
"""

def extract_action(response_text: str) -> Action:
    """Safely extracts the action dict from LLM response."""
    try:
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        parsed = json.loads(clean_text.strip())
        return Action(**parsed)
    except Exception:
        return Action.noop()


def run_inference_task(client: OpenAI, model_name: str, task_id: str, seed: int = 42) -> float:
    """Run the LLM agent against a single task using the strict stdout format."""
    env = CrowdManagementEnv()
    obs = env.reset(seed=seed, options={"task": task_id})
    
    # We will run a max of 30 steps for the LLM to save time during validation.
    MAX_EVAL_STEPS = 30
    steps = 0
    total_reward = 0.0

    while True:
        # Pre-Submission log requirement [START]
        print(f"[START] {json.dumps({'observation': obs.model_dump()})}")
        
        current_state = json.dumps(obs.model_dump(), indent=2)
        user_prompt = f"Current Observation:\n{current_state}\n\nWhat is your next action?"

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
            )
            response_text = response.choices[0].message.content
            action = extract_action(response_text)
        except Exception as e:
            # Fallback to no-op on failure
            action = Action.noop()

        # Pre-Submission log requirement [STEP]
        print(f"[STEP] {json.dumps(action.model_dump())}")

        # Step env
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward
        steps += 1

        if result.terminated or result.truncated or steps >= MAX_EVAL_STEPS:
            # Pre-Submission log requirement [END]
            
            # The grader gives us a deterministic score for full episode completion
            grade = env.grade()
            
            # Strictly output the end marker
            print(f"[END] {json.dumps({'reward': grade.score, 'total_reward': total_reward})}")
            break

    return grade.score

def main():
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        # Some evaluators pass exactly what is written, fallback gently just in case
        print("WARNING: HF_TOKEN environment variable is not set. Assuming unauthenticated or default credentials.")
        hf_token = "dummy_token_if_needed"

    try:
        client = OpenAI(api_key=hf_token, base_url=api_base_url)
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        sys.exit(1)

    # Run for all tasks
    for task in ["easy", "medium", "hard"]:
        run_inference_task(client, model_name, task)

if __name__ == "__main__":
    main()
