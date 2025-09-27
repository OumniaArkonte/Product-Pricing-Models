import os
import time
import json
import logging
import random
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict
from agno.agent import Agent
from agno.team import Team
from agno.models.google import Gemini
from agno.tools import tool

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------
# Model
# ----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "xxxxxxxxxxx")
gemini_model = Gemini(id="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

# ----------------------------
# Metrics
# ----------------------------
@dataclass
class Metrics:
    start_time: float = field(default_factory=time.time)
    api_calls: int = 0
    external_api_calls: int = 0
    tool_calls: Dict[str, int] = field(default_factory=dict)
    tool_successes: Dict[str, int] = field(default_factory=dict)
    inter_agent_interactions: int = 0
    hallucination_score: float = None
    tokens_input: int = 0
    tokens_output: int = 0

    def record_tool_call(self, name: str, success: bool, external: bool=False):
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
        if success:
            self.tool_successes[name] = self.tool_successes.get(name, 0) + 1
        self.api_calls += 1
        if external:
            self.external_api_calls += 1

    def record_interaction(self, n=1):
        self.inter_agent_interactions += n

metrics = Metrics()

# ----------------------------
# Tools
# ----------------------------
@tool
def mask_pii(text: str) -> str:
    import re
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{10,15}\b', '[PHONE]', text)
    metrics.record_tool_call("mask_pii", True)
    return text

@tool
def search_flights(origin: str, dest: str, date_from: str, date_to: str) -> dict:
    time.sleep(0.2)
    success = random.random() > 0.05
    metrics.record_tool_call("search_flights", success, external=True)
    if not success:
        raise RuntimeError("Flight API failed")
    metrics.tokens_input += 100
    metrics.tokens_output += 250
    return {"flights": [{"airline": "MockAir", "price_mad": 2000, "dep": date_from, "arr": date_to}]}

@tool
def search_hotels(city: str, checkin: str, checkout: str, budget_mad:int) -> dict:
    time.sleep(0.15)
    success = random.random() > 0.1
    metrics.record_tool_call("search_hotels", success, external=True)
    if not success:
        raise RuntimeError("Hotel API failed")
    metrics.tokens_input += 50
    metrics.tokens_output += 150
    return {"hotels": [{"name": "MockHotel", "price_mad": 1200, "city": city}]}

@tool
def hallucination_judge(expected_hint: str, agent_output: str) -> dict:
    metrics.record_tool_call("hallucination_judge", True)
    score = 0.2 if "I don't know" in agent_output or "could be" in agent_output else round(0.7 + 0.3 * random.random(), 2)
    metrics.hallucination_score = score
    metrics.tokens_input += 10
    metrics.tokens_output += 20
    return {"score": score, "note": "0-1 (1=low hallucination)"}

# ----------------------------
# Agents
# ----------------------------
search_agent = Agent(
    name="SearchAgent",
    role="Find flights and hotels.",
    model=gemini_model,
    tools=[search_flights, search_hotels],
)

planner_agent = Agent(
    name="PlannerAgent",
    role="Build final itinerary.",
    model=gemini_model,
)

budget_agent = Agent(
    name="BudgetAgent",
    role="Check budget feasibility.",
    model=gemini_model,
)

guardrail_agent = Agent(
    name="GuardrailAgent",
    role="Apply guardrails.",
    model=gemini_model,
    tools=[mask_pii, hallucination_judge],
)

# ----------------------------
# Team
# ----------------------------
travel_team = Team(
    name="TravelGuardrailTeam",
    model=gemini_model,
    members=[guardrail_agent, search_agent, planner_agent, budget_agent],
    instructions=[
        "All inputs must first be checked by GuardrailAgent.",
        "If GuardrailAgent flags a severe risk, refuse gracefully.",
        "Report KPIs after completion."
    ],
    markdown=True
)

# ----------------------------
# Pipeline
# ----------------------------
def run_travel_pipeline(user_request: str):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Guardrail
    logging.info("Running guardrail pre-check...")
    masked = guardrail_agent.run(f"Mask PII in: {user_request}")
    metrics.record_interaction()

    # Search flights and hotels
    logging.info("SearchAgent: searching flights and hotels...")
    search_output = search_agent.run(
        f"Find flights from Lisbon to Lisbon between 2025-11-10 and 2025-11-13 "
        f"and find hotels in Lisbon within 5000 MAD"
    )
    metrics.record_interaction()

    # Build final itinerary
    logging.info("PlannerAgent: building final itinerary...")
    search_mock = {
        "flights": [{"airline": "MockAir", "price_mad": 2000, "dep": "2025-11-10", "arr": "2025-11-13"}],
        "hotels": [{"name": "MockHotel", "price_mad": 1200, "city": "Lisbon"}]
    }
    plan_prompt = f"""
You are a travel planner AI. Using the following flight and hotel information:

Flights and Hotels: {json.dumps(search_mock)}

Generate a complete, human-readable travel plan in plain text including:
- Daily itinerary (3 days)
- Flight info
- Hotel info
- Total estimated cost
- Notes if budget is exceeded

Output only plain text, no code blocks or extra metadata.
"""
    plan_output = planner_agent.run(plan_prompt)
    plan_text = getattr(plan_output, "content", str(plan_output)).strip()
    metrics.record_interaction()

    # Budget check
    logging.info("BudgetAgent: checking budget...")
    budget_output = budget_agent.run(f"Check if estimated total cost in '{plan_text}' fits budget 5000 MAD")
    budget_text = getattr(budget_output, "content", str(budget_output)).strip()
    metrics.record_interaction()

    # Hallucination check
    logging.info("Running hallucination judge...")
    hallucination_output = guardrail_agent.run(f"Judge hallucination for: {plan_text}")
    hallucination_text = getattr(hallucination_output, "content", str(hallucination_output)).strip()
    metrics.record_interaction()

    # Final reply: cleaned, readable text
    final_reply = (
        f"=== Travel Plan ===\n\n"
        f"{plan_text}\n\n"
        f"---\n"
        f"Budget Check:\n{budget_text}\n\n"
        f"Hallucination Check:\n{hallucination_text}\n"
        f"=================="
    )

    # Save report
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/travel_report_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Travel OS Report\nGenerated: {datetime.now(timezone.utc).isoformat()}Z\n\n")
        f.write(f"User Request:\n{user_request}\n\n")
        f.write(final_reply)

    logging.info(f"Report saved to {filename}")

    # KPIs
    est_cost = round((metrics.tokens_input/1_000_000)*1.5 + (metrics.tokens_output/1_000_000)*5.0, 6)
    kpi = {
        "api_calls_total": metrics.api_calls,
        "external_api_calls": metrics.external_api_calls,
        "tool_calls": metrics.tool_calls,
        "tool_successes": metrics.tool_successes,
        "hallucination_score": metrics.hallucination_score,
        "estimated_cost_usd": est_cost
    }

    print("\n Final Travel Report (cleaned):\n")
    print(final_reply)
    print("\n KPIs:", json.dumps(kpi, indent=2))

    return final_reply, kpi

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    user_req = "Plan a 3-day trip to Lisbon between 2025-11-10 and 2025-11-13 with budget 5000 MAD"
    run_travel_pipeline(user_req)
