import time, logging, os, re
from agents import (
    real_estate_team, offtopic_agent, hallucination_agent, pii_detector_agent,
    goal_agent, judge_agent, factcheck_agent, format_agent, comparer_agent,
    benchmark_agent, mistral_model, gemini_model
)
from google.genai.errors import ClientError
from agno.exceptions import ModelProviderError


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

prompt = (
    "Collect all property data from Airtable, mask PII, predict prices, "
    "analyze trends, benchmark against market data, and produce a final client-ready report."
)
# ==============================
# Metrics tracking
# ==============================
metrics = {
    "tasks": [],  # liste de dicts {name, success, completion, latency_sec, tokens_used}
    "total_latency": 0.0,
    "total_tokens": 0
}


# ==============================
# Guardrail regex-based checks
# ==============================

def detect_prompt_injection(prompt: str) -> bool:
    injection_patterns = [
        r"ignore\s+previous\s+instructions",
        r"override\s+rules",
        r"delete\s+.*",
        r"system\s+prompt",
        r"forget\s+instructions"
    ]
    return any(re.search(pattern, prompt, re.IGNORECASE) for pattern in injection_patterns)

def detect_misuse(prompt: str) -> bool:
    forbidden_topics = ["weapons", "drugs", "politics", "violence", "hacking"]
    return any(topic in prompt.lower() for topic in forbidden_topics)

def is_on_topic(text: str) -> bool:
    real_estate_keywords = [
        "property", "real estate", "apartment", "villa", "rent",
        "buy", "sell", "market", "housing", "estate"
    ]
    return any(kw in text.lower() for kw in real_estate_keywords)

def detect_bias(text: str) -> bool:
    sensitive_terms = ["race", "ethnicity", "religion", "gender", "sexual orientation"]
    return any(term in text.lower() for term in sensitive_terms)

# ==============================
# Guardrail + Evaluation step
# ==============================

def run_guardrails(response_text: str):
    results = {}

    # 1. Regex-based checks
    results["Bias"] = "Possible bias" if detect_bias(response_text) else "OK"
    results["OnTopic (regex)"] = "Off-topic" if not is_on_topic(response_text) else "OK"

    # 2. LLM guardrail agents
    results["OffTopic (LLM)"] = offtopic_agent.run(
        f"Check if this text is off-topic: {response_text[:1500]}"
    ).content.strip()

    results["Hallucination"] = hallucination_agent.run(
        f"Check this report for hallucinations or fabricated content: {response_text[:2000]}"
    ).content.strip()

    results["PII"] = pii_detector_agent.run(
        f"Check if there is any PII (emails, phones, names) in this text: {response_text[:2000]}"
    ).content.strip()

    # 3. Nouveaux agents
    results["Goal Adherence"] = goal_agent.run(
        f"Does this report meet the expected goals (clarity, PII masking, insights, recommendations)?\nText: {response_text[:2000]}"
    ).content.strip()

    results["Judge (Quality Score)"] = judge_agent.run(
        f"Give a score (0-10) with justification for this report:\n{response_text[:2000]}"
    ).content.strip()

    results["FactCheck"] = factcheck_agent.run(
        f"Fact-check this report and highlight doubtful or false claims:\n{response_text[:2000]}"
    ).content.strip()

    results["Format (JSON enforced)"] = format_agent.run(
        f"Restructure this report into JSON with keys: title, summary, predictions, insights, recommendations.\nText: {response_text[:2000]}"
    ).content.strip()

    return results

# ==============================
# Pipeline
# ==============================

MAX_STEPS = 20
step_counter = 0
max_retries = 5
retry_delay = 10
attempt = 0

# --- Guardrails before execution ---
if detect_prompt_injection(prompt):
    logging.error("Prompt injection attempt detected. Aborting pipeline.")
    exit(1)

if detect_misuse(prompt):
    logging.error("Misuse detected (forbidden topic). Aborting pipeline.")
    exit(1)

while attempt < max_retries:
    step_counter += 1
    if step_counter > MAX_STEPS:
        logging.error("Infinite loop detected. Aborting pipeline.")
        break

    try:
        logging.info("^^ Launching Real Estate Pricing OS pipeline...\n")

        result = real_estate_team.run(prompt)

        # === Extract text from TeamRunOutput (robuste) ===
        response_text = None
        if hasattr(result, "content") and result.content:
            response_text = result.content
        elif hasattr(result, "output_text") and result.output_text:
            response_text = result.output_text
        elif hasattr(result, "messages") and result.messages:
            response_text = "\n".join(
                [m.get("content", "") for m in result.messages if isinstance(m, dict)]
            )
        elif hasattr(result, "to_string"):
            response_text = result.to_string()
        else:
            response_text = str(result)

        if not response_text or len(response_text) < 50:
            logging.warning("Output may be invalid or incomplete")
        else:
            logging.info("Report successfully generated")

            # === Guardrails + Evaluations ===
            eval_results = run_guardrails(response_text)

            print("\n" + "="*60)
            print("Final Report:\n")
            print(response_text)
            print("\n" + "="*60)
            print("Evaluation Results (Guardrails + Metrics):")
            for k, v in eval_results.items():
                print(f"- {k}: {v[:500]}")
            print("="*60)

            # === Comparaison Gemini vs Mistral corrigé ===
            # --- Mistral ---
            mistral_resp = mistral_model.run(prompt)
            if hasattr(mistral_resp, "content") and mistral_resp.content:
                mistral_report = mistral_resp.content
            elif hasattr(mistral_resp, "messages") and mistral_resp.messages:
                mistral_report = "\n".join(
                    [m.get("content", "") for m in mistral_resp.messages if isinstance(m, dict)]
                )
            else:
                mistral_report = str(mistral_resp)

            # --- Gemini ---
            gemini_resp = gemini_model.generate_content(prompt)
            if (
                hasattr(gemini_resp, "candidates")
                and gemini_resp.candidates
                and hasattr(gemini_resp.candidates[0], "content")
                and hasattr(gemini_resp.candidates[0].content, "parts")
                and gemini_resp.candidates[0].content.parts
            ):
                gemini_report = gemini_resp.candidates[0].content.parts[0].text
            else:
                gemini_report = "No output"

            comparison = comparer_agent.run(
                f"Compare these two reports and decide which is better in clarity, accuracy, and completeness:\n\n[Mistral]\n{mistral_report[:1500]}\n\n[Gemini]\n{gemini_report[:1500]}"
            ).content.strip()
            print("\nComparison Gemini vs Mistral:\n", comparison)

            # === Save report to Markdown file ===
            os.makedirs("reports", exist_ok=True)
            filename = f"reports/report_{time.strftime('%Y%m%d_%H%M%S')}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(response_text + "\n\n")
                f.write("# Evaluation Results\n")
                for k, v in eval_results.items():
                    f.write(f"- {k}: {v}\n")
                f.write("\n## Comparison Gemini vs Mistral\n")
                f.write(comparison)
            logging.info(f"Report saved to {filename}")

        break

    except (ClientError, ModelProviderError) as e:
        if hasattr(e, "status_code") and e.status_code == 429:
            attempt += 1
            wait_time = retry_delay * attempt
            logging.warning(f"Gemini quota exceeded. Retry {attempt}/{max_retries} in {wait_time}s...")
            time.sleep(wait_time)
        elif "429" in str(e):
            attempt += 1
            wait_time = retry_delay * attempt
            logging.warning(f"Provider quota exceeded. Retry {attempt}/{max_retries} in {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise e

else:
    logging.error("Pipeline failed after multiple retries. Check quotas or wait before retrying.")
