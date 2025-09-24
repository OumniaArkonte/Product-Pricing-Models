# run_pipeline_secure.py
import time, logging, os, re
from agents import real_estate_team, offtopic_agent, hallucination_agent, pii_detector_agent
from google.genai.errors import ClientError
from agno.exceptions import ModelProviderError

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

prompt = (
    "Collect all property data from Airtable, mask PII, predict prices, "
    "analyze trends, benchmark against market data, and produce a final client-ready report."
)

# ==============================
#  Guardrail regex-based checks
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


# Infinite loop guard
MAX_STEPS = 20
step_counter = 0

# ==============================
#         Pipeline
# ==============================

max_retries = 5
retry_delay = 10  
attempt = 0

# --- Guardrails before execution ---
if detect_prompt_injection(prompt):
    logging.error(" Prompt injection attempt detected. Aborting pipeline.")
    exit(1)

if detect_misuse(prompt):
    logging.error(" Misuse detected (forbidden topic). Aborting pipeline.")
    exit(1)

while attempt < max_retries:
    step_counter += 1
    if step_counter > MAX_STEPS:
        logging.error(" Infinite loop detected. Aborting pipeline.")
        break

    try:
        logging.info(" ^^ Launching Real Estate Pricing OS pipeline...\n")

        result = real_estate_team.run(prompt)

        # === Extract text from TeamRunOutput ===
        response_text = getattr(result, "content", None)
        if response_text is None and hasattr(result, "to_string"):
            response_text = result.to_string()
        if response_text is None:
            response_text = str(result)

        # === Guardrails on output ===
        guardrail_results = {}

        # 1. Regex-based checks
        guardrail_results["Bias"] = " Possible bias" if detect_bias(response_text) else "OK"
        guardrail_results["OnTopic (regex)"] = " Off-topic" if not is_on_topic(response_text) else "OK"

        # 2. AI Guardrail agents
        guardrail_results["OffTopic (LLM)"] = offtopic_agent.run(
            f"Check if this text is off-topic: {response_text[:1500]}"
        ).content.strip()

        guardrail_results["Hallucination"] = hallucination_agent.run(
            f"Check this report for hallucinations or fabricated content: {response_text[:2000]}"
        ).content.strip()

        guardrail_results["PII"] = pii_detector_agent.run(
            f"Check if there is any PII (emails, phones, names) in this text: {response_text[:2000]}"
        ).content.strip()

        # === Output drift / sanity check ===
        if not response_text or len(response_text) < 50:
            logging.warning(" Output may be invalid or incomplete")
        else:
            logging.info(" Report successfully generated")
            print("\n" + "="*60)
            print(" Final Report:\n")
            print(response_text)
            print("\n" + "="*60)
            print(" Guardrail Results:")
            for k, v in guardrail_results.items():
                print(f"- {k}: {v}")
            print("="*60)

            # === Save report to Markdown file ===
            os.makedirs("reports", exist_ok=True)
            filename = f"reports/report_{time.strftime('%Y%m%d_%H%M%S')}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(response_text + "\n\n")
                f.write("# Guardrail Results\n")
                for k, v in guardrail_results.items():
                    f.write(f"- {k}: {v}\n")
            logging.info(f" Report saved to {filename}")

        break

    except (ClientError, ModelProviderError) as e:
        if hasattr(e, "status_code") and e.status_code == 429:
            attempt += 1
            wait_time = retry_delay * attempt
            logging.warning(f" Gemini quota exceeded. Retry {attempt}/{max_retries} in {wait_time}s...")
            time.sleep(wait_time)
        elif "429" in str(e): 
            attempt += 1
            wait_time = retry_delay * attempt
            logging.warning(f" Provider quota exceeded. Retry {attempt}/{max_retries} in {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise e
else:
    logging.error(" Pipeline failed after multiple retries. Check quotas or wait before retrying.")
