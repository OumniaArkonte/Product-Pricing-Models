from agno.agent import Agent
from agno.team.team import Team
from agno.db.sqlite import SqliteDb
from agno.models.google import Gemini
from agno.models.mistral import MistralChat
from agno.tools.reasoning import ReasoningTools
from agno.tools import tool
from pyairtable import Table
import os, re, time, logging
from dotenv import load_dotenv
from agno.exceptions import ModelProviderError

# === Logging setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# === Load environment variables ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
BASE_ID = os.getenv("BASE_ID")
TABLE_NAME = os.getenv("TABLE_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not GOOGLE_API_KEY or not AIRTABLE_API_KEY or not BASE_ID or not TABLE_NAME:
    raise RuntimeError("Missing environment variables.")

# === DB for memory + sessions ===
db = SqliteDb(db_file="real_estate_agents_secure.db")

# === PII Masking ===
def mask_pii(text: str) -> str:
    """Masque emails, numéros de téléphone, et noms simples."""
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{10,15}\b', '[PHONE]', text)
    return text

# === Airtable tool ===
@tool
def get_all_properties_from_airtable() -> list:
    """Fetch records from Airtable and mask PII."""
    table = Table(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)
    records = table.all()
    result = []
    for rec in records:
        fields = {k: mask_pii(str(v)) for k, v in rec['fields'].items()}
        result.append(fields)
    logging.info(f"Fetched {len(result)} properties from Airtable")
    return result

# === Models ===
mistral_model = MistralChat(id="mistral-large-latest", api_key=MISTRAL_API_KEY)
gemini_model = Gemini(id="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

# === Fallback Wrapper ===

class FallbackModel:
    def __init__(self, primary, fallback, name="FallbackModel"):
        self.primary = primary
        self.fallback = fallback
        self.active_model = primary  
        self.name = name
        self.id = f"{primary.id}-fallback-{fallback.id}"
        self.provider = getattr(primary, "provider", "unknown")

    def response(self, *args, **kwargs):
        from agno.exceptions import ModelProviderError
        try:
            self.active_model = self.primary
            return self.primary.response(*args, **kwargs)
        except ModelProviderError as e:
            if "429" in str(e):
                logging.warning(f" {self.primary.id} quota exceeded. Falling back to {self.fallback.id}")
                self.active_model = self.fallback
                self.provider = getattr(self.fallback, "provider", "unknown")
                return self.fallback.response(*args, **kwargs)
            raise

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "active_model": getattr(self.active_model, "id", str(self.active_model)),
            "provider": self.provider
        }

    def __getattr__(self, item):
        """Délègue tout ce qui n’est pas défini à active_model (Mistral ou Gemini)."""
        return getattr(self.active_model, item)




# === Use fallback for agents ===
fallback_model = FallbackModel(mistral_model, gemini_model)

data_agent = Agent(
    name="DataAgent",
    role="Collect property data from Airtable",
    model=fallback_model,
    tools=[get_all_properties_from_airtable],
    db=db
)

pricing_agent = Agent(
    name="PricingAgent",
    role="Predict real estate prices based on property features",
    model=fallback_model,
    db=db
)

analysis_agent = Agent(
    name="AnalysisAgent",
    role="Analyze property prices and trends",
    model=fallback_model,
    db=db
)

reporting_agent = Agent(
    name="ReportingAgent",
    role="Generate final client report",
    model=fallback_model,
    db=db
)

benchmark_agent = Agent(
    name="BenchmarkAgent",
    role="Compare predicted prices with market trends",
    model=fallback_model,
    db=db
)

# === Guardrail Agents ===

offtopic_agent = Agent(
    name="OffTopicAgent",
    role="Detect if a query is off-topic (not related to real estate pricing, market analysis, or client reporting).",
    model=gemini_model,
    db=db
)

hallucination_agent = Agent(
    name="HallucinationAgent",
    role="Check the generated report for hallucinations or irrelevant fabricated content.",
    model=gemini_model,
    db=db
)

pii_detector_agent = Agent(
    name="PIIDetectorAgent",
    role="Ensure no Personally Identifiable Information (PII) leaks into the final report (emails, phone numbers, names).",
    model=gemini_model,
    db=db
)


# === Team (Gemini orchestrator) ===
real_estate_team = Team(
    name="RealEstatePricingTeamSecure",
    model=gemini_model,  
    members=[
        data_agent,
        pricing_agent,
        analysis_agent,
        reporting_agent,
        benchmark_agent,
        offtopic_agent,       
        hallucination_agent,  
        pii_detector_agent    
    ],
    tools=[ReasoningTools(add_instructions=True)],
    instructions=[
        "Always include a 'title' field when calling analyze().",
        "Produce a client-ready real estate report with masked PII, predictions, insights, and recommendations.",
        "Ensure responses are on-topic, hallucination-free, and without PII leaks."
    ],
    db=db,
    markdown=True
)

