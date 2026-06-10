import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path("C:\\Users\\aemekkawi\\Documents\\rag_results")
OUTPUT_DIR = BASE_DIR / "evaluation_results"

EXPECTED_REPS = set(range(1, 91))
EXPECTED_MODELS = 12
EXPECTED_GENES = 16
EXPECTED_ROWS = len(EXPECTED_REPS) * EXPECTED_MODELS * EXPECTED_GENES

JUDGE_MODEL = "gpt-4o-mini"

CHROMA_DIR = BASE_DIR / "chroma_data" / "pubmed_trd_mdd"
COLLECTION_NAME = "pubmed_trd_mdd"

DEBUG_GENE = None
DEBUG_REP = None
DEBUG_MODELS = None
DEBUG_LOG_PATH = OUTPUT_DIR / "debug_fidelity_trace.txt"

ENABLE_LLM_JUDGE = False

load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_api_key) if _api_key else None

