"""
tasks/transform.py
------------------
Transform task for the LGBTIQ+ compliance pipeline.

Receives extracted texts from the extract task via XCom, builds a
per-state FAISS vector store, then runs a LangGraph devil's advocate
scoring graph for each of the 6 compliance markers.

Graph flow per marker:
    retrieve → challenge → rebut → verdict

Each verdict produces a validated Pydantic record with:
    state, marker, score (0/1/2), justification, cited_article
"""

from __future__ import annotations

import logging
import os
import re
from typing import Annotated, Any, Dict, List, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, field_validator

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
import pathlib
_HERE        = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5

# ── Compliance markers ────────────────────────────────────────────────────────
# Each marker defines the retrieval query and the criteria for score=1 (partial)
# and score=2 (full).  These checklist strings are injected into every node prompt.

MARKERS: Dict[str, Dict[str, str]] = {
    "L1": {
        "name": "Non-discrimination (sexual orientation / gender identity)",
        "query": (
            "discriminación orientación sexual identidad género "
            "protección igualdad"
        ),
        "score_1": (
            "The law contains at least one explicit reference to sexual orientation "
            "OR gender identity as a protected category in non-discrimination clauses."
        ),
        "score_2": (
            "The law explicitly prohibits discrimination based on BOTH sexual "
            "orientation AND gender identity across multiple domains (employment, "
            "education, housing, or public services) with an enforcement mechanism."
        ),
    },
    "L2": {
        "name": "Legal gender-identity recognition",
        "query": (
            "identidad de género reconocimiento legal cambio nombre acta "
            "nacimiento registro civil"
        ),
        "score_1": (
            "The law provides any legal pathway for gender-identity recognition "
            "or name/gender-marker change, even if limited in scope."
        ),
        "score_2": (
            "The law provides a full administrative (non-judicial, non-medical) "
            "procedure to change the legal name and gender marker in official documents."
        ),
    },
    "L3": {
        "name": "Same-sex union or marriage rights",
        "query": (
            "matrimonio unión civil sociedad de convivencia parejas del mismo sexo "
            "derechos patrimoniales"
        ),
        "score_1": (
            "The law recognises same-sex civil unions or domestic partnerships "
            "with at least some patrimonial rights."
        ),
        "score_2": (
            "The law explicitly recognises same-sex marriage with full equal rights "
            "to opposite-sex marriage."
        ),
    },
    "C1": {
        "name": "Hate-crime / aggravated penalty provisions",
        "query": (
            "crimen de odio agravante homofobia transfobia orientación sexual "
            "identidad género delito"
        ),
        "score_1": (
            "The law lists sexual orientation or gender identity as an aggravating "
            "factor in at least one category of crime."
        ),
        "score_2": (
            "The law establishes hate-crime provisions explicitly covering both "
            "sexual orientation and gender identity across multiple crime types "
            "with enhanced penalties."
        ),
    },
    "C2": {
        "name": "Anti-discrimination enforcement mechanism",
        "query": (
            "sanción discriminación queja denuncia procedimiento comisión "
            "derechos humanos mecanismo"
        ),
        "score_1": (
            "The law establishes at least one complaint or sanction mechanism "
            "applicable to discrimination based on sexual orientation or gender identity."
        ),
        "score_2": (
            "The law establishes a specific, independent enforcement body or "
            "formal administrative procedure with binding sanctions for LGBTIQ+ "
            "discrimination."
        ),
    },
    "C3": {
        "name": "Healthcare access without discrimination",
        "query": (
            "salud atención médica discriminación orientación sexual identidad "
            "género acceso igualitario"
        ),
        "score_1": (
            "The law prohibits discrimination in healthcare settings on the basis "
            "of sexual orientation or gender identity in at least general terms."
        ),
        "score_2": (
            "The law explicitly guarantees equal healthcare access for LGBTIQ+ "
            "persons and includes specific provisions for gender-affirming care "
            "or prohibits conversion practices."
        ),
    },
}


# ── Pydantic output model ─────────────────────────────────────────────────────

class ComplianceRecord(BaseModel):
    """Validated output record for one state × marker combination."""

    state: str
    marker: str
    score: int
    justification: str
    cited_article: str

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(f"score must be 0, 1, or 2 — got {v}")
        return v


# ── LangGraph state ───────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    state_name:      str
    marker_key:      str
    marker_meta:     Dict[str, str]
    retriever:       Any                  # FAISS retriever instance
    chunks:          List[str]            # top-k chunks from initial retrieval
    challenges:      List[str]            # arguments against compliance
    rebuttals:       List[str]            # counter-evidence with article refs
    score:           int
    justification:   str
    cited_article:   str


# ── LangGraph nodes ───────────────────────────────────────────────────────────

def _build_llm() -> ChatAnthropic:
    """Instantiate the Claude model used across all nodes."""
    return ChatAnthropic(
        model="claude-sonnet-4-5",
        temperature=0,
        max_tokens=1024,
    )


def node_retrieve(state: GraphState) -> GraphState:
    """
    Retrieve the top-k most relevant chunks from the state's vector store
    using the marker's core query string.

    Populates state['chunks'] with plain text strings.
    """
    docs: List[Document] = state["retriever"].invoke(
        state["marker_meta"]["query"]
    )
    state["chunks"] = [d.page_content for d in docs]
    log.debug("[retrieve] marker=%s  chunks_found=%d", state["marker_key"], len(state["chunks"]))
    return state


def node_challenge(state: GraphState) -> GraphState:
    """
    Given the retrieved chunks, argue from a sceptical position why the law
    does NOT meet the marker criteria.  Starts from a presumption of score=0.

    Populates state['challenges'] with a list of specific objections.
    """
    llm      = _build_llm()
    meta     = state["marker_meta"]
    context  = "\n\n---\n\n".join(state["chunks"])

    system = SystemMessage(content=(
        "You are a rigorous legal analyst reviewing Mexican state laws for "
        "LGBTIQ+ rights compliance. You will argue the AGAINST position: "
        "why this law does NOT meet the required criteria. Be specific, "
        "cite gaps, vague language, or absence of provisions. "
        "Respond with a numbered list of objections only."
    ))
    human = HumanMessage(content=(
        f"Marker: {meta['name']}\n\n"
        f"Score=1 requires: {meta['score_1']}\n"
        f"Score=2 requires: {meta['score_2']}\n\n"
        f"Retrieved legal text:\n{context}\n\n"
        "List every reason why this law FAILS to meet the marker criteria."
    ))

    response = llm.invoke([system, human])
    raw      = response.content.strip()

    # Parse numbered list into individual challenge strings
    challenges = [
        line.lstrip("0123456789.- ").strip()
        for line in raw.splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]
    state["challenges"] = challenges or [raw]
    log.debug("[challenge] marker=%s  challenges=%d", state["marker_key"], len(state["challenges"]))
    return state


def node_rebut(state: GraphState) -> GraphState:
    """
    Query the vector store with all challenges combined, then make a single
    API call asking Claude to rebut each challenge in a numbered list.

    Using one call instead of one-per-challenge reduces API round-trips from
    N to 1 per marker, cutting latency and cost proportionally.

    Populates state['rebuttals'] with a list of rebuttal strings.
    """
    llm       = _build_llm()
    meta      = state["marker_meta"]
    retriever = state["retriever"]

    # Retrieve counter-evidence for all challenges in one combined query
    combined_query = " ".join(state["challenges"])
    counter_docs: List[Document] = retriever.invoke(combined_query)
    counter_context = "\n\n---\n\n".join(d.page_content for d in counter_docs)

    challenges_block = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(state["challenges"])
    )

    system = SystemMessage(content=(
        "You are a legal analyst building the FOR position. Given a list of "
        "objections and counter-evidence from the legal text, produce a "
        "numbered list of rebuttals — one paragraph per objection — each "
        "citing the specific article, section, or clause that addresses it. "
        "If no counter-evidence exists for an objection, write: "
        "'No counter-evidence found.' Keep the same numbering as the objections."
    ))
    human = HumanMessage(content=(
        f"Marker: {meta['name']}\n\n"
        f"Objections:\n{challenges_block}\n\n"
        f"Counter-evidence from legal text:\n{counter_context}\n\n"
        "Write a numbered rebuttal for each objection."
    ))

    response = llm.invoke([system, human])
    raw = response.content.strip()

    # Parse numbered list into individual rebuttal strings
    rebuttals = [
        line.lstrip("0123456789.- ").strip()
        for line in raw.splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]
    state["rebuttals"] = rebuttals or [raw]
    log.debug("[rebut] marker=%s  rebuttals=%d", state["marker_key"], len(state["rebuttals"]))
    return state


def node_verdict(state: GraphState) -> GraphState:
    """
    Compare challenges vs rebuttals and assign a final compliance score.

    Scoring rules:
        0 — No rebuttals hold; the law clearly does not meet the criteria.
        1 — Some rebuttals hold with legal text support (partial coverage).
        2 — All criteria are rebutted with specific legal text references.

    Populates state['score'], state['justification'], and state['cited_article'].
    """
    llm  = _build_llm()
    meta = state["marker_meta"]

    challenges_block = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(state["challenges"])
    )
    rebuttals_block = "\n".join(
        f"{i+1}. {r}" for i, r in enumerate(state["rebuttals"])
    )

    system = SystemMessage(content=(
        "You are a senior legal analyst rendering a final compliance verdict. "
        "Assign a score of 0, 1, or 2 strictly according to the criteria provided. "
        "Respond in this exact JSON format (no markdown):\n"
        '{"score": <0|1|2>, "justification": "<paragraph>", "cited_article": "<article ref or N/A>"}'
    ))
    human = HumanMessage(content=(
        f"Marker: {meta['name']}\n\n"
        f"Score=0 — no criteria met.\n"
        f"Score=1 — partial: {meta['score_1']}\n"
        f"Score=2 — full:    {meta['score_2']}\n\n"
        f"Challenges (arguments AGAINST):\n{challenges_block}\n\n"
        f"Rebuttals (counter-evidence FOR):\n{rebuttals_block}\n\n"
        "Render the final verdict as JSON."
    ))

    response = llm.invoke([system, human])
    raw      = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    # Parse the JSON response
    import json as _json
    try:
        parsed          = _json.loads(raw)
        state["score"]          = int(parsed.get("score", 0))
        state["justification"]  = str(parsed.get("justification", raw))
        state["cited_article"]  = str(parsed.get("cited_article", "N/A"))
    except (_json.JSONDecodeError, ValueError):
        log.warning("[verdict] Could not parse JSON for marker=%s — defaulting to score=0", state["marker_key"])
        state["score"]         = 0
        state["justification"] = raw
        state["cited_article"] = "N/A"

    log.info(
        "[verdict] state=%s  marker=%s  score=%d",
        state["state_name"], state["marker_key"], state["score"],
    )
    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    """
    Compile the LangGraph devil's advocate scoring graph.

    Flow: retrieve → challenge → rebut → verdict → END
    """
    graph = StateGraph(GraphState)
    graph.add_node("retrieve",  node_retrieve)
    graph.add_node("challenge", node_challenge)
    graph.add_node("rebut",     node_rebut)
    graph.add_node("verdict",   node_verdict)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",  "challenge")
    graph.add_edge("challenge", "rebut")
    graph.add_edge("rebut",     "verdict")
    graph.add_edge("verdict",   END)

    return graph.compile()


# ── Vector store builder ──────────────────────────────────────────────────────

def _build_vector_store(texts: List[str]) -> Any:
    """
    Chunk all extracted texts with CharacterTextSplitter and index them
    in a FAISS vector store using a local HuggingFace embedding model.

    Returns a retriever configured for TOP_K results.
    """
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n",
    )
    docs: List[Document] = []
    for text in texts:
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk))

    if not docs:
        raise ValueError("No chunks produced — extracted texts are empty.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    store = FAISS.from_documents(docs, embeddings)
    return store.as_retriever(search_kwargs={"k": TOP_K})


# ── Shared scoring logic ──────────────────────────────────────────────────────

def _score_single_state(state_name: str, file_paths: List[str]) -> List[Dict]:
    """
    Build a FAISS vector store from the given file paths and run the LangGraph
    devil's advocate graph for all 6 markers.

    Returns a list of up to 6 ComplianceRecord dicts, or empty list if no
    usable text was found.
    """
    texts = []
    for fp in file_paths:
        p = pathlib.Path(fp)
        if p.exists():
            content = p.read_text(encoding="utf-8").strip()
            if content:
                texts.append(content)
        else:
            log.warning("[%s] File not found, skipping: %s", state_name, fp)

    if not texts:
        log.warning("[%s] No usable text — all documents were empty. Skipping.", state_name)
        return []

    log.info("[%s] Building vector store from %d documents.", state_name, len(texts))
    try:
        retriever = _build_vector_store(texts)
    except ValueError as exc:
        log.error("[%s] Vector store failed: %s", state_name, exc)
        return []

    graph   = _build_graph()
    records: List[Dict] = []

    for marker_key, marker_meta in MARKERS.items():
        initial_state: GraphState = {
            "state_name":    state_name,
            "marker_key":    marker_key,
            "marker_meta":   marker_meta,
            "retriever":     retriever,
            "chunks":        [],
            "challenges":    [],
            "rebuttals":     [],
            "score":         0,
            "justification": "",
            "cited_article": "N/A",
        }

        try:
            final_state = graph.invoke(initial_state)
            record = ComplianceRecord(
                state         = state_name,
                marker        = marker_key,
                score         = final_state["score"],
                justification = final_state["justification"],
                cited_article = final_state["cited_article"],
            )
            records.append(record.model_dump())
        except Exception as exc:
            log.error("[%s] Graph failed for marker=%s: %s", state_name, marker_key, exc)
            records.append(ComplianceRecord(
                state         = state_name,
                marker        = marker_key,
                score         = 0,
                justification = f"Pipeline error: {exc}",
                cited_article = "N/A",
            ).model_dump())

    return records


# ── Airflow callables ─────────────────────────────────────────────────────────

def transform_state(state_name: str, ti) -> List[Dict]:
    """
    Airflow task callable — score a single state against all 6 markers.

    Pulls the full file-path index from XCom (task_id='extract_all_states'),
    selects the paths for this state, builds a FAISS vector store, and runs
    the LangGraph devil's advocate graph for each marker.

    Returns
    -------
    list of dict
        Up to 6 ComplianceRecord dicts (one per marker).
        Passed to load_results via XCom.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    file_index: Dict[str, List[str]] = ti.xcom_pull(task_ids="extract_all_states")
    if not file_index:
        raise ValueError("XCom returned empty file index from extract_all_states.")

    file_paths = file_index.get(state_name, [])
    if not file_paths:
        log.warning("[%s] No file paths found in XCom — returning empty.", state_name)
        return []

    records = _score_single_state(state_name, file_paths)
    log.info("[%s] transform_state complete: %d records", state_name, len(records))
    return records


def transform_to_scores(ti) -> List[Dict]:
    """
    Legacy single-task callable — scores all states sequentially.
    Kept for ad-hoc use; the main pipeline uses transform_state() per state.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    file_index: Dict[str, List[str]] = ti.xcom_pull(task_ids="extract_all_states")
    if not file_index:
        raise ValueError("XCom returned empty file index from extract_all_states.")

    records: List[Dict] = []
    for state_name, file_paths in file_index.items():
        records.extend(_score_single_state(state_name, file_paths))

    log.info("transform_to_scores complete: %d records produced.", len(records))
    return records
