"""
AI Feedback-to-Roadmap Copilot — MVP2

Upgrade from MVP1:
- LangChain Documents
- Chroma persistent vector store
- Local or OpenAI embeddings
- Evidence-grounded RAG Q&A
- Better PM visualizations

Run:
    pip install -r requirements.txt
    cp .env.example .env
    streamlit run app.py
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import hashlib
import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

APP_TITLE = "AI Feedback-to-Roadmap Copilot"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "feedback_dataset.xlsx"
CHROMA_PARENT_DIR = Path("chroma_store")
EXPORT_DIR = Path("exports")

DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "Local sentence-transformers")
DEFAULT_LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

SEVERITY_SCORE = {
    "critical": 5,
    "critical_blocker": 5,
    "high": 4,
    "strong_objection": 4,
    "medium": 3,
    "moderate": 3,
    "low": 2,
    "nice_to_have": 2,
    "unknown": 1,
}

SEGMENT_SCORE = {
    "enterprise": 5,
    "mid-market": 4,
    "commercial": 3,
    "startup": 2,
    "market": 2,
    "unknown": 1,
}

THREAT_SCORE = {
    "high": 5,
    "medium": 3,
    "low": 2,
    "unknown": 1,
}

ROADMAP_LABELS = [
    (8.0, "Build first"),
    (6.5, "Next roadmap candidate"),
    (5.0, "Validate with more discovery"),
    (0.0, "Backlog / monitor"),
]

REQUIRED_SHEETS = ["User Feedback", "Sales Notes", "Market Research"]


@dataclass
class LoadedData:
    user_feedback: pd.DataFrame
    sales_notes: pd.DataFrame
    market_research: pd.DataFrame
    normalized: pd.DataFrame
    data_hash: str


# -----------------------------
# Helpers
# -----------------------------


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalized_key(value: Any) -> str:
    return clean_text(value).lower().replace(" ", "_").replace("-", "_") or "unknown"


def normalize_label(value: Any) -> str:
    return clean_text(value) or "unknown"


def score_from_map(value: Any, mapping: Dict[str, int]) -> int:
    return mapping.get(normalized_key(value), mapping["unknown"])


def scale_1_to_10(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_v, max_v = values.min(), values.max()
    if max_v == min_v:
        return pd.Series(np.full(len(values), 5.0), index=series.index)
    return 1.0 + 9.0 * (values - min_v) / (max_v - min_v)


def roadmap_label(score: float) -> str:
    for threshold, label in ROADMAP_LABELS:
        if score >= threshold:
            return label
    return "Backlog / monitor"


def dataframe_hash(df: pd.DataFrame) -> str:
    stable_csv = df.sort_index(axis=1).to_csv(index=False)
    return hashlib.sha256(stable_csv.encode("utf-8")).hexdigest()[:16]


def as_metadata_value(value: Any) -> Any:
    """Chroma metadata values must be scalar and non-null."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# -----------------------------
# Data loading / normalization
# -----------------------------


@st.cache_data(show_spinner=False)
def read_excel_from_bytes(file_bytes: Optional[bytes]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if file_bytes:
        source: Any = io.BytesIO(file_bytes)
    else:
        source = DEFAULT_DATASET_PATH

    try:
        user_feedback = pd.read_excel(source, sheet_name="User Feedback")
        if file_bytes:
            source.seek(0)

        sales_notes = pd.read_excel(source, sheet_name="Sales Notes")
        if file_bytes:
            source.seek(0)

        market_research = pd.read_excel(source, sheet_name="Market Research")

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find default dataset at: {DEFAULT_DATASET_PATH}. "
            "Upload a workbook or check that ai_feedback_roadmap_mvp2/data/feedback_dataset.xlsx exists."
        ) from exc

    except ValueError as exc:
        raise ValueError(f"Workbook must contain sheets: {', '.join(REQUIRED_SHEETS)}") from exc

    return user_feedback, sales_notes, market_research


def normalize_user_feedback(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["record_id"] = df.get("feedback_id", pd.Series(range(len(df)))).astype(str)
    out["source_type"] = "user_feedback"
    out["source_channel"] = df.get("source_channel", "unknown")
    out["date"] = df.get("date", "")
    out["persona"] = df.get("persona", "unknown")
    out["team_or_segment"] = df.get("team", df.get("company_segment", "unknown"))
    out["company_segment"] = df.get("company_segment", "unknown")
    out["product_area"] = df.get("product_area", "unknown")
    out["theme"] = df.get("theme", "unknown")
    out["text"] = df.get("comment", "")
    out["desired_outcome"] = df.get("desired_outcome", "")
    out["pm_action_hint"] = df.get("pm_action_hint", "")
    out["severity_raw"] = df.get("severity", "unknown")
    out["frequency_signal"] = pd.to_numeric(df.get("frequency_signal", 1), errors="coerce").fillna(1)
    out["revenue_impact_usd"] = 0.0
    out["market_threat"] = "unknown"
    out["competitor_mentioned"] = ""
    out["evidence_source"] = out["record_id"].map(lambda x: f"User feedback {x}")
    return out


def normalize_sales_notes(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["record_id"] = df.get("note_id", pd.Series(range(len(df)))).astype(str)
    out["source_type"] = "sales_note"
    out["source_channel"] = df.get("sales_role", "sales")
    out["date"] = df.get("date", "")
    out["persona"] = df.get("sales_role", "customer-facing team")
    out["team_or_segment"] = df.get("account_segment", "unknown")
    out["company_segment"] = df.get("account_segment", "unknown")
    out["product_area"] = df.get("product_area", "unknown")
    out["theme"] = df.get("theme", "unknown")
    out["text"] = df.get("note", "")
    out["desired_outcome"] = df.get("requested_next_step", "")
    out["pm_action_hint"] = df.get("pm_action_hint", "")
    out["severity_raw"] = df.get("blocker_level", "unknown")
    out["frequency_signal"] = 1.0
    out["revenue_impact_usd"] = pd.to_numeric(df.get("arr_impact_usd", 0), errors="coerce").fillna(0.0)
    out["market_threat"] = "unknown"
    out["competitor_mentioned"] = df.get("competitor_mentioned", "")
    out["evidence_source"] = out["record_id"].map(lambda x: f"Sales note {x}")
    return out


def normalize_market_research(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["record_id"] = df.get("snippet_id", pd.Series(range(len(df)))).astype(str)
    out["source_type"] = "market_research"
    out["source_channel"] = df.get("source_type", "market")
    out["date"] = df.get("date", "")
    out["persona"] = "market / competitor signal"
    out["team_or_segment"] = df.get("competitor_or_market", "unknown")
    out["company_segment"] = "market"
    out["product_area"] = df.get("product_area", "unknown")
    out["theme"] = df.get("theme", "unknown")
    out["text"] = df.get("snippet", "")
    out["desired_outcome"] = df.get("opportunity", df.get("implication", ""))
    out["pm_action_hint"] = df.get("implication", "")
    out["severity_raw"] = df.get("threat_level", "unknown")
    out["frequency_signal"] = 1.0
    out["revenue_impact_usd"] = 0.0
    out["market_threat"] = df.get("threat_level", "unknown")
    out["competitor_mentioned"] = df.get("competitor_or_market", "")
    out["evidence_source"] = out["record_id"].map(lambda x: f"Market research {x}")
    return out


def load_and_normalize(file_bytes: Optional[bytes]) -> LoadedData:
    user_feedback, sales_notes, market_research = read_excel_from_bytes(file_bytes)

    normalized = pd.concat(
        [
            normalize_user_feedback(user_feedback),
            normalize_sales_notes(sales_notes),
            normalize_market_research(market_research),
        ],
        ignore_index=True,
    )

    for col in [
        "source_type",
        "source_channel",
        "date",
        "persona",
        "team_or_segment",
        "company_segment",
        "product_area",
        "theme",
        "text",
        "desired_outcome",
        "pm_action_hint",
        "severity_raw",
        "market_threat",
        "competitor_mentioned",
        "evidence_source",
    ]:
        normalized[col] = normalized[col].fillna("").astype(str).map(lambda x: x.strip() or "unknown")

    normalized["severity_score"] = normalized["severity_raw"].apply(lambda x: score_from_map(x, SEVERITY_SCORE))
    normalized["segment_score"] = normalized["company_segment"].apply(lambda x: score_from_map(x, SEGMENT_SCORE))
    normalized["threat_score"] = normalized["market_threat"].apply(lambda x: score_from_map(x, THREAT_SCORE))
    normalized["revenue_scaled"] = scale_1_to_10(normalized["revenue_impact_usd"])

    normalized["business_impact_score"] = np.where(
        normalized["source_type"].eq("sales_note"),
        normalized["revenue_scaled"],
        np.where(
            normalized["source_type"].eq("market_research"),
            normalized["threat_score"] * 2,
            normalized["severity_score"] + normalized["segment_score"],
        ),
    )

    normalized["evidence_text"] = normalized.apply(
        lambda r: f"[{r['evidence_source']}] {r['text']}", axis=1
    )
    normalized["doc_id"] = normalized.apply(
        lambda r: f"{r['source_type']}::{r['record_id']}", axis=1
    )
    normalized["rag_content"] = normalized.apply(
        lambda r: (
            f"Evidence source: {r['evidence_source']}\n"
            f"Source type: {r['source_type']}\n"
            f"Channel: {r['source_channel']}\n"
            f"Persona: {r['persona']}\n"
            f"Segment: {r['company_segment']}\n"
            f"Product area: {r['product_area']}\n"
            f"Theme: {r['theme']}\n"
            f"Severity: {r['severity_raw']}\n"
            f"Revenue impact USD: {r['revenue_impact_usd']}\n"
            f"Competitor mentioned: {r['competitor_mentioned']}\n"
            f"Feedback: {r['text']}\n"
            f"Desired outcome / implication: {r['desired_outcome']}\n"
            f"PM action hint: {r['pm_action_hint']}"
        ),
        axis=1,
    )

    data_hash = dataframe_hash(normalized[["doc_id", "rag_content", "theme", "source_type"]])
    return LoadedData(user_feedback, sales_notes, market_research, normalized, data_hash)


# -----------------------------
# Product analytics
# -----------------------------


def top_evidence(group: pd.DataFrame, max_items: int = 4) -> List[str]:
    ranked = group.sort_values(
        ["severity_score", "business_impact_score", "frequency_signal"],
        ascending=False,
    )
    return ranked["evidence_text"].head(max_items).tolist()


def persona_impact(group: pd.DataFrame) -> str:
    counts = group["persona"].value_counts().head(4)
    return ", ".join([f"{persona} ({count})" for persona, count in counts.items()])


def source_mix(group: pd.DataFrame) -> str:
    counts = group["source_type"].value_counts()
    return ", ".join([f"{src}: {count}" for src, count in counts.items()])


def compute_theme_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    max_frequency = max(float(df["frequency_signal"].sum()), 1.0)

    for theme, group in df.groupby("theme", dropna=False):
        mentions = len(group)
        weighted_frequency = float(group["frequency_signal"].sum())
        frequency_score = 10.0 * weighted_frequency / max_frequency
        avg_severity = float(group["severity_score"].mean())
        avg_business = float(group["business_impact_score"].mean())
        revenue = float(group["revenue_impact_usd"].sum())
        source_count = int(group["source_type"].nunique())
        confidence = min(10.0, mentions / 2.0 + source_count * 1.5)
        strategic_fit = 8.0 if any(
            token in str(theme).lower()
            for token in ["timeline", "root", "alert", "triage", "postmortem", "workflow"]
        ) else 6.0

        priority_score = (
            frequency_score * 0.25
            + (avg_severity * 2.0) * 0.25
            + avg_business * 0.20
            + strategic_fit * 0.15
            + confidence * 0.10
            + min(10.0, revenue / 25000.0) * 0.05
        )

        rows.append(
            {
                "theme": theme,
                "mentions": mentions,
                "weighted_frequency": round(weighted_frequency, 2),
                "avg_severity": round(avg_severity, 2),
                "avg_business_impact": round(avg_business, 2),
                "revenue_signal_usd": round(revenue, 0),
                "confidence": round(confidence, 2),
                "strategic_fit": round(strategic_fit, 2),
                "priority_score": round(priority_score, 2),
                "roadmap_recommendation": roadmap_label(priority_score),
                "persona_impact": persona_impact(group),
                "source_mix": source_mix(group),
                "evidence": top_evidence(group),
            }
        )

    return pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)


def make_theme_brief(theme_row: pd.Series) -> str:
    evidence = "\n".join([f"- {item}" for item in theme_row["evidence"]])
    return f"""### {theme_row['theme']}

**Roadmap recommendation:** {theme_row['roadmap_recommendation']}  
**Priority score:** {theme_row['priority_score']} / 10  
**Mentions:** {theme_row['mentions']}  
**Persona impact:** {theme_row['persona_impact']}  
**Revenue signal:** ${theme_row['revenue_signal_usd']:,.0f}

**Evidence-backed summary:**  
This theme appears across {theme_row['source_mix']}. The strongest signal is a combination of user pain, business impact, and cross-source confidence. The PM should validate whether solving this theme reduces triage time, lowers incident coordination overhead, or improves enterprise buying confidence.

**Representative evidence:**
{evidence}
"""


def deterministic_prd(theme_row: pd.Series) -> str:
    evidence = "\n".join([f"- {item}" for item in theme_row["evidence"]])
    theme = theme_row["theme"]
    return f"""# PRD: {theme}

## 1. Problem
Developer incident management teams are reporting repeated pain around **{theme}**. The evidence shows this is not an isolated request: it appears across {theme_row['source_mix']} and impacts {theme_row['persona_impact']}.

## 2. Target users
- Primary: SREs, DevOps engineers, on-call engineers
- Secondary: Engineering managers, platform teams, customer-facing teams
- Buyer / stakeholder: VP Engineering, Head of Platform, Security/Compliance leaders for enterprise accounts

## 3. Evidence
{evidence}

## 4. Product opportunity
Build an evidence-backed workflow that helps incident teams reduce ambiguity during incidents and convert noisy operational signals into faster decisions.

## 5. MVP requirements
1. Ingest relevant incident signals from alerts, deployment events, Slack/PagerDuty/Jira notes, and customer/support feedback.
2. Retrieve evidence related to a selected incident theme or PM question.
3. Generate a concise summary with citations to the underlying evidence.
4. Show affected personas, source mix, severity, and business impact.
5. Allow the PM or incident owner to accept, reject, or edit recommendations.

## 6. Non-goals
- No autonomous remediation in MVP.
- No automatic production rollback.
- No unsupported root-cause claims without evidence.
- No use of confidential customer data in demos or portfolio material.

## 7. Success metrics
- Reduce time to synthesize feedback themes by 50%.
- Increase PM confidence in roadmap decisions through evidence-backed recommendations.
- Improve PRD draft quality as rated by PM/engineering reviewers.
- Keep hallucination rate below 5% in evidence-grounded outputs.
- Increase percentage of roadmap items linked to customer/sales/market evidence.

## 8. AI evaluation plan
- Create a gold set of 30-50 manually labeled feedback records.
- Measure retrieval precision: does the system retrieve the right evidence for the PM question?
- Measure answer groundedness: does every claim map to retrieved evidence?
- Measure hallucination rate: did the model invent unsupported claims?
- Collect human usefulness score from PM/engineering reviewers.

## 9. Launch plan
- Pilot with 2-3 platform/SRE PMs or senior engineers.
- Review generated summaries and PRDs weekly.
- Compare AI-assisted roadmap synthesis against manual synthesis.
- Expand data sources only after evidence quality is trusted.

## 10. Open questions
- Which integrations matter most for MVP: Slack, PagerDuty, Jira, GitHub, Datadog, or deploy logs?
- Which persona gets the first workflow: on-call engineer, PM, EM, or Sales Engineer?
- How should confidence be displayed when retrieved evidence is weak?
"""


# -----------------------------
# LangChain + Chroma RAG
# -----------------------------


def build_documents(df: pd.DataFrame) -> List[Document]:
    documents: List[Document] = []
    for _, row in df.iterrows():
        metadata = {
            "doc_id": row["doc_id"],
            "record_id": row["record_id"],
            "source_type": row["source_type"],
            "source_channel": row["source_channel"],
            "persona": row["persona"],
            "company_segment": row["company_segment"],
            "product_area": row["product_area"],
            "theme": row["theme"],
            "severity_raw": row["severity_raw"],
            "severity_score": int(row["severity_score"]),
            "business_impact_score": float(row["business_impact_score"]),
            "revenue_impact_usd": float(row["revenue_impact_usd"]),
            "evidence_source": row["evidence_source"],
        }
        metadata = {k: as_metadata_value(v) for k, v in metadata.items()}
        documents.append(Document(page_content=row["rag_content"], metadata=metadata))
    return documents


def get_embedding_id(embedding_provider: str, embedding_model: str) -> str:
    raw_id = f"{embedding_provider}:{embedding_model}".encode("utf-8")
    return hashlib.sha256(raw_id).hexdigest()[:10]


def get_chroma_dir(data_hash: str, embedding_provider: str, embedding_model: str) -> Path:
    embedding_id = get_embedding_id(embedding_provider, embedding_model)
    return CHROMA_PARENT_DIR / f"feedback_{data_hash}_{embedding_id}"


def get_embedding_function(embedding_provider: str, embedding_model: str) -> Embeddings:
    if embedding_provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or your shell environment.")
        return OpenAIEmbeddings(model=embedding_model)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=embedding_model)


def build_or_load_vectorstore(
    df: pd.DataFrame,
    data_hash: str,
    embedding_provider: str,
    embedding_model: str,
    rebuild: bool = False,
) -> Chroma:
    persist_dir = get_chroma_dir(data_hash, embedding_provider, embedding_model)
    collection_name = f"feedback_roadmap_{data_hash}_{get_embedding_id(embedding_provider, embedding_model)}"
    embeddings = get_embedding_function(embedding_provider, embedding_model)

    if rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)

    if persist_dir.exists() and any(persist_dir.iterdir()):
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    persist_dir.mkdir(parents=True, exist_ok=True)
    documents = build_documents(df)
    ids = [doc.metadata["doc_id"] for doc in documents]
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )


def ai_failure_message(exc: Exception) -> str:
    error_text = str(exc)
    normalized_error = error_text.lower()

    if "insufficient_quota" in normalized_error or "exceeded your current quota" in normalized_error:
        return (
            "OpenAI is configured, but the current project or account has no available quota. "
            "Check your OpenAI billing and usage limits, or switch to an API key with quota. "
            "Analytics tabs still work; RAG and AI PRD generation are disabled until quota is available."
        )
    if "openai_api_key is not set" in normalized_error or "api key" in normalized_error:
        return (
            "OPENAI_API_KEY is missing. Add it to `.env` or your shell environment, then restart Streamlit. "
            "Analytics tabs still work; RAG generation is disabled."
        )
    if "rate limit" in normalized_error or "error code: 429" in normalized_error:
        return (
            "OpenAI rate-limited this request. Wait a bit and try again, or use a project/key with higher limits. "
            "Analytics tabs still work while AI generation is unavailable."
        )
    if "langchain_huggingface" in normalized_error or "sentence_transformers" in normalized_error:
        return (
            "Local embeddings need the Hugging Face embedding dependencies. Run `pip install -r requirements.txt`, "
            "then restart Streamlit. The local model may download once on first use."
        )
    return f"AI retrieval or generation failed. Analytics tabs still work. Details: {error_text}"


def format_docs_for_prompt(docs: Iterable[Document]) -> str:
    chunks = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        chunks.append(
            f"Evidence {idx}\n"
            f"Record: {meta.get('evidence_source', '')}\n"
            f"Source: {meta.get('source_type', '')}\n"
            f"Persona: {meta.get('persona', '')}\n"
            f"Theme: {meta.get('theme', '')}\n"
            f"Product area: {meta.get('product_area', '')}\n"
            f"Revenue impact: {meta.get('revenue_impact_usd', 0)}\n"
            f"Content:\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(chunks)


def retrieve_rag_docs(
    vectorstore: Chroma,
    question: str,
    top_k: int = 8,
    source_filter: Optional[str] = None,
) -> List[Document]:
    search_kwargs: Dict[str, Any] = {"k": top_k}
    if source_filter and source_filter != "All":
        search_kwargs["filter"] = {"source_type": source_filter}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(question)


def deterministic_rag_answer(question: str, docs: List[Document]) -> str:
    if not docs:
        return "No relevant evidence was retrieved for this question."

    evidence_lines = []
    for idx, doc in enumerate(docs[:5], start=1):
        meta = doc.metadata
        evidence_lines.append(
            f"- Evidence {idx}: {meta.get('theme', 'Unknown theme')} from "
            f"{meta.get('evidence_source', 'unknown source')} "
            f"({meta.get('persona', 'unknown persona')}, {meta.get('product_area', 'unknown area')})"
        )

    return (
        "OpenAI answer generation is unavailable, so here is a retrieval-only summary.\n\n"
        f"Question: {question}\n\n"
        "Most relevant evidence signals:\n"
        + "\n".join(evidence_lines)
        + "\n\nReview the retrieved evidence below before making a roadmap decision."
    )


def generate_answer_from_docs(question: str, docs: List[Document], chat_model: str) -> str:
    context = format_docs_for_prompt(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI Product Manager assistant. Answer using only the evidence provided. "
                "Do not invent facts. If evidence is weak, say so. Return practical roadmap-oriented guidance.",
            ),
            (
                "user",
                "Question: {question}\n\nEvidence base:\n{context}\n\n"
                "Return:\n"
                "1. Direct answer\n"
                "2. Supporting evidence bullets\n"
                "3. PM recommendation\n"
                "4. Risks or unknowns",
            ),
        ]
    )
    llm = ChatOpenAI(model=chat_model, temperature=0.2)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


def answer_with_rag(
    vectorstore: Chroma,
    question: str,
    chat_model: str,
    top_k: int = 8,
    source_filter: Optional[str] = None,
) -> Tuple[str, List[Document]]:
    docs = retrieve_rag_docs(vectorstore, question, top_k=top_k, source_filter=source_filter)
    answer = generate_answer_from_docs(question, docs, chat_model)
    return answer, docs


def ai_prd_from_evidence(
    theme_row: pd.Series,
    vectorstore: Chroma,
    chat_model: str,
    top_k: int = 10,
) -> str:
    query = (
        f"Build a product requirements document for the product opportunity: {theme_row['theme']}. "
        "Focus on developer incident management, SRE, DevOps, enterprise buyer impact, evidence, metrics, and risks."
    )
    docs = vectorstore.as_retriever(search_kwargs={"k": top_k}).invoke(query)
    context = format_docs_for_prompt(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior AI Product Manager. Create a concise PRD grounded only in supplied evidence. "
                "Use PM language. Do not invent customer quotes or unsupported market claims.",
            ),
            (
                "user",
                "Theme: {theme}\nPriority score: {priority}\nRoadmap recommendation: {recommendation}\n\n"
                "Evidence:\n{context}\n\n"
                "Write a PRD with sections: Problem, Users, Evidence, MVP, Non-goals, Success metrics, "
                "AI evaluation, Risks, Launch plan, Open questions.",
            ),
        ]
    )
    llm = ChatOpenAI(model=chat_model, temperature=0.2)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "theme": theme_row["theme"],
            "priority": theme_row["priority_score"],
            "recommendation": theme_row["roadmap_recommendation"],
            "context": context,
        }
    )


# -----------------------------
# UI components
# -----------------------------


def render_evidence_cards(records: pd.DataFrame, limit: int = 5) -> None:
    for _, row in records.head(limit).iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['evidence_source']}** · `{row['source_type']}` · **{row['theme']}**")
            st.write(row["text"])
            cols = st.columns(4)
            cols[0].caption(f"Persona: {row['persona']}")
            cols[1].caption(f"Segment: {row['company_segment']}")
            cols[2].caption(f"Severity: {row['severity_raw']}")
            cols[3].caption(f"Revenue: ${float(row['revenue_impact_usd']):,.0f}")


def render_doc_cards(docs: List[Document]) -> None:
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        with st.expander(
            f"Evidence {idx}: {meta.get('evidence_source', '')} · {meta.get('theme', '')}",
            expanded=idx <= 3,
        ):
            st.caption(
                f"Source: {meta.get('source_type', '')} | Persona: {meta.get('persona', '')} | "
                f"Product area: {meta.get('product_area', '')} | Revenue: ${float(meta.get('revenue_impact_usd', 0)):,.0f}"
            )
            st.text(doc.page_content)


def filter_normalized(df: pd.DataFrame, source_filter: str, theme_filter: str, persona_filter: str) -> pd.DataFrame:
    out = df.copy()
    if source_filter != "All":
        out = out[out["source_type"] == source_filter]
    if theme_filter != "All":
        out = out[out["theme"] == theme_filter]
    if persona_filter != "All":
        out = out[out["persona"] == persona_filter]
    return out


def render_sidebar(loaded: LoadedData) -> Tuple[str, str, str, int, str, str, str]:
    st.sidebar.header("Controls")
    st.sidebar.caption(f"Dataset hash: `{loaded.data_hash}`")

    source_options = ["All"] + sorted(loaded.normalized["source_type"].unique().tolist())
    theme_options = ["All"] + sorted(loaded.normalized["theme"].unique().tolist())
    persona_options = ["All"] + sorted(loaded.normalized["persona"].unique().tolist())

    source_filter = st.sidebar.selectbox("Source filter", source_options)
    theme_filter = st.sidebar.selectbox("Theme filter", theme_options)
    persona_filter = st.sidebar.selectbox("Persona filter", persona_options)
    top_k = st.sidebar.slider("RAG top-k evidence", 3, 15, 8)

    embedding_provider = st.sidebar.selectbox(
        "Embedding provider",
        ["Local sentence-transformers", "OpenAI"],
        index=0 if DEFAULT_EMBEDDING_PROVIDER != "OpenAI" else 1,
    )
    default_embedding_model = (
        DEFAULT_OPENAI_EMBEDDING_MODEL
        if embedding_provider == "OpenAI"
        else DEFAULT_LOCAL_EMBEDDING_MODEL
    )
    embedding_model = st.sidebar.text_input("Embedding model", default_embedding_model)
    chat_model = st.sidebar.text_input("Chat model", DEFAULT_CHAT_MODEL)

    api_ready = bool(os.getenv("OPENAI_API_KEY", "").strip())
    if api_ready:
        st.sidebar.success("OPENAI_API_KEY detected for AI text generation")
    else:
        st.sidebar.warning("OPENAI_API_KEY missing. Local retrieval works; AI text generation is disabled.")

    return source_filter, theme_filter, persona_filter, top_k, embedding_provider, embedding_model, chat_model


# -----------------------------
# Main app
# -----------------------------


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("MVP2: LangChain + Chroma + local/OpenAI embeddings for evidence-backed AI PM decisions")

    uploaded_file = st.sidebar.file_uploader("Upload feedback workbook", type=["xlsx"])
    file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None

    try:
        loaded = load_and_normalize(file_bytes)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    source_filter, theme_filter, persona_filter, top_k, embedding_provider, embedding_model, chat_model = render_sidebar(loaded)
    filtered_df = filter_normalized(loaded.normalized, source_filter, theme_filter, persona_filter)
    theme_summary = compute_theme_summary(filtered_df if len(filtered_df) else loaded.normalized)
    top10 = theme_summary.head(10).copy()

    tab_overview, tab_roadmap, tab_rag, tab_prd, tab_explorer, tab_method = st.tabs(
        [
            "Overview",
            "Roadmap Priorities",
            "RAG Evidence Q&A",
            "Draft PRD",
            "Evidence Explorer",
            "Methodology",
        ]
    )

    with tab_overview:
        st.subheader("Dataset overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total records", len(filtered_df))
        c2.metric("User feedback", int((filtered_df["source_type"] == "user_feedback").sum()))
        c3.metric("Sales notes", int((filtered_df["source_type"] == "sales_note").sum()))
        c4.metric("Market snippets", int((filtered_df["source_type"] == "market_research").sum()))
        c5.metric("Themes", filtered_df["theme"].nunique())

        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown("#### Top themes by roadmap priority")
            fig = px.bar(
                top10.sort_values("priority_score"),
                x="priority_score",
                y="theme",
                orientation="h",
                hover_data=["mentions", "avg_severity", "avg_business_impact", "roadmap_recommendation"],
                labels={"priority_score": "Priority score", "theme": "Theme"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### Opportunity map")
            fig2 = px.scatter(
                top10,
                x="weighted_frequency",
                y="avg_business_impact",
                size="mentions",
                hover_name="theme",
                hover_data=["avg_severity", "revenue_signal_usd", "roadmap_recommendation"],
                labels={
                    "weighted_frequency": "Weighted frequency",
                    "avg_business_impact": "Business impact",
                },
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Source mix by theme")
        source_mix_df = (
            filtered_df.groupby(["theme", "source_type"]).size().reset_index(name="count")
        )
        source_mix_df = source_mix_df[source_mix_df["theme"].isin(top10["theme"])]
        fig3 = px.bar(
            source_mix_df,
            x="theme",
            y="count",
            color="source_type",
            barmode="stack",
            labels={"theme": "Theme", "count": "Records", "source_type": "Source"},
        )
        fig3.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig3, use_container_width=True)

    with tab_roadmap:
        st.subheader("Top 10 roadmap priorities")
        display_cols = [
            "theme",
            "priority_score",
            "roadmap_recommendation",
            "mentions",
            "weighted_frequency",
            "avg_severity",
            "avg_business_impact",
            "revenue_signal_usd",
            "confidence",
            "persona_impact",
            "source_mix",
        ]
        st.dataframe(top10[display_cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Download roadmap priorities CSV",
            data=top10[display_cols].to_csv(index=False),
            file_name="roadmap_priorities.csv",
            mime="text/csv",
        )

        selected_theme = st.selectbox("Inspect theme", top10["theme"].tolist())
        selected_row = top10[top10["theme"] == selected_theme].iloc[0]
        st.markdown(make_theme_brief(selected_row))

        st.markdown("#### Evidence records for selected theme")
        records = filtered_df[filtered_df["theme"] == selected_theme].sort_values(
            ["severity_score", "business_impact_score", "frequency_signal"], ascending=False
        )
        render_evidence_cards(records, limit=8)

    with tab_rag:
        st.subheader("Ask the evidence base")
        st.write(
            "This is the true RAG loop: semantic retrieval from Chroma using local or OpenAI embeddings, "
            "then optional LLM answer generation grounded in the retrieved evidence."
        )

        question = st.text_area(
            "PM question",
            value="What should we build first for enterprise SRE and DevOps teams, and why?",
            height=90,
        )
        rebuild = st.checkbox("Force rebuild Chroma index", value=False)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            run_rag = st.button("Run RAG answer", type="primary")
        with col_b:
            st.caption(
                "First run will create embeddings for the 180 feedback records and persist them locally in `chroma_store/`. "
                "Local sentence-transformers may download the model once, then reuse it from cache."
            )

        if run_rag:
            try:
                with st.spinner("Building/loading Chroma and retrieving evidence..."):
                    vectorstore = build_or_load_vectorstore(
                        loaded.normalized,
                        loaded.data_hash,
                        embedding_provider=embedding_provider,
                        embedding_model=embedding_model,
                        rebuild=rebuild,
                    )
                    docs = retrieve_rag_docs(
                        vectorstore,
                        question,
                        top_k=top_k,
                        source_filter=source_filter,
                    )
                    if os.getenv("OPENAI_API_KEY", "").strip():
                        try:
                            rag_answer = generate_answer_from_docs(question, docs, chat_model)
                        except Exception as exc:
                            st.warning(ai_failure_message(exc))
                            rag_answer = deterministic_rag_answer(question, docs)
                    else:
                        rag_answer = deterministic_rag_answer(question, docs)
                st.markdown("### Answer")
                st.markdown(rag_answer)
                st.markdown("### Retrieved evidence")
                render_doc_cards(docs)
            except Exception as exc:
                st.error(ai_failure_message(exc))

    with tab_prd:
        st.subheader("Draft PRD for top opportunity")
        prd_theme = st.selectbox("Select opportunity", top10["theme"].tolist(), key="prd_theme")
        prd_row = top10[top10["theme"] == prd_theme].iloc[0]
        st.markdown("#### Deterministic PRD draft")
        st.markdown(deterministic_prd(prd_row))

        st.divider()
        st.markdown("#### Optional AI-enhanced PRD")
        if st.button("Generate AI-enhanced PRD from Chroma evidence"):
            try:
                with st.spinner("Retrieving evidence and generating PRD..."):
                    vectorstore = build_or_load_vectorstore(
                        loaded.normalized,
                        loaded.data_hash,
                        embedding_provider=embedding_provider,
                        embedding_model=embedding_model,
                        rebuild=False,
                    )
                    ai_prd = ai_prd_from_evidence(prd_row, vectorstore, chat_model=chat_model, top_k=top_k)
                st.markdown(ai_prd)
            except Exception as exc:
                st.error(ai_failure_message(exc))

    with tab_explorer:
        st.subheader("Evidence explorer")
        keyword = st.text_input("Keyword filter", "")
        explorer_df = filtered_df.copy()
        if keyword.strip():
            pattern = keyword.strip().lower()
            explorer_df = explorer_df[
                explorer_df["rag_content"].str.lower().str.contains(pattern, regex=False)
            ]
        st.caption(f"Showing {len(explorer_df)} records")
        st.dataframe(
            explorer_df[
                [
                    "evidence_source",
                    "source_type",
                    "persona",
                    "company_segment",
                    "product_area",
                    "theme",
                    "severity_raw",
                    "business_impact_score",
                    "revenue_impact_usd",
                    "text",
                    "desired_outcome",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tab_method:
        st.subheader("Methodology")
        st.markdown(
            """
## MVP2 architecture

```text
Excel feedback data
    ↓
Normalize user feedback + sales notes + market research
    ↓
Create LangChain Documents with metadata
    ↓
Local sentence-transformers or OpenAIEmbeddings
    ↓
Persistent Chroma vector database
    ↓
Retriever gets top-k relevant evidence
    ↓
ChatOpenAI generates answer using only retrieved evidence, or app shows retrieval-only evidence when no API key is available
```

## Roadmap scoring

```text
Priority Score =
  frequency × 25%
+ severity × 25%
+ business impact × 20%
+ strategic fit × 15%
+ confidence × 10%
+ revenue signal × 5%
```

## Why this is PM-relevant

The app does not just summarize. It helps a PM defend roadmap choices with:

- user pain frequency
- sales/revenue signals
- market/competitive signal
- persona impact
- cited evidence
- AI evaluation plan
- draft PRD

## Limitations

- Theme taxonomy is still mostly supplied by the dataset. A future MVP3 should add automatic theme discovery.
- Scoring weights are illustrative and should be tuned with PM/leadership input.
- This is a demo system. Do not use confidential company data.
"""
        )


if __name__ == "__main__":
    main()
