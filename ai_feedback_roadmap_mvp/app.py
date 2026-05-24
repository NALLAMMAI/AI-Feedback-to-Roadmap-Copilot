"""
AI Feedback-to-Roadmap Copilot — MVP v1

Goal:
Turn user feedback, sales notes, and market research into evidence-backed
roadmap priorities and a draft PRD for a developer incident management platform.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Optional:
    export OPENAI_API_KEY="..."
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

DEFAULT_DATASET_PATH = "data/feedback_dataset.xlsx"

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

SEGMENT_WEIGHT = {
    "enterprise": 5,
    "mid-market": 4,
    "commercial": 3,
    "startup": 2,
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


@dataclass
class LoadedData:
    user_feedback: pd.DataFrame
    sales_notes: pd.DataFrame
    market_research: pd.DataFrame
    normalized: pd.DataFrame


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_severity(value: object) -> int:
    text = clean_text(value).lower().replace(" ", "_")
    return SEVERITY_SCORE.get(text, SEVERITY_SCORE["unknown"])


def normalize_segment(value: object) -> int:
    text = clean_text(value).lower()
    return SEGMENT_WEIGHT.get(text, SEGMENT_WEIGHT["unknown"])


def normalize_threat(value: object) -> int:
    text = clean_text(value).lower()
    return THREAT_SCORE.get(text, THREAT_SCORE["unknown"])


def scale_1_to_10(series: pd.Series) -> pd.Series:
    """Min-max scale numeric values to 1..10, safely."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_v, max_v = values.min(), values.max()
    if max_v == min_v:
        return pd.Series(np.full(len(values), 5.0), index=series.index)
    return 1.0 + 9.0 * (values - min_v) / (max_v - min_v)


def read_excel_dataset(uploaded_file: Optional[object]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read either uploaded workbook or bundled default workbook."""
    source = uploaded_file if uploaded_file is not None else DEFAULT_DATASET_PATH
    try:
        user_feedback = pd.read_excel(source, sheet_name="User Feedback")
        sales_notes = pd.read_excel(source, sheet_name="Sales Notes")
        market_research = pd.read_excel(source, sheet_name="Market Research")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find {DEFAULT_DATASET_PATH}. Put the dataset at that path or upload a workbook."
        ) from exc
    except ValueError as exc:
        raise ValueError(
            "Workbook must contain sheets named: User Feedback, Sales Notes, Market Research."
        ) from exc
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
    out["severity_raw"] = df.get("severity", "unknown")
    out["frequency_signal"] = pd.to_numeric(df.get("frequency_signal", 1), errors="coerce").fillna(1)
    out["revenue_impact_usd"] = 0
    out["market_threat"] = "unknown"
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
    out["severity_raw"] = df.get("blocker_level", "unknown")
    out["frequency_signal"] = 1
    out["revenue_impact_usd"] = pd.to_numeric(df.get("arr_impact_usd", 0), errors="coerce").fillna(0)
    out["market_threat"] = "unknown"
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
    out["severity_raw"] = df.get("threat_level", "unknown")
    out["frequency_signal"] = 1
    out["revenue_impact_usd"] = 0
    out["market_threat"] = df.get("threat_level", "unknown")
    out["evidence_source"] = out["record_id"].map(lambda x: f"Market research {x}")
    return out


def load_and_normalize(uploaded_file: Optional[object]) -> LoadedData:
    user_feedback, sales_notes, market_research = read_excel_dataset(uploaded_file)

    normalized = pd.concat(
        [
            normalize_user_feedback(user_feedback),
            normalize_sales_notes(sales_notes),
            normalize_market_research(market_research),
        ],
        ignore_index=True,
    )

    normalized["text"] = normalized["text"].fillna("").astype(str)
    normalized["theme"] = normalized["theme"].fillna("unknown").astype(str)
    normalized["product_area"] = normalized["product_area"].fillna("unknown").astype(str)
    normalized["persona"] = normalized["persona"].fillna("unknown").astype(str)
    normalized["severity_score"] = normalized["severity_raw"].apply(normalize_severity)
    normalized["segment_score"] = normalized["company_segment"].apply(normalize_segment)
    normalized["threat_score"] = normalized["market_threat"].apply(normalize_threat)
    normalized["revenue_scaled"] = scale_1_to_10(normalized["revenue_impact_usd"])

    # Business impact proxy by source.
    # Sales uses ARR, market uses threat score, user feedback uses severity + segment.
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

    return LoadedData(user_feedback, sales_notes, market_research, normalized)


def top_evidence(group: pd.DataFrame, max_items: int = 5) -> List[str]:
    ranked = group.sort_values(
        ["severity_score", "business_impact_score", "frequency_signal"],
        ascending=False,
    )
    return ranked["evidence_text"].head(max_items).tolist()


def persona_impact(group: pd.DataFrame) -> str:
    counts = group["persona"].fillna("unknown").value_counts().head(4)
    return ", ".join([f"{persona} ({count})" for persona, count in counts.items()])


def source_mix(group: pd.DataFrame) -> str:
    counts = group["source_type"].value_counts()
    return ", ".join([f"{src}: {count}" for src, count in counts.items()])


def compute_theme_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for theme, group in df.groupby("theme"):
        mentions = len(group)
        weighted_frequency = group["frequency_signal"].sum()
        avg_severity = group["severity_score"].mean()
        avg_business = group["business_impact_score"].mean()
        revenue = group["revenue_impact_usd"].sum()
        confidence = min(10.0, np.log1p(mentions) * 3.0)

        # Explainable MVP scoring model.
        frequency_score = min(10.0, np.log1p(weighted_frequency) * 2.5)
        priority_score = (
            frequency_score * 0.25
            + avg_severity * 1.5 * 0.25
            + avg_business * 0.20
            + confidence * 0.15
            + min(10.0, revenue / 25000.0) * 0.10
        )
        priority_score = round(float(priority_score), 2)

        recommendation = next(label for threshold, label in ROADMAP_LABELS if priority_score >= threshold)

        rows.append(
            {
                "theme": theme,
                "product_area": group["product_area"].mode().iat[0] if not group["product_area"].mode().empty else "unknown",
                "mentions": mentions,
                "weighted_frequency": int(weighted_frequency),
                "avg_severity": round(float(avg_severity), 2),
                "business_impact_score": round(float(avg_business), 2),
                "sales_arr_signal_usd": int(revenue),
                "confidence_score": round(float(confidence), 2),
                "priority_score": priority_score,
                "roadmap_recommendation": recommendation,
                "persona_impact": persona_impact(group),
                "source_mix": source_mix(group),
                "evidence": top_evidence(group, max_items=5),
            }
        )

    return pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)


def make_evidence_summary(theme_row: pd.Series) -> str:
    return (
        f"**{theme_row['theme']}** appears in {theme_row['mentions']} signals "
        f"across {theme_row['source_mix']}. It mainly affects {theme_row['persona_impact']}. "
        f"The theme has a severity score of {theme_row['avg_severity']}/5, "
        f"business impact score of {theme_row['business_impact_score']}/10, and "
        f"a sales ARR signal of ${theme_row['sales_arr_signal_usd']:,}. "
        f"Recommended action: **{theme_row['roadmap_recommendation']}**."
    )


def suggested_opportunity(theme: str) -> str:
    lowered = theme.lower()
    if "timeline" in lowered:
        return "AI Incident Timeline Builder"
    if "alert" in lowered or "dedup" in lowered:
        return "Alert Correlation and Deduplication Engine"
    if "root" in lowered or "rca" in lowered:
        return "Evidence-Linked AI Root Cause Assistant"
    if "postmortem" in lowered:
        return "AI Postmortem Drafting Assistant"
    if "slack" in lowered or "jira" in lowered or "integration" in lowered:
        return "Incident Workflow Integration Hub"
    if "executive" in lowered or "visibility" in lowered or "status" in lowered:
        return "Incident Impact and Executive Summary Dashboard"
    if "compliance" in lowered or "audit" in lowered:
        return "Enterprise Incident Audit Trail"
    return f"{theme} Improvement Initiative"


def generate_prd_template(top_theme: pd.Series, supporting_records: pd.DataFrame) -> str:
    opportunity = suggested_opportunity(str(top_theme["theme"]))
    evidence = top_evidence(supporting_records, max_items=6)
    personas = supporting_records["persona"].value_counts().head(5).index.tolist()
    personas_text = ", ".join(personas)

    return f"""# PRD: {opportunity}

## 1. Background
Product teams building incident management platforms receive repeated feedback around **{top_theme['theme']}**. The evidence appears across {top_theme['source_mix']} and affects {top_theme['persona_impact']}.

## 2. Problem Statement
During production incidents, teams struggle with **{top_theme['theme']}**, which increases triage time, creates coordination overhead, and reduces trust in the incident management workflow.

## 3. Target Users
Primary personas: {personas_text}

## 4. Evidence
{chr(10).join([f"- {item}" for item in evidence])}

## 5. Goals
- Reduce time required to understand the incident state.
- Make incident decisions more evidence-backed.
- Improve trust in AI-assisted incident workflows.
- Convert messy operational signals into clear, reviewable recommendations.

## 6. Non-Goals
- No autonomous remediation in v1.
- No production changes without human approval.
- No unsupported AI claims without cited evidence.
- No replacement for existing PagerDuty, Jira, Slack, or observability tools in v1.

## 7. MVP Requirements
1. Ingest incident-related signals from feedback, sales notes, and market research.
2. Retrieve supporting evidence for the selected theme.
3. Generate an evidence-backed summary of the customer pain.
4. Show impacted personas and source mix.
5. Recommend a roadmap priority with explainable scoring.
6. Generate a draft PRD or Jira epic for PM review.
7. Allow the PM to accept, edit, or reject the AI recommendation.

## 8. Success Metrics
- 30% reduction in time to synthesize feedback into roadmap themes.
- 80%+ of generated summaries rated useful by PM reviewers.
- <5% unsupported claim rate in AI-generated summaries.
- Increase in evidence-linked roadmap decisions.
- Higher confidence from sales/support that repeated pain points are tracked.

## 9. AI Evaluation Plan
- Create a 30–50 item human-labeled gold dataset.
- Measure theme classification accuracy.
- Measure evidence retrieval precision.
- Track hallucination / unsupported-claim rate.
- Ask PMs to rate generated PRDs for usefulness and completeness.

## 10. Risks and Mitigations
- **Risk:** AI over-prioritizes noisy but low-value feedback.  
  **Mitigation:** Combine frequency with severity, revenue impact, and confidence.
- **Risk:** AI invents unsupported roadmap recommendations.  
  **Mitigation:** Require every recommendation to cite source evidence.
- **Risk:** Sales feedback biases roadmap toward enterprise-only requests.  
  **Mitigation:** Display source mix and segment breakdown for every theme.

## 11. Launch Plan
- Pilot with one platform PM and one SRE/devtools team.
- Review recommendations weekly against human PM judgment.
- Expand to sales/support users after trust and evidence quality are validated.

## 12. Open Questions
- Which feedback sources should be trusted most for roadmap prioritization?
- How should PMs adjust scoring weights by product strategy?
- What is the minimum evidence threshold before creating a roadmap candidate?
"""


def maybe_generate_with_openai(prompt: str, enabled: bool) -> Optional[str]:
    if not enabled:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("OPENAI_API_KEY is not set. Falling back to local template output.")
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI Product Manager. Be concise, evidence-backed, and practical.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as exc:  # pragma: no cover - UI fallback
        st.warning(f"OpenAI generation failed; using local template. Error: {exc}")
        return None


def retrieve_evidence(df: pd.DataFrame, query: str, top_k: int = 8) -> pd.DataFrame:
    corpus = (
        df["theme"].fillna("")
        + " "
        + df["product_area"].fillna("")
        + " "
        + df["persona"].fillna("")
        + " "
        + df["text"].fillna("")
    ).tolist()

    if not query.strip() or not corpus:
        return df.head(0)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()

    result = df.copy()
    result["retrieval_score"] = scores
    return result.sort_values("retrieval_score", ascending=False).head(top_k)


def display_metric_cards(theme_summary: pd.DataFrame, normalized: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total signals", len(normalized))
    col2.metric("Themes", theme_summary["theme"].nunique())
    col3.metric("Sales ARR signal", f"${int(normalized['revenue_impact_usd'].sum()):,}")
    col4.metric("Top score", f"{theme_summary['priority_score'].max():.2f}/10")


def main() -> None:
    st.set_page_config(page_title="AI Feedback-to-Roadmap Copilot", layout="wide")
    st.title("AI Feedback-to-Roadmap Copilot")
    st.caption("MVP v1 for a developer incident management platform")

    with st.sidebar:
        st.header("Data")
        uploaded_file = st.file_uploader(
            "Upload feedback workbook",
            type=["xlsx"],
            help="Expected sheets: User Feedback, Sales Notes, Market Research",
        )
        use_llm = st.toggle("Use OpenAI for richer writing", value=False)
        st.divider()
        st.header("Scoring")
        st.write("MVP scoring combines frequency, severity, business impact, confidence, and ARR signal.")

    try:
        data = load_and_normalize(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    normalized = data.normalized
    theme_summary = compute_theme_summary(normalized)
    top10 = theme_summary.head(10)

    display_metric_cards(theme_summary, normalized)

    tabs = st.tabs(
        [
            "Top 10 Themes",
            "Evidence Summaries",
            "Persona + Business Impact",
            "Roadmap Priorities",
            "Draft PRD",
            "Ask the Evidence Base",
            "Raw Data",
        ]
    )

    with tabs[0]:
        st.subheader("Top 10 Themes")
        display_cols = [
            "theme",
            "product_area",
            "mentions",
            "avg_severity",
            "business_impact_score",
            "sales_arr_signal_usd",
            "confidence_score",
            "priority_score",
            "roadmap_recommendation",
        ]
        st.dataframe(top10[display_cols], use_container_width=True, hide_index=True)

        csv = top10[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download top 10 themes CSV", csv, "top_10_themes.csv", "text/csv")

    with tabs[1]:
        st.subheader("Evidence-backed summaries")
        selected_theme = st.selectbox("Choose theme", top10["theme"].tolist())
        row = top10[top10["theme"].eq(selected_theme)].iloc[0]
        st.markdown(make_evidence_summary(row))
        st.markdown("### Evidence")
        for item in row["evidence"]:
            st.markdown(f"- {item}")

        theme_records = normalized[normalized["theme"].eq(selected_theme)]
        st.markdown("### All records for this theme")
        st.dataframe(
            theme_records[["record_id", "source_type", "persona", "product_area", "severity_raw", "revenue_impact_usd", "text"]],
            use_container_width=True,
            hide_index=True,
        )

    with tabs[2]:
        st.subheader("Persona impact and business impact")
        persona_counts = normalized.groupby(["theme", "persona"]).size().reset_index(name="mentions")
        persona_counts = persona_counts.sort_values(["theme", "mentions"], ascending=[True, False])
        st.dataframe(persona_counts, use_container_width=True, hide_index=True)

        st.markdown("### Business impact by theme")
        impact_cols = [
            "theme",
            "source_mix",
            "persona_impact",
            "business_impact_score",
            "sales_arr_signal_usd",
            "confidence_score",
        ]
        st.dataframe(top10[impact_cols], use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Suggested roadmap priorities")
        roadmap = top10.copy()
        roadmap["suggested_opportunity"] = roadmap["theme"].apply(suggested_opportunity)
        roadmap["why_now"] = roadmap.apply(
            lambda r: (
                f"{r['mentions']} signals, severity {r['avg_severity']}/5, "
                f"business impact {r['business_impact_score']}/10, ARR signal ${r['sales_arr_signal_usd']:,}."
            ),
            axis=1,
        )
        roadmap_cols = [
            "suggested_opportunity",
            "theme",
            "priority_score",
            "roadmap_recommendation",
            "why_now",
        ]
        st.dataframe(roadmap[roadmap_cols], use_container_width=True, hide_index=True)

        st.markdown("### Recommended sequence")
        for i, r in roadmap.head(5).iterrows():
            st.markdown(
                f"**{i + 1}. {r['suggested_opportunity']}** — {r['roadmap_recommendation']}  \n"
                f"{r['why_now']}"
            )

    with tabs[4]:
        st.subheader("Draft PRD for top opportunity")
        top_theme = top10.iloc[0]
        supporting = normalized[normalized["theme"].eq(top_theme["theme"])]
        local_prd = generate_prd_template(top_theme, supporting)

        if use_llm:
            prompt = (
                "Create a concise but complete PM PRD from this evidence. "
                "Keep evidence citations as bullet points.\n\n"
                f"Top theme row:\n{top_theme.to_dict()}\n\n"
                f"Evidence:\n{supporting[['evidence_source','persona','product_area','text']].head(12).to_dict(orient='records')}"
            )
            llm_prd = maybe_generate_with_openai(prompt, enabled=True)
            st.markdown(llm_prd or local_prd)
        else:
            st.markdown(local_prd)

        st.download_button(
            "Download PRD markdown",
            local_prd.encode("utf-8"),
            "draft_prd_top_opportunity.md",
            "text/markdown",
        )

    with tabs[5]:
        st.subheader("Ask the evidence base")
        query = st.text_input(
            "Ask a product question",
            value="What should we build first to reduce incident triage time?",
        )
        results = retrieve_evidence(normalized, query, top_k=8)
        if len(results) > 0:
            st.markdown("### Retrieved evidence")
            for _, r in results.iterrows():
                st.markdown(
                    f"- **{r['evidence_source']}** | {r['theme']} | {r['persona']} | score {r['retrieval_score']:.3f}  \n"
                    f"  {r['text']}"
                )

            context = "\n".join(results["evidence_text"].tolist())
            local_answer = (
                "Based on the retrieved evidence, prioritize the highest-frequency/high-severity themes that "
                "directly reduce triage time and increase trust. Review the cited records above before creating "
                "a roadmap commitment."
            )
            if use_llm:
                prompt = f"Answer this PM question using only the evidence below.\nQuestion: {query}\n\nEvidence:\n{context}"
                answer = maybe_generate_with_openai(prompt, enabled=True) or local_answer
                st.markdown("### Suggested answer")
                st.markdown(answer)
            else:
                st.markdown("### Suggested answer")
                st.markdown(local_answer)
        else:
            st.info("Enter a question to retrieve evidence.")

    with tabs[6]:
        st.subheader("Normalized data")
        st.dataframe(normalized, use_container_width=True, hide_index=True)

        st.markdown("### Original sheets")
        with st.expander("User Feedback"):
            st.dataframe(data.user_feedback, use_container_width=True, hide_index=True)
        with st.expander("Sales Notes"):
            st.dataframe(data.sales_notes, use_container_width=True, hide_index=True)
        with st.expander("Market Research"):
            st.dataframe(data.market_research, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
