# AI Feedback-to-Roadmap Copilot — MVP2

 It upgrades MVP1 into a real RAG-based workflow using:

- **LangChain** orchestration
- **Chroma** persistent vector database
- **Local sentence-transformers embeddings** by default, with optional OpenAI embeddings
- **OpenAI chat model** via `gpt-4o-mini` for optional generated answers/PRDs
- Better product-management visualizations

The app turns product feedback for a developer incident management platform into:

- Top 10 themes
- Evidence-backed theme summaries
- Persona impact
- Business/revenue impact
- Suggested roadmap priorities
- Draft PRD for the top opportunity
- RAG Q&A over the feedback evidence base

## Setup

```bash
cd ai_feedback_roadmap_mvp2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Optional: edit .env and set OPENAI_API_KEY for AI-generated answers/PRDs
streamlit run app.py
```

On Windows PowerShell:

```powershell
cd ai_feedback_roadmap_mvp2
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

## What is new in MVP2?

### MVP1

MVP1 used TF-IDF retrieval from scikit-learn. That was a RAG-style prototype but not a standard LangChain/Chroma RAG stack.

### MVP2

MVP2 builds a Chroma vector database from feedback rows. Each feedback row is converted to a LangChain `Document` with metadata. Local sentence-transformers embeddings are stored in Chroma by default, or you can switch to OpenAI embeddings in the sidebar. The RAG tab retrieves semantically relevant evidence and, when `OPENAI_API_KEY` is available, passes it to an LLM with strict instructions to answer only from the retrieved evidence. Without an API key, the app still shows retrieval-only evidence.

## App tabs

1. **Overview** — dataset counts, theme chart, priority scatter plot, source mix.
2. **Roadmap Priorities** — top 10 themes, impact scoring, recommendation labels, evidence cards.
3. **RAG Evidence Q&A** — semantic search + evidence-grounded answer generation.
4. **Draft PRD** — generated PRD for the top roadmap opportunity.
5. **Evidence Explorer** — filter and inspect all underlying records.
6. **Methodology** — explains the scoring model and RAG architecture.

## Data

The bundled synthetic dataset is in:

```text
data/feedback_dataset.xlsx
```

It contains:

- 100 user feedback comments
- 50 sales notes
- 30 market / competitor research snippets

You can upload another workbook in the app. It must contain these sheets:

- `User Feedback`
- `Sales Notes`
- `Market Research`


## Notes

- This is a portfolio MVP, not a production incident-management product.
- Do not use confidential company data.
- API calls are only needed if you select OpenAI embeddings or generate AI answers/PRDs with OpenAI.
- The default local embedding model may download once on first use, then runs from cache.
