# AI Feedback-to-Roadmap Copilot — MVP v1

A local-first Streamlit MVP for a developer incident management platform.

## Inputs
- 100 user feedback comments
- 50 sales notes
- 30 competitor / market research snippets

## Outputs
- Top 10 themes
- Evidence-backed summaries
- Persona impact
- Revenue / business impact
- Suggested roadmap priorities
- Draft PRD for the top opportunity
- Simple RAG-style Q&A over feedback evidence

## Run locally

```bash
cd ai_feedback_roadmap_mvp
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The app includes a synthetic dataset in `data/feedback_dataset.xlsx`.
You can also upload your own Excel file with these sheets:
- `User Feedback`
- `Sales Notes`
- `Market Research`

## Optional OpenAI enhancement

The MVP works without an API key using deterministic templates.
If you want higher-quality summaries/PRDs, set:

```bash
export OPENAI_API_KEY="your_key_here"
```

Then enable the LLM toggle in the sidebar.

## Notes
- This is a PM portfolio MVP, not a production product.
- Data is synthetic and safe for demos.
- The scoring model is intentionally simple and explainable.
