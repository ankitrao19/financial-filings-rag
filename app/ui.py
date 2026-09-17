"""
ui.py — thin Gradio front end. Talks to the FastAPI service over HTTP only; no pipeline imports.

Run (API must be up first):
    venv/bin/python app/ui.py                         # http://localhost:7860
    RAG_API_URL=http://host:8000 venv/bin/python app/ui.py
"""

import os
import uuid

import gradio as gr
import httpx

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
TIMEOUT = 120

EXAMPLES = [
    ["What was Costco's total revenue in fiscal 2024?", "COST"],
    ["How did Costco's net income change from fiscal 2022 to fiscal 2024?", "COST"],
    ["How many warehouses did Costco operate worldwide as of September 1, 2024?", "COST"],
]


def load_tickers():
    try:
        return httpx.get(f"{API_URL}/tickers", timeout=10).json()
    except httpx.HTTPError:
        return []


def ask(question, ticker, k, filing, prompt, session_id):
    if not question.strip():
        return "Type a question.", "", "", None, gr.update(visible=False)
    try:
        r = httpx.post(f"{API_URL}/ask", timeout=TIMEOUT, json={
            "question": question, "ticker": ticker or None, "k": int(k),
            "filing": filing, "prompt": prompt, "session_id": session_id,
        })
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"API error {e.response.status_code}: {e.response.text}", "", "", None, gr.update(visible=False)
    except httpx.HTTPError as e:
        return f"Cannot reach API at {API_URL}: {e}", "", "", None, gr.update(visible=False)

    data = r.json()
    trace = f"[open trace in Langfuse]({data['trace_url']})" if data["trace_url"] else "tracing off"
    info = (f"**filing filter:** {data['filing_filter'] or 'none'} ({data['inference_reason']}) · "
            f"**latency:** {data['latency_ms']} ms · {trace}")
    sources = "\n\n".join(
        f"**{s['rank']}. {s['ticker']} {s['form']} {s['filing_date']}** · distance {s['distance']}\n\n"
        f"> {s['snippet'].replace(chr(10), ' ')[:300]}…"
        for s in data["sources"]
    )
    return data["answer"], info, sources, data["trace_id"], gr.update(visible=bool(data["trace_url"]))


def send_feedback(trace_id, value):
    if not trace_id:
        return "Ask something first."
    try:
        httpx.post(f"{API_URL}/feedback", json={"trace_id": trace_id, "value": value}, timeout=10).raise_for_status()
        return "Thanks — recorded in Langfuse."
    except httpx.HTTPError as e:
        return f"Feedback failed: {e}"


with gr.Blocks(title="Filings RAG") as demo:
    session_id = gr.State(lambda: str(uuid.uuid4()))
    trace_id = gr.State(None)

    gr.Markdown("# Financial-filings RAG\nAsk about 10-K / 10-Q filings for 15 companies.")
    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(label="Question", lines=2)
            ticker = gr.Dropdown(choices=load_tickers(), label="Company", value=None)
            with gr.Accordion("Retrieval settings", open=False):
                k = gr.Slider(1, 50, value=15, step=1, label="Chunks (k)")
                filing = gr.Radio(["inferred", "none"], value="inferred", label="Filing filter")
                prompt = gr.Radio(["v1", "v2"], value="v1", label="Answer prompt")
            submit = gr.Button("Ask", variant="primary")
            gr.Examples(EXAMPLES, inputs=[question, ticker])
        with gr.Column(scale=4):
            answer = gr.Markdown(label="Answer")
            info = gr.Markdown()
            with gr.Row(visible=False) as feedback_row:
                up = gr.Button("👍 Helpful", size="sm")
                down = gr.Button("👎 Not helpful", size="sm")
            feedback_msg = gr.Markdown()
            with gr.Accordion("Sources", open=False):
                sources = gr.Markdown()

    inputs = [question, ticker, k, filing, prompt, session_id]
    outputs = [answer, info, sources, trace_id, feedback_row]
    submit.click(ask, inputs, outputs)
    question.submit(ask, inputs, outputs)
    up.click(lambda t: send_feedback(t, 1), trace_id, feedback_msg)
    down.click(lambda t: send_feedback(t, 0), trace_id, feedback_msg)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
