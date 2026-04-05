"""
Paper Researcher — terminal-style Gradio demo with parallel streaming.
Usage: uv run instruction_tuning/app.py models/mlx/my-model [--mlx]
"""
import sys, os, argparse, re, json, ast
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from paper_researcher import PaperResearcher, QAPair, Triplet, RetrievalResult
from paper_researcher.tasks import SYSTEM_PROMPT
import paper_researcher.tasks as t
import gradio as gr

MAX_NEW_TOKENS = 512

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="Path to model directory")
parser.add_argument("--mlx", action="store_true", help="Use MLX backend")
parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
parser.add_argument("--n", type=int, default=2, choices=[1, 2, 3, 4], help="Number of generations")
args, _ = parser.parse_known_args()

researcher = PaperResearcher(args.model_path, mlx=args.mlx)

# ── Theme + CSS ───────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("JetBrains Mono"), "Fira Code", "monospace"],
).set(
    # backgrounds
    body_background_fill="#0d1117",
    block_background_fill="#0d1117",
    panel_background_fill="#0d1117",
    # inputs
    input_background_fill="#161b22",
    input_border_color="#30363d",
    input_border_color_focus="#58a6ff",
    # blocks
    block_border_color="#30363d",
    block_border_width="1px",
    block_label_text_color="#58a6ff",
    block_label_text_size="11px",
    # body text
    body_text_color="#c9d1d9",
    body_text_size="12px",
    # buttons
    button_primary_background_fill="#238636",
    button_primary_background_fill_hover="#2ea043",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#21262d",
    button_secondary_border_color="#30363d",
    button_secondary_text_color="#c9d1d9",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

footer { display: none !important; }

/* Green output text */
.output-box textarea { color: #ffffff !important; font-size: 24px !important; line-height: 1.6 !important; }

/* Dropdown list items */
ul[role="listbox"] { background: #161b22 !important; }
ul[role="listbox"] li { color: #e6edf3 !important; background: #161b22 !important; }
ul[role="listbox"] li:hover { background: #21262d !important; }

/* Amber messages preview */
.messages-box textarea { color: #e3b341 !important; font-size: 11px !important; }

/* Temperature headers */
.temp-header p {
    color: #f78166 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    margin: 4px 0 2px 0 !important;
}

/* Left panel textareas — fixed heights, no layout shift
   Overhead: header(60) + mode block(70) + button(52) + labels(40) + gaps/padding(50) = ~272px */
#left-panel .input-box textarea {
    height: calc((100vh - 272px) * 0.28) !important;
    overflow-y: auto !important;
    resize: none !important;
}
#left-panel .messages-box textarea {
    height: calc((100vh - 272px) * 0.62) !important;
    overflow-y: auto !important;
    resize: none !important;
}

/* Full-width generate button */
.gen-btn button { width: 100% !important; letter-spacing: 1px !important; }

/* Stats bar */
.stats-bar p {
    color: #8b949e !important;
    font-size: 14px !important;
    letter-spacing: 1px !important;
    margin: 2px 0 4px 0 !important;
}

/* Tighten spacing */
.block { padding: 6px !important; }
.gap, .gap-2 { gap: 6px !important; }

/* Left divider */
.left-col { border-right: 1px solid #21262d !important; }

"""

# Dynamic textarea height — bypasses Gradio's wrapper div chain
_PANEL_H   = "100vh - 60px"
_ROW_COUNT = 2 if args.n > 2 else 1
_BOTTOM_PAD = 80  # aligns with generate button + padding
_BOX_H     = f"calc(({_PANEL_H}) / {_ROW_COUNT} - {_BOTTOM_PAD}px)"

CSS += f"""
#right-panel .output-box textarea {{
    height: {_BOX_H} !important;
    overflow-y: auto !important;
    resize: none !important;
}}
"""

# ── Modes ─────────────────────────────────────────────────────────────────────

def _fmt_bullets(items):  return "\n".join(f"- {b}" for b in items)
def _fmt_qa(pairs):       return "\n\n".join(f"Q: {p.question}\nA: {p.answer}" for p in pairs)
def _fmt_triplets(items): return "\n".join(f"({t.subject}, {t.relation}, {t.object})" for t in items)
def _fmt_retrieval(r):
    idx = f"Passage {r.index + 1}" if r.index is not None else "None"
    return f"Passage: {idx}\n\n{r.reasoning}"

def _split2(text):
    parts = text.split("\n\n", 1)
    return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")

def _parse_retrieval(text):
    passages = re.findall(r"Passage \d+:\s*(.*?)(?=Passage \d+:|Question:|$)", text, re.DOTALL)
    passages = [p.strip() for p in passages if p.strip()]
    q = re.search(r"Question:\s*(.+)", text, re.DOTALL)
    return (q.group(1).strip() if q else ""), passages

MODES = {
    "bullets":      {"desc": "Extract key points as bullets",          "hint": "Paste passage...",                                  "call": lambda r, x, **k: _fmt_bullets(r.extract_bullets(x, **k))},
    "qa_pairs":     {"desc": "Generate Q&A pairs",                     "hint": "Paste passage...",                                  "call": lambda r, x, **k: _fmt_qa(r.generate_qa_pairs(x, **k))},
    "question":     {"desc": "Generate a question from passage",       "hint": "Paste passage...",                                  "call": lambda r, x, **k: r.generate_question(x, **k)},
    "fact":         {"desc": "Extract a single fact",                  "hint": "Paste passage...",                                  "call": lambda r, x, **k: r.extract_fact(x, **k)},
    "answer":       {"desc": "Answer question given passage",          "hint": "passage...\n\n[blank line]\n\nquestion",            "call": lambda r, x, **k: r.answer(*reversed(_split2(x)), **k)},
    "rephrase":     {"desc": "Rephrase and elaborate",                 "hint": "Paste passage...",                                  "call": lambda r, x, **k: r.rephrase(x, **k)},
    "continuation": {"desc": "Continue passage from beginning",        "hint": "Paste start of passage...",                        "call": lambda r, x, **k: r.continue_from(x, **k)},
    "triplets":     {"desc": "Extract knowledge graph triplets",       "hint": "Paste passage...",                                  "call": lambda r, x, **k: _fmt_triplets(r.extract_triplets(x, **k))},
    "comparison":   {"desc": "Compare two passages",                   "hint": "passage 1...\n\n[blank line]\n\npassage 2",        "call": lambda r, x, **k: r.compare(*_split2(x), **k)},
    "retrieval":    {"desc": "Find which passage answers a question",  "hint": "Passage 1: ...\n\nPassage 2: ...\n\nQuestion: ...", "call": lambda r, x, **k: _fmt_retrieval(r.find_relevant(*_parse_retrieval(x), **k))},
}

MODE_KEYS    = list(MODES.keys())
MODE_CHOICES = [f"{k}  —  {MODES[k]['desc']}" for k in MODE_KEYS]

INSTRUCTION_MAP = {
    "bullets":      t.BULLETS_INSTRUCTION,
    "qa_pairs":     t.QA_PAIRS_INSTRUCTION,
    "question":     t.QUESTION_FROM_PASSAGE_INSTRUCTION,
    "fact":         t.FACT_FROM_PASSAGE_INSTRUCTION,
    "answer":       t.QA_ANSWER_INSTRUCTION,
    "rephrase":     t.REPHRASE_INSTRUCTION,
    "continuation": t.CONTINUATION_INSTRUCTION,
    "triplets":     t.TRIPLETS_INSTRUCTION,
    "comparison":   t.COMPARISON_INSTRUCTION,
    "retrieval":    t.RETRIEVAL_INSTRUCTION,
}

def key_from_choice(c): return c.split("  —  ")[0].strip()

def on_mode_change(choice):
    key = key_from_choice(choice)
    return gr.update(placeholder=MODES[key]["hint"])

def build_messages_preview(choice, text):
    if not text.strip(): return ""
    key  = key_from_choice(choice)
    inst = INSTRUCTION_MAP.get(key, "")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{inst}\n\n{text}"},
    ]
    return json.dumps(messages, indent=2)

# ── JSON pretty-print ─────────────────────────────────────────────────────────

def _maybe_prettify(text: str) -> str:
    stripped = text.strip()

    # Python list literal → join items with \n\n
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            val = ast.literal_eval(stripped)
            if isinstance(val, list):
                return "\n\n".join(str(item) for item in val)
        except (ValueError, SyntaxError):
            pass

    # JSON object/array
    try:
        return json.dumps(json.loads(stripped), indent=2)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fenced ```json block
    m = re.search(r"```json\s*([\s\S]+?)```", stripped)
    if m:
        try:
            pretty = json.dumps(json.loads(m.group(1)), indent=2)
            return stripped[:m.start()] + f"```json\n{pretty}\n```" + stripped[m.end():]
        except (json.JSONDecodeError, ValueError):
            pass

    return text

# ── Parallel streaming ────────────────────────────────────────────────────────

def build_prompt(mode_key, user_text):
    inst = INSTRUCTION_MAP[mode_key]
    tok  = researcher._backend.tokenizer
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{inst}\n\n{user_text}"},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _fmt_stats(stats: dict) -> str:
    if not stats:
        return ""
    return (
        f"// {stats['tokens']} tok · "
        f"{stats['tps']:.1f} tok/s · "
        f"{stats['peak_memory']:.2f} GB peak"
    )


def generate_stream(mode_choice, user_text):
    n = args.n
    if not user_text.strip():
        yield [""] + ["// no input"] * n
        return

    key    = key_from_choice(mode_choice)
    prompt = build_prompt(key, user_text)
    temp   = args.temperature

    results = [""] * n
    stats_text = ""

    for idx in range(n):
        try:
            for chunk in researcher._backend.stream(
                prompt, temperature=temp, max_new_tokens=MAX_NEW_TOKENS
            ):
                results[idx] += chunk
                yield [stats_text] + [_maybe_prettify(r) for r in results]
            stats_text = _fmt_stats(getattr(researcher._backend, "_last_stats", {}))
            yield [stats_text] + [_maybe_prettify(r) for r in results]
        except Exception as e:
            results[idx] = f"// error: {e}"
            yield [stats_text] + [_maybe_prettify(r) for r in results]

# ── UI ────────────────────────────────────────────────────────────────────────

model_name = os.path.basename(os.path.normpath(args.model_path))

with gr.Blocks(title="paper_researcher") as demo:

    gr.Markdown(
        f"<span style='color:#58a6ff;font-size:13px;font-weight:600'>// paper_researcher</span>"
        f"&nbsp;&nbsp;<span style='color:#8b949e;font-size:11px'>{model_name}</span>"
    )

    with gr.Row(equal_height=False):

        # ── Left ──────────────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes=["left-col"], elem_id="left-panel"):

            mode_dd = gr.Dropdown(
                choices=MODE_CHOICES, value=MODE_CHOICES[0], label="mode",
            )

            user_box = gr.Textbox(
                label="input", lines=7,
                placeholder=MODES[MODE_KEYS[0]]["hint"],
                elem_classes=["input-box"],
            )

            messages_box = gr.Textbox(
                label="messages", lines=11,
                interactive=False,
                elem_classes=["messages-box"],
            )

            gen_btn = gr.Button("▶  GENERATE", variant="primary", elem_classes=["gen-btn"])

        # ── Right: grid from args.n ───────────────────────────────────────────
        with gr.Column(scale=1, elem_id="right-panel"):
            stats_md = gr.Markdown("", elem_classes=["stats-bar"])
            n = args.n
            cols_per_row = 2 if n > 1 else 1
            output_cols  = []
            for row_start in range(0, n, cols_per_row):
                with gr.Row():
                    for i in range(row_start, min(row_start + cols_per_row, n)):
                        output_cols.append(
                            gr.Textbox(
                                show_label=False, lines=14,
                                interactive=False,
                                elem_classes=["output-box"],
                            )
                        )

    # ── Wiring ────────────────────────────────────────────────────────────────

    mode_dd.change(fn=on_mode_change,          inputs=mode_dd,             outputs=user_box)
    mode_dd.change(fn=build_messages_preview,  inputs=[mode_dd, user_box], outputs=messages_box)
    user_box.change(fn=build_messages_preview, inputs=[mode_dd, user_box], outputs=messages_box)

    gen_btn.click(
        fn=generate_stream,
        inputs=[mode_dd, user_box],
        outputs=[stats_md] + output_cols,
        show_progress=False,
    )

demo.launch(css=CSS, theme=THEME)
