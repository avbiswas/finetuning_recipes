"""
Inference using the exact prompt format from training data.
Each mode maps to the instructions used during dataset generation.

Usage:
    uv run instruction_tuning/task_chat.py -m models/mlx/smollm-135m_instruct_v1 --mlx
"""
import argparse
import sys
import tty
import termios
from dataclasses import dataclass
from typing import Callable

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """


@dataclass
class Mode:
    description: str           # shown in the menu
    instructions: list[str]    # exact instruction strings from training
    input_hint: str = "Paste passage (Enter=newline, Esc+Enter=submit):"
    build_input: Callable[[str], str] = None  # optional: transforms raw text into the input field

    def format_user_message(self, instruction: str, text: str) -> str:
        inp = self.build_input(text) if self.build_input else text
        return f"{instruction}\n\n{inp}"


# ── Exact instructions from text_albumentations tasks ─────────────────────────

MODES: dict[str, Mode] = {

    # ── Bullets ────────────────────────────────────────────────────────────────
    "bullets_markdown": Mode(
        description="Extract key points as markdown bullet points",
        instructions=[
            "Extract the important points from this passage as markdown bullet points.",
            "Summarize this passage as markdown bullet points.",
        ],
    ),
    "bullets_python": Mode(
        description="Extract key points as a Python list of strings",
        instructions=[
            "Extract the important points from this passage as a Python list of strings.",
            "Return a Python list of the key points from this passage.",
        ],
    ),

    # ── QA — generation ────────────────────────────────────────────────────────
    "qa_pairs_markdown": Mode(
        description="Generate question-answer pairs from a passage (markdown output)",
        instructions=[
            "\nGiven this passage of text, generate a list of important question answer pairs.\n    ",
            "Generate a set of questions from this passage in markdown format.",
            "List the important questions answered by this passage using markdown.",
            "Generate some important facts from this passage in markdown bullet points.",
            "Generate one question and its corresponding answer from this passage in markdown format.",
        ],
    ),
    "qa_pairs_json": Mode(
        description="Generate question-answer pairs from a passage (JSON output)",
        instructions=[
            "\nGiven this passage of text, generate a list of important question answer pairs.\n    Generate as a list of json containing 'question' and 'answer' keys",
            "Generate a list of questions from this passage. Return a JSON array of strings.",
            "List the important questions answered by this passage. Return a JSON array of strings.",
            "Generate some facts from this passage. Return a JSON array of strings.",
            "Generate one question and it's corresponding answer from this passage. Return answer as a json of question and answer",
        ],
    ),
    "question_from_passage": Mode(
        description="Generate a single question from a passage",
        instructions=[
            "Generate a question from this passage",
        ],
    ),
    "fact_from_passage": Mode(
        description="Extract a single important fact from a passage",
        instructions=[
            "Generate an important fact or piece of information from this passage",
        ],
    ),

    # ── QA — answering ─────────────────────────────────────────────────────────
    "qa_answer_with_passage": Mode(
        description="Answer a question — input: passage then question on next line",
        instructions=[
            "Answer the user's question given the provided passage",
        ],
        input_hint="Paste: passage, then blank line, then question:",
        build_input=lambda text: (
            # Expects: "<passage>\n\n<question>"
            # Formats as training did: "Passage: <passage>\n\nQuestion: <question>\nWhat is the answer?"
            (lambda parts: f"Passage: {parts[0]}\n\nQuestion: {parts[1]}\nWhat is the answer?")(
                text.split("\n\n", 1) if "\n\n" in text else [text, ""]
            )
        ),
    ),
    "qa_answer_question_only": Mode(
        description="Answer a question — passage is baked into the instruction, input is just the question",
        instructions=[],  # built dynamically — see loop below
        input_hint="First enter the passage, then separately enter the question.",
    ),

    # ── Rephrase ───────────────────────────────────────────────────────────────
    "rephrase": Mode(
        description="Rephrase and elaborate a passage",
        instructions=[
            "\nGiven this passage, rephrase it. Elaborate on the sentences by explaining the meaning. Only present content that is strictly present in the passage, do not introduce new concepts outside the scope of this input. Do not re-quote the original. Only generate answers.\n    ",
        ],
    ),

    # ── Continuation ───────────────────────────────────────────────────────────
    "continuation_from_start": Mode(
        description="Continue a passage from its beginning (first ~10%)",
        instructions=[
            "You are given the beginning of a passage. Continue the passage by generating all remaining text after the provided beginning. Do not repeat the provided beginning.",
        ],
    ),
    "continuation_from_20pct": Mode(
        description="Continue a passage from its first 20%",
        instructions=[
            "You are given the first 20% of a passage. Generate the rest of the passage exactly after the provided text. Do not repeat the provided text.",
        ],
    ),
    "continuation_infill": Mode(
        description="Fill the missing middle — input: beginning + ending separated by blank line",
        instructions=[
            "You are given the beginning and the ending of a passage. Generate only the missing middle section that connects them. Do not repeat the provided beginning or ending.",
        ],
        input_hint="Paste beginning, blank line, then ending:",
        build_input=lambda text: (
            (lambda parts: f"Beginning:\n{parts[0]}\n\nEnding:\n{parts[1]}")(
                text.split("\n\n", 1) if "\n\n" in text else [text, ""]
            )
        ),
    ),

    # ── Triplets ───────────────────────────────────────────────────────────────
    "triplets_markdown": Mode(
        description="Extract knowledge graph triplets (markdown output)",
        instructions=[
            "Extract knowledge graph triplets from this passage in markdown format.",
            "List the subject-relation-object triplets from this passage as markdown bullet points.",
        ],
    ),
    "triplets_json": Mode(
        description="Extract knowledge graph triplets (JSON output)",
        instructions=[
            "Extract knowledge graph triplets from this passage and return them as JSON.",
            "Return a JSON array of subject-relation-object triplets supported by this passage.",
        ],
    ),

    # ── Retrieval ──────────────────────────────────────────────────────────────
    "retrieval_markdown": Mode(
        description="Identify which passage answers a question (markdown output) — input: passages + question",
        instructions=[
            "Read the passages and identify which passage answers the question. Return the passage number and a short justification in markdown.",
        ],
        input_hint="Paste: Passage 1: ...\n\nPassage 2: ...\n\nQuestion: ...",
    ),
    "retrieval_json": Mode(
        description="Identify which passage answers a question (JSON output) — input: passages + question",
        instructions=[
            "Read the passages and identify which passage answers the question. Return a JSON object with keys 'passage' and 'why'. Use a passage number when an answer is present, or null when none of the passages answer the question.",
        ],
        input_hint="Paste: Passage 1: ...\n\nPassage 2: ...\n\nQuestion: ...",
    ),

    # ── Comparison ─────────────────────────────────────────────────────────────
    "comparison": Mode(
        description="Compare two passages — input: passage 1, blank line, passage 2",
        instructions=[
            "\nGiven 2 passages of text, generate a detailed comparison of the two\n    ",
        ],
        input_hint="Paste passage 1, blank line, then passage 2:",
        build_input=lambda text: (
            (lambda parts: f"Passage 1:\n{parts[0]}\n\nPassage 2:\n{parts[1]}")(
                text.split("\n\n", 1) if "\n\n" in text else [text, ""]
            )
        ),
    ),
}


# ── Terminal helpers ──────────────────────────────────────────────────────────

def multiline_input(prompt="Input: ") -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf = []
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                next_ch = sys.stdin.read(1)
                if next_ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    break
                else:
                    buf.append(ch + next_ch)
            elif ch in ("\r", "\n"):
                buf.append("\n")
                sys.stdout.write("\r\n")
                sys.stdout.flush()
            elif ch in ("\x03", "\x04"):
                sys.stdout.write("\n")
                raise KeyboardInterrupt
            elif ch == "\x7f":
                if buf:
                    removed = buf.pop()
                    if removed == "\n":
                        sys.stdout.write("\x1b[A\x1b[9999C")
                    else:
                        sys.stdout.write("\b \b")
                    sys.stdout.flush()
            else:
                buf.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return "".join(buf).strip()


def pick_mode() -> str:
    print("\nAvailable modes:")
    keys = list(MODES.keys())
    for i, key in enumerate(keys):
        print(f"  {i+1:2}. {key:35s} — {MODES[key].description}")
    while True:
        try:
            choice = input("\nMode name or number: ").strip()
            if choice in MODES:
                return choice
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        except (ValueError, KeyboardInterrupt):
            pass
        print("Invalid choice.")


def pick_variant(mode_key: str) -> str:
    instructions = MODES[mode_key].instructions
    if not instructions or len(instructions) == 1:
        return instructions[0] if instructions else ""
    print(f"\nVariants:")
    for i, v in enumerate(instructions):
        print(f"  {i+1}. {v[:90].strip()}{'...' if len(v) > 90 else ''}")
    while True:
        try:
            choice = input(f"Pick [1-{len(instructions)}] (default 1): ").strip()
            idx = int(choice) - 1 if choice else 0
            if 0 <= idx < len(instructions):
                return instructions[idx]
        except (ValueError, KeyboardInterrupt):
            return instructions[0]


if __name__ == "__main__":
    # ── Args ──────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(description="Task-mode inference with training-exact prompts.")
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--mode", choices=list(MODES.keys()), default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--mlx", action="store_true")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────

    if args.mlx:
        from mlx_lm import load, generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler
        model, tokenizer = load(args.model_path)

        def respond(prompt):
            return mlx_generate(model, tokenizer, prompt=prompt,
                                max_tokens=args.max_new_tokens,
                                sampler=make_sampler(temp=0.0), verbose=False)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")

        def respond(prompt):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    # ── Main loop ─────────────────────────────────────────────────────────────

    print(f"\nModel: {args.model_path}")
    print("Enter=newline, Esc+Enter=submit, Ctrl+C=quit\n")

    while True:
        try:
            mode_key = args.mode or pick_mode()
            mode = MODES[mode_key]

            if mode_key == "qa_answer_question_only":
                print("\nPaste the passage first:")
                passage = multiline_input("> ")
                instruction = f"Given the provided passage, answer the user's question. Passage: {passage}"
                print("\nNow enter the question:")
                question = multiline_input("> ")
                user_message = f"{instruction}\n\n{question}"
            else:
                instruction = pick_variant(mode_key)
                print(f"\n{mode.input_hint}")
                text = multiline_input("> ")
                if not text:
                    continue
                user_message = mode.format_user_message(instruction, text)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            print(f"\nAssistant:\n{respond(prompt)}\n")
            print("─" * 60)

        except KeyboardInterrupt:
            print("\nBye.")
            sys.exit(0)
