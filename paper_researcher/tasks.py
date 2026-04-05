# Exact instruction strings from training data (text_albumentations tasks).
# These must not be changed — the model was trained on these exact prompts.

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """

# ── Bullets ───────────────────────────────────────────────────────────────────

BULLETS_INSTRUCTION = (
    "Extract the important points from this passage as markdown bullet points."
)

# ── QA pairs ──────────────────────────────────────────────────────────────────

QA_PAIRS_INSTRUCTION = (
    "\nGiven this passage of text, generate a list of important question answer pairs.\n    "
)

QUESTION_FROM_PASSAGE_INSTRUCTION = "Generate a question from this passage"

FACT_FROM_PASSAGE_INSTRUCTION = (
    "Generate an important fact or piece of information from this passage"
)

# ── QA answering ──────────────────────────────────────────────────────────────

QA_ANSWER_INSTRUCTION = "Answer the user's question given the provided passage"

def build_qa_answer_input(passage: str, question: str) -> str:
    return f"Passage: {passage}\n\nQuestion: {question}\nWhat is the answer?"

# ── Rephrase ──────────────────────────────────────────────────────────────────

REPHRASE_INSTRUCTION = (
    "\nGiven this passage, rephrase it. Elaborate on the sentences by explaining the meaning. "
    "Only present content that is strictly present in the passage, do not introduce new concepts "
    "outside the scope of this input. Do not re-quote the original. Only generate answers.\n    "
)

# ── Continuation ──────────────────────────────────────────────────────────────

CONTINUATION_INSTRUCTION = (
    "You are given the beginning of a passage. "
    "Continue the passage by generating all remaining text after the provided beginning. "
    "Do not repeat the provided beginning."
)

# ── Triplets ──────────────────────────────────────────────────────────────────

TRIPLETS_INSTRUCTION = (
    "Extract knowledge graph triplets from this passage and return them as JSON. Return a JSON array of subject-relation-object triplets supported by this passage"
)

# ── Comparison ────────────────────────────────────────────────────────────────

COMPARISON_INSTRUCTION = (
    "\nGiven 2 passages of text, generate a detailed comparison of the two\n    "
)

def build_comparison_input(passage_a: str, passage_b: str) -> str:
    return f"Passage 1:\n{passage_a}\n\nPassage 2:\n{passage_b}"

# ── Retrieval ─────────────────────────────────────────────────────────────────

RETRIEVAL_INSTRUCTION = (
    "Read the passages and identify which passage answers the question. "
    "Return the passage number and a short justification in markdown."
)

def build_retrieval_input(question: str, passages: list[str]) -> str:
    formatted = "\n\n".join(
        f"Passage {i+1}:\n{p}" for i, p in enumerate(passages)
    )
    return f"{formatted}\n\nQuestion: {question}"
