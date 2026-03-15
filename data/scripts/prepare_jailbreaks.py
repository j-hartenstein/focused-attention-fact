"""Prepare jailbreak (xc, xw) pairs from Mechanistic-Anomaly-Detection/llama3-jailbreaks.

Strategy per split:
  - harmful_autodan:   xw = full prompt; xc = text after "[PROMPT]:" marker (×AUTODAN_MULTIPLY)
  - harmful_msj:       xw = full prompt; xc = last user turn (final harmful question)
  - harmful_human_mt:  xw = full prompt; xc = last user turn (escalated harmful request)
  - harmful_prefill:   xw = full prompt; xc = user content (the Llama prefill IS the wrapper)
  - harmful_gcg:       xw = full prompt; xc = user turn with last 15 tokens removed (GCG suffix)
  - harmful_best_of_n: xw = full prompt; xc = last user turn lowercased as clean single-turn prompt

xc is reformatted as a clean Llama-3 single-turn prompt.
xw is the jailbreak prompt with the same system block (Cutting Knowledge Date, Today Date, etc.)
prepended so clean and wrapped share the same outer structure.

Outputs:
  data/processed/jailbreaks.jsonl
  Each line: {"clean": str, "wrapped": str, "attack": str, "core_text": str}

  "attack" values: "autodan" | "msj" | "prefill" | "human_mt" | "gcg" | "best_of_n"
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
OUT_PATH = Path("data/processed/jailbreaks.jsonl")

# Llama-3 special token constants
LLAMA3_USER_OPEN = "<|start_header_id|>user<|end_header_id|>"
LLAMA3_ASST_OPEN = "<|start_header_id|>assistant<|end_header_id|>"
LLAMA3_EOT = "<|eot_id|>"

# autodan: repeat each row this many times (same xc/xw — augments training set)
AUTODAN_MULTIPLY = 5

# gcg: adversarial suffix length in tokens (strip from end of user turn to get clean prompt)
GCG_SUFFIX_LEN_TOKENS = 20


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def format_clean(tokenizer, instruction: str) -> str:
    """Format a bare instruction as a clean Llama-3 single-turn prompt."""
    messages = [{"role": "user", "content": instruction.strip()}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# Sentinel used to recover the "prefix" (BOS + system block + user header) from the chat template.
_SENTINEL = "\x00"


def get_system_user_prefix(tokenizer) -> str:
    """Return the string that appears before the user message content (BOS + system block + user header + newlines)."""
    full = format_clean(tokenizer, _SENTINEL)
    if _SENTINEL in full:
        return full.split(_SENTINEL, 1)[0]
    # Fallback: no sentinel (e.g. template normalized it); use single-char placeholder.
    full = format_clean(tokenizer, "A")
    idx = full.find("A")
    return full[:idx] if idx >= 0 else ""


def wrap_with_system_prompt(tokenizer, xw: str) -> str:
    """
    Prepend the same system block (Cutting Knowledge Date, Today Date, etc.) to the wrapped prompt.
    xw is assumed to be Llama-3 formatted with <|start_header_id|>user<|end_header_id|> and user content until EOT.
    Returns: prefix (system + user header) + user_content + newlines + tail (EOT + assistant + any prefill).
    """
    if LLAMA3_USER_OPEN not in xw or LLAMA3_EOT not in xw:
        return xw
    parts = xw.split(LLAMA3_USER_OPEN, 1)
    after_user = parts[1]
    idx_eot = after_user.find(LLAMA3_EOT)
    if idx_eot < 0:
        return xw
    user_content = after_user[:idx_eot].strip()
    tail = after_user[idx_eot:]
    prefix = get_system_user_prefix(tokenizer)
    return prefix + user_content + "\n\n" + tail


def extract_last_user_turn(prompt: str) -> str | None:
    """Extract the text of the final user message from a Llama-3 formatted prompt string."""
    parts = prompt.split(LLAMA3_USER_OPEN)
    if len(parts) < 2:
        return None
    last = parts[-1]
    # Drop everything from EOT onward (assistant turn / end of text)
    last = last.split(LLAMA3_EOT)[0]
    last = last.split(LLAMA3_ASST_OPEN)[0]
    return last.strip() or None


def _normalize_fullwidth_to_ascii(s: str) -> str:
    """Convert fullwidth (wide) Unicode characters to ASCII so regex patterns match."""
    # Fullwidth ASCII variant range U+FF01–U+FF5E → U+0021–U+007E
    result = []
    for c in s:
        code = ord(c)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:  # Ideographic space → space
            result.append(" ")
        else:
            result.append(c)
    return "".join(result)


def has_strong_context_dependency(text: str) -> bool:
    """
    Return True when text has strong evidence it depends on prior turns and
    is not self-contained as a standalone clean prompt.
    """
    # Normalize fullwidth Unicode to ASCII so patterns match obfuscated text (e.g. "Ｃｈａｎｇｅ" → "change")
    t = _normalize_fullwidth_to_ascii(text).lower().strip()

    # Very short prompts are often follow-up fragments (e.g. "is code involved?")
    if len(t) < 25:
        return True

    # Start with continuation or context-dependent opener
    if re.search(r"^\s*detail\s+for\b", t):
        return True
    if re.search(r"^\s*make\s+sure\s+(?:it'?s|its)\s+clear\b", t):
        return True
    if re.search(r"^\s*and\b", t):
        return True
    if re.search(r"^\s*we\s+are\b", t):
        return True
    if re.search(r"^\s*refer\s+to\b", t):
        return True

    # Explicit references: "sentences 3-5", "option 2", "first paragraph", "last sentence", "2nd step", etc.
    units = r"sentence|sentences|paragraph|paragraphs|line|lines|step|steps|option|options|slide|slides|side|section|sections|part|parts|item|items|point|points|bullet|bullets|block|blocks|chunk|chunks|segment|segments|passage|passages|excerpt|excerpts"
    # Unit + cardinal number(s): "paragraph 3", "sentences 1-5", "lines 2 to 4"
    if re.search(rf"\b({units})\s+\d+(?:\s*(?:-|to|through)\s*\d+)?\b", t):
        return True
    # Ordinal + unit: "first paragraph", "second sentence", "last option", "next step"
    ordinals = r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|last|next"
    if re.search(rf"\b({ordinals})\s+({units})\b", t):
        return True
    # Ordinal + number + units: "last two paragraphs", "first three sentences", "next two steps"
    if re.search(rf"\b({ordinals})\s+(one|two|three|four|five|\d+)\s+({units})\b", t):
        return True
    # Numeric ordinal + unit: "1st paragraph", "2nd sentence", "3rd option"
    if re.search(rf"\b\d+(?:st|nd|rd|th)\s+({units})\b", t):
        return True
    # Remove/delete/drop + the + (optional ordinal) + optional number + units: "remove the first two sentences", "delete the last paragraph"
    if re.search(
        rf"\b(remove|delete|drop|omit|cut|strip)\s+the\s+(({ordinals})\s+)?(one|two|three|four|five|\d+)\s+({units})\b",
        t,
    ):
        return True
    if re.search(rf"\b(remove|delete|drop|omit|cut|strip)\s+the\s+({ordinals})\s+({units})\b", t):
        return True

    # Common continuation/fix-up cues from prior turns.
    continuation = (
        r"\b(that's|that is|this is|here it is|okay|here's|here is|there's|there is|no i'm sorry|i'm sorry|my bad|ops|oops|now|i think|"
        r"truncated again|ended abruptly|repeat the whole thing|same as above|as above|as before|your idea|just made|"
        r"start over|from scratch|cut off|cut short|got cut|incomplete|partial|again from the top|one more time|reverse|"
        r"same thing|that one|this one|let me redo|i meant|i mean|sure here|from the top)\b"
    )
    if re.search(continuation, t):
        return True

    # Direct deictic references to prior context (single cue is enough).
    deictic = (
        r"\b(the above|from above|as discussed|as mentioned|mentioned|any mention|you mentioned|you forgot|as stated|as said|as noted|as indicated|as requested|as asked|remember|"
        r"earlier|previous(?:ly)?|prior|aforementioned|aforesaid|foregoing|preceding|"
        r"continue where you left off|before that|from before|in the last|from the last)\b"
    )
    if re.search(deictic, t):
        return True

    # Numbered follow-up: "That was #3. Try #2.", "that was option 1", "try number 4"
    if re.search(r"\bthat was\s+(?:#\d+|number\s*\d+|option\s*\d+|\d+)\b", t):
        return True
    if re.search(r"\btry\s+(?:#\d+|number\s*\d+|option\s*\d+)\b", t):
        return True
    # "For #1" / "for #2" — refers to prior numbered item
    if re.search(r"\bfor\s+(?:#\d+|number\s*\d+|option\s*\d+)\b", t):
        return True

    # "X to/on/for/regarding/about this/that/it" — response/answer/thoughts/feedback pointing at prior context.
    action_noun = r"response|answer|reply|reaction|thoughts?|feedback|comment|opinion|view|take|interpretation|summary|analysis|review|critique|thought|reflection|remark"
    pointer = r"this|that|it"
    if re.search(rf"\b({action_noun})\s+(?:to|on|for|regarding|about|concerning)\s+({pointer})\b", t):
        return True
    if re.search(rf"\b(respond|reply|react|comment|expand|elaborate)\s+(?:to|on|about)\s+({pointer})\b", t):
        return True
    # "regarding/about/concerning this/that" at start or as clear reference.
    if re.search(rf"\b(regarding|about|concerning|re|re:)\s+({pointer})\b", t):
        return True

    # Follow-up questions about "it" / prior context: "is there any code involved?", "does that involve X?"
    if re.search(r"\b(?:is there any|are there any|any)\s+\w+\s+involved\b", t):
        return True
    if re.search(r"\binvolved\s+in\s+(this|that|it)\b", t):
        return True
    if re.search(r"\b(?:does|is|was)\s+(it|that)\s+(?:involve|contain|include|have)\b", t):
        return True
    if re.search(r"\b(?:is|are)\s+(it|that)\s+related\s+to\b", t):
        return True

    # "Your" + prior output artifact: "discussing your findings", "expand on your response", "your analysis above"
    your_artifact = r"findings|response|answer|output|analysis|summary|explanation|reasoning|conclusion|draft|version|reply|interpretation|take"
    if re.search(rf"\b(?:discussing|discuss|expand on|elaborate on|continue|based on)\s+your\s+({your_artifact})\b", t):
        return True
    if re.search(rf"\byour\s+({your_artifact})\s+(?:above|below|here|there)\b", t):
        return True

    # Rewrite/copy with reference to prior text: "rewrite word for word", "verbatim", "exactly as (written|above)"
    if re.search(r"\b(word\s+for\s+word|verbatim|exactly\s+as\s+(?:written|stated|above|given)|same\s+(?:wording|words|text))\b", t):
        return True
    if re.search(r"\b(rewrite|reproduce|copy|repeat)\s+(?:it|that|the\s+above)\s+(?:word\s+for\s+word|verbatim|exactly)\b", t):
        return True

    # Reference to content "in the X" / "from the X": "in the article", "from the text", "in the passage"
    in_artifact = r"article|text|passage|document|response|excerpt|paragraph|message|post|email|report|conversation"
    if re.search(rf"\b(?:in|from|from within)\s+the\s+({in_artifact})\b", t):
        return True

    # "The topic" from prior context without specifying it: "on the topic from above", "about the topic we discussed"
    if re.search(r"\b(?:on|about|regarding)\s+the\s+topic\s+from\b", t):
        return True
    if re.search(r"\bthe\s+topic\s+(?:from|we\s+discussed|you\s+mentioned|above|earlier|before)\b", t):
        return True
    if re.search(r"\b(?:same|that|this)\s+topic\s+(?:from|above|we\s+discussed)\b", t):
        return True

    # Continuation-style imperatives: "Now output", "Now write", "Then give me", "Reoutput"
    if re.search(r"\b(now|next|then)\s+(?:output|write|give|provide|produce|return|generate)\b", t):
        return True
    if re.search(r"\bre-?output\b", t):
        return True

    # "Write it / do it" — "it" refers to prior content: "Can you write it in Python?", "Please rewrite it"
    if re.search(
        r"^\s*(?:can you|could you|would you|please|just)\s+(?:write|rewrite|rephrase|convert|translate)\s+it\b",
        t,
    ):
        return True

    # "Fix it" / "correct it" — "it" = prior output: "fix it", "please correct it", "redo it"
    if re.search(
        r"^\s*(?:please\s+|just\s+)?(fix|correct|change|improve|redo|repair|update)\s+it\b",
        t,
    ):
        return True
    if re.search(
        r"\b(fix|correct|redo|repair)\s+it\b",
        t,
    ) and len(t) < 80:
        return True

    # "How do these" / "What are those" — "these"/"those" refer to prior list or content
    if re.search(
        r"^\s*(?:how|what|why|when|where)\s+(?:do|are|did|were|can)\s+(these|those)\b",
        t,
    ):
        return True
    if re.search(
        r"^\s*(?:do|are|did|were)\s+(these|those)\b",
        t,
    ):
        return True
    if re.search(
        r"^\s*which\s+of\s+(these|those)\b",
        t,
    ):
        return True

    # Contains: rewrite/refer to prior items or modify prior content
    if re.search(r"\brewrite\s+those\b", t):
        return True
    if re.search(r"\bprovide\s+a\s+modified\s+example\b", t):
        return True
    if re.search(r"\bremove\s+every\s+instance\s+of\b", t):
        return True
    if re.search(r"\bone\s+more\s+change\b", t):
        return True
    if re.search(r"\bafter\s+.+\s+(?:add|remove)\b", t):
        return True
    if re.search(r"\breplace\s+.+\s+with\b", t):
        return True
    if re.search(r"\bchange\s+.+\s+to\b", t):
        return True
    if re.search(
        r"\bwrite\s+email\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
        t,
    ):
        return True

    # Comparative focus/emphasis — implies adjusting prior content: "Focus more on how it never happened", "Emphasize X more"
    if re.search(
        r"\b(focus|emphasize|stress|highlight|expand|dwell|concentrate|elaborate)\s+more\s+(?:on|about)?\b",
        t,
    ):
        return True
    if re.search(
        r"\b(focus|emphasize|stress|expand|dwell|concentrate)\s+less\s+(?:on|about)?\b",
        t,
    ):
        return True
    if re.search(r"\b(more\s+on|less\s+on|more\s+about|less\s+about)\b", t):
        return True

    # Adjustment imperatives — imply modifying prior output: "Increase realism", "Decrease the length", "Add more detail"
    if re.search(
        r"\b(increase|decrease|reduce|improve|enhance|boost|lower|add\s+more|add\s+less)\s+(?:the\s+)?(realism|detail|length|complexity|clarity|specificity|depth|tone|formality|emphasis)\b",
        t,
    ):
        return True
    if re.search(
        r"\b(more|less)\s+(realism|detail|length|complexity|clarity|specificity|depth|emphasis)\b",
        t,
    ):
        return True

    # Follow-up style opener + editing intent (either opener or edit intent with reference is enough).
    opener = re.search(
        r"^\s*(this|that|it(?:'s| is)?|yes|yeah|ok|okay|great|thanks?|thank you|wow|sure|alright|perfect|good|nice|cool|right|correct|exactly|indeed|absolutely|one more)\b",
        t,
    )
    edit_intent = re.search(
        r"\b(edit|convert|rewrite|rephrase|paraphrase|shorten|expand|change|omit|emphasize|make|modify|alter|adjust|revise|refine|update|adapt|transform)\b",
        t,
    )
    if opener and edit_intent:
        return True

    # Editing instructions: trigger on edit verb + prior artifact, OR edit verb + demonstrative ("the/that/your X").
    edit_verb = re.search(
        r"\b(replace|fix|convert|rewrite|rephrase|paraphrase|shorten|expand|change|omit|emphasize|make|modify|alter|adjust|revise|refine|update|adapt|transform|edit|list)\b",
        t,
    )
    prior_artifact = re.search(
        r"\b(response|answer|reply|output|blog|entry|post|script|draft|version|paragraph|line|lines|sentence|sentences|tone|style|format|document|content|piece|essay|article|section|message)\b",
        t,
    )
    demonstrative_artifact = re.search(
        r"\b(the|that|your|this)\s+(to|response|answer|reply|output|text|draft|acronym|version|paragraph|output|message|content)\b",
        t,
    )
    if edit_verb and (prior_artifact or demonstrative_artifact):
        return True

    # Transform/edit requests: trigger on transform + missing target, OR transform alone in short prompts.
    transform = re.search(
        r"\b(rewrite|rephrase|paraphrase|summari[sz]e|translate|shorten|expand|continue|modify|revise|refine)\b",
        t,
    )
    missing_target = re.search(
        r"\b(report|response|answer|text|conversation|message|details|above|earlier|previous|prior|output|reply|content|that)\b",
        t,
    )
    if transform and missing_target:
        return True
    if transform and len(t) < 120:
        return True

    # Bare transform commands are often context-dependent (e.g., "convert tone to argumentative").
    if re.fullmatch(
        r"\s*(convert|rewrite|rephrase|paraphrase|shorten|expand|translate|modify|revise|refine|adapt|response)\b[\s\w-]{0,80}",
        t,
    ):
        return True

    return False


# ---------------------------------------------------------------------------
# Per-split processors
# ---------------------------------------------------------------------------

def process_autodan(rows, tokenizer) -> list[dict]:
    """
    AutoDAN: long 'Developer Mode' preamble followed by '[PROMPT]: <instruction>'.
    Extract core_text from after the [PROMPT] marker.
    Multiply each row ×AUTODAN_MULTIPLY (same xc, same xw — the universal
    wrapper is fixed, so repeats are useful training augmentation).
    """
    pairs = []
    marker_re = re.compile(r"\[PROMPT\]\s*:?\s*", re.IGNORECASE)

    for row in rows:
        prompt = row["prompt"]

        markers = list(marker_re.finditer(prompt))
        if markers:
            core_text = prompt[markers[-1].end():].strip()
            # Strip any trailing Llama tokens that bled in
            core_text = core_text.split(LLAMA3_EOT)[0].strip()
        else:
            # Fallback: last user turn
            core_text = extract_last_user_turn(prompt)

        if not core_text or len(core_text) < 15:
            continue

        xc = format_clean(tokenizer, core_text)
        xw = wrap_with_system_prompt(tokenizer, prompt)

        if xc.strip() == xw.strip():
            continue  # extraction failed — wrapper and clean are identical

        for _ in range(AUTODAN_MULTIPLY):
            pairs.append({"clean": xc, "wrapped": xw, "attack": "autodan", "core_text": core_text})

    return pairs


def process_msj(rows, tokenizer) -> list[dict]:
    """
    MSJ (many-shot jailbreaking): N fake Q&A pairs prepended to the final harmful question.
    xc = last user turn (the actual harmful instruction).
    xw = full prompt (with all the fake assistant context).
    """
    pairs = []
    for row in rows:
        prompt = row["prompt"]
        core_text = extract_last_user_turn(prompt)
        if not core_text or len(core_text) < 15:
            continue

        xc = format_clean(tokenizer, core_text)
        xw = wrap_with_system_prompt(tokenizer, prompt)
        if xc.strip() == xw.strip():
            continue

        pairs.append({"clean": xc, "wrapped": xw, "attack": "msj", "core_text": core_text})

    return pairs


def process_prefill(rows, tokenizer) -> list[dict]:
    """
    Prefill: the prompt ends with '<|start_header_id|>assistant<|end_header_id|>' with no
    prior assistant EOT, coercing the model to complete from the assistant position.
    The user content is the harmful instruction; the wrapper is the prefill structure itself.
    xc = user content re-wrapped normally.
    xw = original prompt (with the prefill trick).
    """
    pairs = []
    user_content_re = re.compile(
        re.escape(LLAMA3_USER_OPEN) + r"\s*(.*?)(?:" + re.escape(LLAMA3_EOT) + r"|\Z)",
        re.DOTALL,
    )
    for row in rows:
        prompt = row["prompt"]
        m = user_content_re.search(prompt)
        core_text = m.group(1).strip() if m else extract_last_user_turn(prompt)

        if not core_text or len(core_text) < 15:
            continue

        xc = format_clean(tokenizer, core_text)
        xw = wrap_with_system_prompt(tokenizer, prompt)
        if xc.strip() == xw.strip():
            continue

        pairs.append({"clean": xc, "wrapped": xw, "attack": "prefill", "core_text": core_text})

    return pairs


def process_human_mt(rows, tokenizer) -> tuple[list[dict], int]:
    """
    Human MT: multi-turn escalation (human machine translated jailbreaks).
    The full conversation is xw; the last user turn is the terminal harmful instruction (xc).
    """
    pairs = []
    filtered_strong_context = 0
    for row in rows:
        prompt = row["prompt"]
        core_text = extract_last_user_turn(prompt)
        if not core_text or len(core_text) < 15:
            continue
        if has_strong_context_dependency(core_text):
            filtered_strong_context += 1
            continue

        xc = format_clean(tokenizer, core_text)
        xw = wrap_with_system_prompt(tokenizer, prompt)
        if xc.strip() == xw.strip():
            continue

        pairs.append({"clean": xc, "wrapped": xw, "attack": "human_mt", "core_text": core_text})

    return pairs, filtered_strong_context


def process_gcg(rows, tokenizer) -> list[dict]:
    """
    GCG (Greedy Coordinate Gradient) jailbreaks.
    User content has a 15-token adversarial suffix; remove it to get the clean instruction.
    xw = full prompt; xc = clean prompt (user turn with last GCG_SUFFIX_LEN_TOKENS tokens removed).
    """
    pairs = []
    for row in rows:
        prompt = row["prompt"]
        user_content = extract_last_user_turn(prompt)
        if not user_content or len(user_content.strip()) < 10:
            continue

        ids = tokenizer.encode(user_content, add_special_tokens=False)
        if len(ids) <= GCG_SUFFIX_LEN_TOKENS:
            continue
        clean_text = tokenizer.decode(
            ids[:-GCG_SUFFIX_LEN_TOKENS], skip_special_tokens=True
        ).strip()
        if not clean_text or len(clean_text) < 15:
            continue

        xc = format_clean(tokenizer, clean_text)
        xw = wrap_with_system_prompt(tokenizer, prompt)
        if xc.strip() == xw.strip():
            continue
        pairs.append({"clean": xc, "wrapped": xw, "attack": "gcg", "core_text": clean_text})
    return pairs


def process_best_of_n(rows, tokenizer) -> list[dict]:
    """
    Best-of-N jailbreaks.
    xw = full prompt; xc = last user turn lowercased and formatted as clean single-turn prompt.
    """
    pairs = []
    for row in rows:
        prompt = row["prompt"]
        core_text = extract_last_user_turn(prompt)
        if not core_text or len(core_text) < 15:
            continue
        core_text_lower = core_text.lower()
        xc = format_clean(tokenizer, core_text_lower)
        xw = wrap_with_system_prompt(tokenizer, prompt)
        if xc.strip() == xw.strip():
            continue
        pairs.append({"clean": xc, "wrapped": xw, "attack": "best_of_n", "core_text": core_text_lower})
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PROCESSORS: dict[str, object] = {
    "harmful_autodan": process_autodan,
    "harmful_msj": process_msj,
    "harmful_prefill": process_prefill,
    "harmful_human_mt": process_human_mt,
    "harmful_gcg": process_gcg,
    "harmful_best_of_n": process_best_of_n,
}

ATTACK_TO_SPLIT = {
    "autodan": "harmful_autodan",
    "msj": "harmful_msj",
    "prefill": "harmful_prefill",
    "human_mt": "harmful_human_mt",
    "gcg": "harmful_gcg",
    "best_of_n": "harmful_best_of_n",
}


TEST_DIR = Path("data/processed/test")
TEST_N_DEFAULT = 200


def main(args):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    print("Loading dataset: Mechanistic-Anomaly-Detection/llama3-jailbreaks")
    ds = load_dataset("Mechanistic-Anomaly-Detection/llama3-jailbreaks")
    print(f"Available splits: {list(ds.keys())}")

    selected_splits = list(PROCESSORS.keys())
    if args.attacks:
        selected_splits = [ATTACK_TO_SPLIT[a] for a in args.attacks]
        print(f"Selected attacks: {args.attacks}")
        print(f"Selected splits: {selected_splits}")

    all_pairs: list[dict] = []
    filtered_counts: Counter[str] = Counter()

    for split_name in selected_splits:
        processor = PROCESSORS[split_name]
        if split_name not in ds:
            print(f"  [skip] '{split_name}' not in dataset")
            continue

        rows = list(ds[split_name])
        if split_name == "harmful_human_mt":
            pairs, n_filtered = processor(rows, tokenizer)
            filtered_counts["human_mt_strong_context"] += n_filtered
        else:
            pairs = processor(rows, tokenizer)
        print(f"  {split_name}: {len(rows)} rows → {len(pairs)} pairs")
        all_pairs.extend(pairs)

    print(f"\nTotal before sampling: {len(all_pairs)}")

    rng = random.Random(42)
    rng.shuffle(all_pairs)
    if args.n_samples and len(all_pairs) > args.n_samples:
        all_pairs = all_pairs[: args.n_samples]
        print(f"Sampled to: {len(all_pairs)}")
    elif args.n_samples and len(all_pairs) < args.n_samples:
        print(f"Requested {args.n_samples} samples, but only {len(all_pairs)} valid pairs were available.")

    # Hold out exactly n_test for test set (or all if fewer available)
    n_test = min(args.n_test, len(all_pairs))
    test_pairs = all_pairs[:n_test]
    train_pairs = all_pairs[n_test:]

    with open(OUT_PATH, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")

    test_path = TEST_DIR / "jailbreaks.jsonl"
    with open(test_path, "w") as f:
        for p in test_pairs:
            f.write(json.dumps(p) + "\n")

    train_counts = Counter(p["attack"] for p in train_pairs)
    print(f"\nWrote {len(train_pairs)} pairs to {OUT_PATH}")
    for attack, n in sorted(train_counts.items()):
        print(f"  {attack:20s}: {n}")
    if filtered_counts["human_mt_strong_context"] > 0:
        print(f"  filtered_human_mt_strong_context: {filtered_counts['human_mt_strong_context']}")

    test_counts = Counter(p["attack"] for p in test_pairs)
    print(f"\nWrote {len(test_pairs)} pairs to {test_path}")
    for attack, n in sorted(test_counts.items()):
        print(f"  {attack:20s}: {n}")

    # Sanity-check a few examples
    if args.preview and train_pairs:
        print("\n--- Preview (first 2 train pairs) ---")
        for p in train_pairs[:2]:
            print(f"attack: {p['attack']}")
            print(f"core:   {p['core_text'][:120]}")
            print(f"xc:     {p['clean'][:200]}")
            print(f"xw:     {p['wrapped'][:200]}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=sorted(ATTACK_TO_SPLIT.keys()),
        help="Optional subset of attacks to include (e.g., --attacks human_mt). Default: all.",
    )
    parser.add_argument(
        "--n-test", type=int, default=TEST_N_DEFAULT,
        help=f"Number of pairs to hold out as test set (default {TEST_N_DEFAULT}).",
    )
    parser.add_argument("--preview", action="store_true",
                        help="Print a few examples after processing")
    args = parser.parse_args()
    main(args)
