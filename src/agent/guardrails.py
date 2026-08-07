"""Checks run either side of a model call.

Two things reach a prompt that this code did not write: the question the user
typed, and the filing text the retriever returned. The first is a direct
injection surface, the second an indirect one — a document can carry an
instruction just as easily as a person can.

Nothing here claims to stop a determined attacker. Pattern matching catches the
obvious phrasings and misses a paraphrase. What it does give is a record: every
check writes its finding into the trace, so a run that saw something odd says so
rather than failing quietly. The one deterministic check is on the way out —
a citation either points at a passage the retriever returned or it does not.
"""

import re

# Phrasings that only appear when text is addressing the model rather than
# describing a business. A 10-K does not tell anyone to disregard anything.
INSTRUCTION_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\b",
    r"\bdisregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|instructions)\b",
    r"\b(system|developer)\s*(prompt|message|instruction)",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\s+(a|an|if)\b",
    r"\bnew\s+instructions?\b",
    r"\boverride\s+(your|the)\b",
    r"\breveal\s+(your|the)\s+(prompt|instructions|system)",
    r"```\s*system",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INSTRUCTION_PATTERNS]


def screen(text: str) -> list[str]:
    """
    Looks for text that is addressing the model rather than answering it.

    Args:
        text (str): Either a user's question or a passage from a filing.

    Returns:
        list[str]: The matched phrases, empty when nothing looks like an
            instruction. Callers record these; they do not have to act on them.
    """
    return [m.group(0) for p in _COMPILED for m in p.finditer(text or "")]


def fence(label: str, body: str) -> str:
    """
    Wraps retrieved text so the model can tell a passage from an instruction.

    Delimiting is the part of this file that actually earns its place. Without
    it a filing's prose and the prompt's rules arrive as one undifferentiated
    block, and the model has no way to know which is which.

    Args:
        label (str): The citation label for the passage, e.g. "chunk 157, ...".
        body (str): The passage text itself, straight from the index.

    Returns:
        str: The passage inside explicit begin/end markers.
    """
    return f"<<<PASSAGE {label}>>>\n{body}\n<<<END PASSAGE>>>"


def inspect_output(answer: str) -> list[str]:
    """
    Looks for signs that the model repeated its input instead of answering from it.

    Two things should never reach the user. The fence markers are ours, so seeing
    one means the prompt itself is being echoed back. Instruction-shaped phrasing
    in an answer means text that was meant to be read as evidence has been carried
    through into the output.

    Args:
        answer (str): The generated answer, before it is shown to anyone.

    Returns:
        list[str]: What was found, empty when the answer looks clean.
    """
    found = []
    if "<<<PASSAGE" in (answer or "") or "<<<END PASSAGE" in (answer or ""):
        found.append("prompt markers echoed")
    found.extend(screen(answer))
    return found


def check_citations(answer: str, retrieved_ids: set[int]) -> tuple[int, set[int]]:
    """
    Counts an answer's citations and finds any the retriever never returned.

    The count matters as much as the mismatch. An answer that cites nothing at
    all passes a "do the citations resolve?" test trivially while being exactly
    as ungrounded as one that invents them.

    Args:
        answer (str): The generated answer, citing passages as "[chunk 157, ...]".
        retrieved_ids (set[int]): The chunk ids actually available to the answer.

    Returns:
        tuple[int, set[int]]: How many citations were made, and the cited ids
            with no matching passage.
    """
    cited = {int(n) for n in re.findall(r"\[chunk (\d+)", answer or "")}
    return len(cited), cited - retrieved_ids
