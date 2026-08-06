"""Draws the agent's state graph for the presentation deck.

Boxes and arrows are placed explicitly rather than by a layout algorithm, so
the retry loop stays readable instead of being hidden under the forward path.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BOX_W, BOX_H = 2.9, 1.5
FILL = "lightblue"
DECIDE_FILL = "#ffd8a8"
DYNAMIC = "#c1440e"

# Node centres. The main path runs along one row; the user sits below it,
# because two different nodes can hand control back to a person.
NODES = {
    "input": (1.7, 7.0, "User\nInput"),
    "plan": (5.4, 7.0, "Plan\n(decompose)"),
    "clarify": (9.1, 7.0, "Clarify\n(DuckDB)"),
    "retrieve": (12.8, 7.0, "Retrieve\n(FAISS + BM25)"),
    "decide": (16.5, 7.0, "Decide\n(choose action)"),
    "answer": (20.2, 7.0, "Answer\n(synthesize + cite)"),
    "end": (23.4, 7.0, "END"),
    "user": (11.6, 2.2, "User\n(clarifies)"),
}


def box(ax, key):
    """Draws one node and returns its centre."""
    x, y, label = NODES[key]
    fill = DECIDE_FILL if key == "decide" else FILL
    w = BOX_W * 0.62 if key == "end" else BOX_W
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - BOX_H / 2), w, BOX_H,
            boxstyle="round,pad=0.06,rounding_size=0.15",
            facecolor=fill, edgecolor="black", linewidth=1.4,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=10.5, fontweight="bold")
    return x, y


def arrow(ax, start, end, rad=0.0, dashed=False, label=None, label_pos=None,
          label_offset=(0, 0.42)):
    """Draws an edge between two points, optionally curved and labelled."""
    colour = DYNAMIC if dashed else "dimgray"
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=18,
            color=colour, linewidth=1.6,
            linestyle="--" if dashed else "-",
            shrinkA=2, shrinkB=2,
        )
    )
    if label:
        lx, ly = label_pos or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(lx + label_offset[0], ly + label_offset[1], label,
                ha="center", va="center", fontsize=9,
                color=colour if dashed else "black")


def create_agent_graph():
    """Renders the state graph to agent_graph.png."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for key in NODES:
        box(ax, key)

    def right(key):
        x, y, _ = NODES[key]
        w = BOX_W * 0.62 if key == "end" else BOX_W
        return (x + w / 2, y)

    def left(key):
        x, y, _ = NODES[key]
        w = BOX_W * 0.62 if key == "end" else BOX_W
        return (x - w / 2, y)

    def top(key):
        x, y, _ = NODES[key]
        return (x, y + BOX_H / 2)

    def bottom(key):
        x, y, _ = NODES[key]
        return (x, y - BOX_H / 2)

    # The straight path: what happens when nothing needs correcting.
    arrow(ax, right("input"), left("plan"))
    arrow(ax, right("plan"), left("clarify"), label="sub-queries")
    arrow(ax, right("clarify"), left("retrieve"), label="1 match\n(clear)",
          label_offset=(0, 1.35))
    arrow(ax, right("retrieve"), left("decide"), label="chunks")
    arrow(ax, right("decide"), left("answer"), label="answer")
    arrow(ax, right("answer"), left("end"), label="response")

    # Handing control back to the person, and picking it up again.
    arrow(ax, bottom("clarify"), (NODES["user"][0] - 0.9, NODES["user"][1] + BOX_H / 2),
          rad=-0.15, label="> 1 match\n(ambiguous)", label_pos=(9.3, 4.4),
          label_offset=(-0.6, 0))
    arrow(ax, (NODES["user"][0] - BOX_W / 2, NODES["user"][1]), bottom("plan"),
          rad=-0.2, label="new context", label_pos=(6.4, 4.0), label_offset=(-0.7, 0))

    # The actions the Decide node can choose instead of answering.
    arrow(ax, top("decide"), top("retrieve"), rad=0.55, dashed=True,
          label="refine / broaden  (rewrite the search, retry)",
          label_pos=(14.65, 9.05), label_offset=(0, 0))
    arrow(ax, bottom("decide"), (NODES["user"][0] + BOX_W / 2, NODES["user"][1]),
          rad=0.25, dashed=True, label="ask_user",
          label_pos=(17.9, 3.5), label_offset=(0, 0))

    ax.text(21.3, 4.9, "model chooses one of four actions;\nretry budget enforced in code",
            ha="center", va="top", fontsize=8.5, style="italic", color=DYNAMIC)

    ax.set_title("Agentic RAG State Graph", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlim(0, 25.2)
    ax.set_ylim(0.8, 10.0)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("agent_graph.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    create_agent_graph()
