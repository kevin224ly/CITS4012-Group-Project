"""Generate architecture diagrams for each model in the CITS4012 project.

The specification requires clear, per-model diagrams that match the notebook
implementations. This script outputs three vertical PNG diagrams that can be
embedded in a two-column report without further editing.
"""

from __future__ import annotations

from pathlib import Path

import graphviz

OUTPUT_DIR = Path("diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "input": "#d8efff",
    "embedding": "#ffe0b2",
    "encoder": "#e0d4ff",
    "attention": "#fff2b3",
    "fusion": "#d0f0e0",
    "pool": "#c8e6c9",
    "mlp": "#b2dfdb",
    "output": "#90a4ae",
    "support": "#f5f5f5",
}


def add_node(graph: graphviz.Digraph, name: str, label: str, fill: str) -> None:
    graph.node(
        name,
        label,
        shape="box",
        style="filled,rounded",
        fillcolor=fill,
        color="#1f3a5f",
        fontname="Helvetica",
        fontsize="10",
        margin="0.05,0.035",
    )


def build_model_a_graph() -> graphviz.Digraph:
    dot = graphviz.Digraph("ModelA", format="png")
    dot.attr(rankdir="TB", fontname="Helvetica", fontsize="11", ranksep="0.3", nodesep="0.25")
    dot.attr("graph", bgcolor="white", margin="0.15", pad="0.0", label="Model A: BiLSTM + Bilinear Cross-Attention", labelloc="t")
    dot.attr("node", fontname="Helvetica", fontsize="10", margin="0.05,0.035")
    dot.attr("edge", arrowsize="0.7", penwidth="1.0")

    with dot.subgraph() as top:
        top.attr(rank="same")
        add_node(top, "A_InputPrem", "Input - Premise\nTokens p1 ... pLp", PALETTE["input"])
        add_node(top, "A_InputHyp", "Input - Hypothesis\nTokens h1 ... hLh", PALETTE["input"])

    add_node(dot, "A_Embedding", "Embedding Layer\nWord2Vec (200d, trainable)", PALETTE["embedding"])

    with dot.subgraph() as encoders:
        encoders.attr(rank="same")
        add_node(
            encoders,
            "A_EncPrem",
            "BiLSTM Encoder (Premise)\n2 x 128 hidden, return sequences",
            PALETTE["encoder"],
        )
        add_node(
            encoders,
            "A_EncHyp",
            "BiLSTM Encoder (Hypothesis)\n2 x 128 hidden, return sequences",
            PALETTE["encoder"],
        )

    add_node(
        dot,
        "A_CrossAttn",
        "Bilinear Cross-Attention\nalpha = softmax(QP KH^T / sqrt(d))\n+ attention fetcher",
        PALETTE["attention"],
    )

    with dot.subgraph() as fusion:
        fusion.attr(rank="same")
        add_node(
            fusion,
            "A_FusionPrem",
            "Feature Fusion (Premise)\n[Hp, Cp, |Hp-Cp|, Hp*Cp]",
            PALETTE["fusion"],
        )
        add_node(
            fusion,
            "A_FusionHyp",
            "Feature Fusion (Hypothesis)\n[Hh, Ch, |Hh-Ch|, Hh*Ch]",
            PALETTE["fusion"],
        )

    with dot.subgraph() as pooling:
        pooling.attr(rank="same")
        add_node(pooling, "A_PoolPrem", "Pooling (Premise)\nGlobal max + mean -> vp", PALETTE["pool"])
        add_node(pooling, "A_PoolHyp", "Pooling (Hypothesis)\nGlobal max + mean -> vh", PALETTE["pool"])

    add_node(dot, "A_Concat", "Concatenate\nv = [vp, vh]", PALETTE["fusion"])
    add_node(dot, "A_MLP", "Classifier MLP\nDense(256, ReLU) + Dropout(0.3)", PALETTE["mlp"])
    add_node(dot, "A_Output", "Output Layer\nDense(2, softmax)", PALETTE["output"])

    dot.edge("A_InputPrem", "A_Embedding")
    dot.edge("A_InputHyp", "A_Embedding")
    dot.edge("A_Embedding", "A_EncPrem")
    dot.edge("A_Embedding", "A_EncHyp")
    dot.edge("A_EncPrem", "A_CrossAttn")
    dot.edge("A_EncHyp", "A_CrossAttn")
    dot.edge("A_CrossAttn", "A_FusionPrem")
    dot.edge("A_CrossAttn", "A_FusionHyp")
    dot.edge("A_FusionPrem", "A_PoolPrem")
    dot.edge("A_FusionHyp", "A_PoolHyp")
    dot.edge("A_PoolPrem", "A_Concat")
    dot.edge("A_PoolHyp", "A_Concat")
    dot.edge("A_Concat", "A_MLP")
    dot.edge("A_MLP", "A_Output")

    add_node(
        dot,
        "A_Training",
        "Training Notes\n- Optimiser: Adam (lr=2e-3)\n- Batch size: 64 (5 epochs)\n- Uses validation metrics for reporting",
        PALETTE["support"],
    )
    dot.edge("A_Output", "A_Training", style="dashed", color="#455a64")

    return dot


def build_model_b_graph() -> graphviz.Digraph:
    dot = graphviz.Digraph("ModelB", format="png")
    dot.attr(rankdir="TB", fontname="Helvetica", fontsize="11", ranksep="0.3", nodesep="0.25")
    dot.attr("graph", bgcolor="white", margin="0.15", pad="0.0", label="Model B: ESIM-Style BiGRU", labelloc="t")
    dot.attr("node", fontname="Helvetica", fontsize="10", margin="0.05,0.035")
    dot.attr("edge", arrowsize="0.7", penwidth="1.0")

    add_node(
        dot,
        "B_Input",
        "Input Pair\nPremise / Hypothesis token sequences",
        PALETTE["input"],
    )
    add_node(
        dot,
        "B_Embedding",
        "Embedding Layer\nTrainable embeddings (shared vocab)",
        PALETTE["embedding"],
    )
    add_node(
        dot,
        "B_Encoder",
        "Shared BiGRU Encoder\nHidden size 300, bidirectional",
        PALETTE["encoder"],
    )
    add_node(
        dot,
        "B_SoftAlign",
        "Soft Alignment + Local Inference\nAttention weights + elementwise comparisons",
        PALETTE["attention"],
    )
    add_node(
        dot,
        "B_Compose",
        "Composition BiGRU\nProcesses enhanced inference signals",
        PALETTE["encoder"],
    )
    add_node(
        dot,
        "B_Pooling",
        "Pooling\nGlobal max + mean -> sentence representation",
        PALETTE["pool"],
    )
    add_node(
        dot,
        "B_MLP",
        "Classifier Head\nDropout(0.3) + MLP (ReLU)\nProduces logits",
        PALETTE["mlp"],
    )
    add_node(
        dot,
        "B_Output",
        "Output Layer\nSoftmax -> {entails, neutral}",
        PALETTE["output"],
    )

    dot.edges(
        [
            ("B_Input", "B_Embedding"),
            ("B_Embedding", "B_Encoder"),
            ("B_Encoder", "B_SoftAlign"),
            ("B_SoftAlign", "B_Compose"),
            ("B_Compose", "B_Pooling"),
            ("B_Pooling", "B_MLP"),
            ("B_MLP", "B_Output"),
        ]
    )

    add_node(
        dot,
        "B_Notes",
        "Training Notes\n- Optimiser: AdamW + scheduler\n- Batch size: 64, early stopping\n- Exposes alignment matrices for plots",
        PALETTE["support"],
    )
    dot.edge("B_Output", "B_Notes", style="dashed", color="#455a64")

    return dot


def build_model_c_graph() -> graphviz.Digraph:
    dot = graphviz.Digraph("ModelC", format="png")
    dot.attr(rankdir="TB", fontname="Helvetica", fontsize="11", ranksep="0.3", nodesep="0.25")
    dot.attr("graph", bgcolor="white", margin="0.15", pad="0.0", label="Model C: Transformer Cross-Encoder", labelloc="t")
    dot.attr("node", fontname="Helvetica", fontsize="10", margin="0.05,0.035")
    dot.attr("edge", arrowsize="0.7", penwidth="1.0")

    add_node(
        dot,
        "C_Input",
        "Input Sequence\n[CLS] premise [SEP] hypothesis",
        PALETTE["input"],
    )
    add_node(
        dot,
        "C_Emb",
        "Embedding Sum\nToken + Segment + Position (dim=256)",
        PALETTE["embedding"],
    )
    add_node(
        dot,
        "C_Encoder",
        "Transformer Encoder\n3 layers, 4 heads, FFN dim=512",
        PALETTE["encoder"],
    )
    add_node(
        dot,
        "C_Cls",
        "[CLS] Representation\nLayerNorm + dropout",
        PALETTE["attention"],
    )
    add_node(
        dot,
        "C_MLP",
        "Classification Head\nLinear -> GELU -> Linear",
        PALETTE["mlp"],
    )
    add_node(
        dot,
        "C_Output",
        "Output Layer\nSoftmax probabilities",
        PALETTE["output"],
    )

    dot.edges(
        [
            ("C_Input", "C_Emb"),
            ("C_Emb", "C_Encoder"),
            ("C_Encoder", "C_Cls"),
            ("C_Cls", "C_MLP"),
            ("C_MLP", "C_Output"),
        ]
    )

    add_node(
        dot,
        "C_Notes",
        "Training Notes\n- Batch size 32, length 256\n- AdamW + linear warmup (10%)\n- Gradient clipping & best-checkpoint export",
        PALETTE["support"],
    )
    dot.edge("C_Output", "C_Notes", style="dashed", color="#455a64")

    return dot


def main() -> None:
    builders = {
        "modelA_architecture": build_model_a_graph,
        "modelB_architecture": build_model_b_graph,
        "modelC_architecture": build_model_c_graph,
    }
    for filename, builder in builders.items():
        dot = builder()
        path = OUTPUT_DIR / filename
        dot.render(path.as_posix(), cleanup=True)
        print(f"Saved {path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
