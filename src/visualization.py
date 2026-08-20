import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import PLOTS_DIR

def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 14

def plot_hallucination_comparison(metrics_dict: dict, save_path: Path = None):
    """
    Bar chart comparing Hallucination Rate, Faithfulness, and Accuracy across configurations.
    """
    set_plot_style()
    save_path = save_path or (PLOTS_DIR / "hallucination_reduction.png")
    
    systems = list(metrics_dict.keys())
    hallucination_rates = [metrics_dict[s]["hallucination_rate_pct"] for s in systems]
    faithfulness = [metrics_dict[s]["avg_faithfulness_pct"] for s in systems]
    accuracy = [metrics_dict[s]["accuracy_score_pct"] for s in systems]
    
    x = np.arange(len(systems))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    rects1 = ax.bar(x - width, hallucination_rates, width, label="Hallucination Rate (%)", color="#e74c3c", alpha=0.9)
    rects2 = ax.bar(x, faithfulness, width, label="Faithfulness (%)", color="#2ecc71", alpha=0.9)
    rects3 = ax.bar(x + width, accuracy, width, label="Factual Accuracy (%)", color="#3498db", alpha=0.9)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Impact of RAG on Hallucination Rate, Faithfulness & Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontweight="semibold")
    ax.set_ylim(0, 110)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Attach labels above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved {save_path}")


def plot_category_breakdown(df_results: pd.DataFrame, save_path: Path = None):
    """
    Category-wise hallucination and faithfulness comparison for Baseline vs RAG Top-3.
    """
    set_plot_style()
    save_path = save_path or (PLOTS_DIR / "faithfulness_by_category.png")
    
    categories = df_results["category"].unique()
    cats_display = [c.replace("_", " ") for c in categories]
    
    base_halluc = []
    rag_halluc = []
    base_faith = []
    rag_faith = []

    for cat in categories:
        sub_df = df_results[df_results["category"] == cat]
        
        # Baseline
        base_h = (sub_df["baseline_hallucinated"].sum() / len(sub_df)) * 100
        base_f = (sub_df["baseline_faithfulness"].mean()) * 100
        
        # RAG Top-3
        rag_h = (sub_df["rag_k3_hallucinated"].sum() / len(sub_df)) * 100
        rag_f = (sub_df["rag_k3_faithfulness"].mean()) * 100
        
        base_halluc.append(base_h)
        rag_halluc.append(rag_h)
        base_faith.append(base_f)
        rag_faith.append(rag_f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    x = np.arange(len(cats_display))
    width = 0.35

    # Subplot 1: Hallucination Rate by Category
    axes[0].bar(x - width/2, base_halluc, width, label="Baseline (No RAG)", color="#e74c3c", alpha=0.85)
    axes[0].bar(x + width/2, rag_halluc, width, label="RAG (Top-3 Strict)", color="#27ae60", alpha=0.85)
    axes[0].set_title("Hallucination Rate by Query Category")
    axes[0].set_ylabel("Hallucination Rate (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cats_display, rotation=15, ha="right", fontweight="semibold")
    axes[0].set_ylim(0, 110)
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Subplot 2: Faithfulness by Category
    axes[1].bar(x - width/2, base_faith, width, label="Baseline (No RAG)", color="#95a5a6", alpha=0.85)
    axes[1].bar(x + width/2, rag_faith, width, label="RAG (Top-3 Strict)", color="#2980b9", alpha=0.85)
    axes[1].set_title("Faithfulness / Groundedness by Category")
    axes[1].set_ylabel("Average Faithfulness (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats_display, rotation=15, ha="right", fontweight="semibold")
    axes[1].set_ylim(0, 110)
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved {save_path}")


def plot_ablation_comparison(metrics_dict: dict, save_path: Path = None):
    """
    Ablation chart comparing Top-3, Top-5, and Strict vs Loose Grounding.
    """
    set_plot_style()
    save_path = save_path or (PLOTS_DIR / "top_k_ablation.png")

    configs = ["RAG (Top-3 Strict)", "RAG (Top-5 Strict)", "RAG (Top-3 Loose)"]
    h_rates = [metrics_dict[c]["hallucination_rate_pct"] for c in configs if c in metrics_dict]
    accuracies = [metrics_dict[c]["accuracy_score_pct"] for c in configs if c in metrics_dict]

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    x = np.arange(len(configs))
    width = 0.35

    rects1 = ax1.bar(x - width/2, h_rates, width, label="Hallucination Rate (%)", color="#e67e22", alpha=0.9)
    rects2 = ax1.bar(x + width/2, accuracies, width, label="Factual Accuracy (%)", color="#16a085", alpha=0.9)

    ax1.set_ylabel("Score (%)")
    ax1.set_title("Ablation Study: Context Window (Top-K) & Grounding Prompt Strictness")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontweight="semibold")
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax1.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved {save_path}")
