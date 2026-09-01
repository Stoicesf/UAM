# -*- coding: utf-8 -*-
"""Final submission polish for AC_DSGF_EN.md (UTF-8 safe)."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(r"f:\UAM")
EN = ROOT / "paper" / "ac_dsgf" / "AC_DSGF_EN.md"
TAB = ROOT / "paper" / "tables" / "table_comm_pareto_points.csv"
FIG = ROOT / "paper" / "ac_dsgf" / "figures"
FIG_CN = ROOT / "paper" / "ac_dsgf_cn" / "figures"


def patch_en(text: str) -> str:
    # Title
    text = text.replace(
        "# Constraint-aware Adaptive Communication Topology Optimization for Multi-UAV Swarm Systems",
        "# Learning Adaptive Communication Topologies for Communication-Constrained UAV Swarm Coordination",
    )

    # Abstract — soft-budget / sparsity story
    old_abs = """Existing multi-UAV communication strategies typically rely on predefined or manually designed communication structures, limiting adaptability under dynamic mission conditions. Although attention-based methods improve information aggregation, the learned weights do not directly determine the communication topology under explicit communication constraints. We formulate communication topology construction as a constrained optimization problem and propose **AC-DSGF** for task-aware sparse topology selection via Candidate Edge Scoring, Budget-Constrained Edge Selection, and Residual Recovery. A communication scaling bound under a degree constraint shows \(\\eta_N=\\mathcal{O}(1/N)\). AC-DSGF achieves competitive task performance under identical communication budgets while maintaining scalability across different swarm sizes."""
    new_abs = """Existing multi-UAV swarm policies often treat communication topology as fixed infrastructure (fully connected, radius, or \(k\)-nearest graphs). Attention weights improve message aggregation but do not decide whether a link should be activated under bandwidth limits—messages may still be transmitted before weighting. We study *when and with whom* UAVs should communicate under soft budget constraints, and propose **AC-DSGF**, which learns adaptive communication sparsity patterns via Candidate Edge Scoring, Budget-Constrained Edge Selection, and Residual Recovery. A degree-constrained scaling bound shows \(\\eta_N=\\mathcal{O}(1/N)\). Under identical communication budgets, AC-DSGF achieves competitive coordination while inducing substantially lower Soft Communication Activation Mass (SCAM) across swarm scales."""
    if old_abs in text:
        text = text.replace(old_abs, new_abs)

    # Introduction rewrite
    old_intro = """### 1.1 Background: topology as a decision factor

In large UAV swarms, communication is limited not only by bandwidth but also by dynamic topology, energy, occlusion, and link instability. Hence the **communication structure itself** is a first-class factor for cooperative performance (*dynamic topology*, *communication budget*, *scalability*)—beyond the vague claim that “communication is expensive.”
### 1.2 Three gaps

**Gap 1 — Predefined topology lacks adaptability.**  
Geometric fully connected / radius / KNN graphs set \(A_t=f(x_t)\) and do not co-optimize with the task under dynamic mission conditions (not framed as “fixed graphs cause high cost.”

**Gap 2 — Attention weighting ≠ topology optimization.**  
Although attention-based methods improve information aggregation, the learned weights do not directly determine the communication topology under explicit communication constraints (*who is important* ≠*who should communicate*).

**Gap 3 — Joint topology–task optimization.**  
> Communication topology should be optimized jointly with task objectives under explicit budgets \(C(\\mathcal{G}_t)\\le B\).

![Fig.0 Motivation](figures/Fig0_motivation_topology.png)

**Fig. 0** (a) Fixed geometric topology; (b) attention fusion (importance ≠ who should communicate); (c) proposed constrained adaptive topology \(G_t\) under budget.

### 1.3 Approach and contributions

AC-DSGF jointly learns edge decisions and control, sparsifies under budget, and uses residual guidance as a safety fallback; theory and identical-budget experiments close the loop from formulation to topology adaptation, scaling, and robustness.

1. **Problem:** We formulate adaptive communication topology selection as a constrained optimization problem for multi-UAV systems.  
2. **Method:** We propose AC-DSGF, a constraint-aware dynamic sparse graph framework that jointly considers task relevance and communication budgets (Candidate Edge Scoring → Budget-Constrained Edge Selection → Residual Recovery).  
3. **Analysis & validation:** We provide scalability analysis and extensive evaluations demonstrating effective topology adaptation under varying swarm scales and conditions."""

    new_intro = """### 1.1 Layer 1 — Topology is treated as fixed infrastructure

Multi-UAV swarms commonly rely on **predefined** communication structures—fully connected, radius graphs, or \(k\)-nearest neighbors—where
\[
A_t=f(x_t)
\]
is fixed by geometry. In this view, topology is *infrastructure*, not a decision variable.

### 1.2 Layer 2 — Attention is not communication activation

Many MARL methods learn attention weights \(\\alpha_{ij}\) for **message aggregation**. Aggregation weighting answers *who is important* for fusion, but typically assumes messages are already available. Under bandwidth limits, the operational question is whether a link should be **activated** at all (*who should communicate*).

### 1.3 Layer 3 — Soft budgets in deployment

Bandwidth, energy, and interference force soft communication budgets. We therefore ask:

> **When and with whom should UAVs communicate?**

![Fig.0 Motivation](figures/Fig0_motivation_topology.png)

**Fig. 0** (a) Fixed geometric topology as infrastructure; (b) attention fusion (importance ≠ activation); (c) learned adaptive sparsity pattern \(G_t\) under soft budget.

### 1.4 Approach and contributions

AC-DSGF learns adaptive communication sparsity under soft budgets via Candidate Edge Scoring, Budget-Constrained Edge Selection, and Residual Recovery, with training surrogate \(\\mathbb{E}[R-\\lambda_c C]\).

1. **Problem:** We formulate adaptive communication topology learning as a joint optimization between swarm task performance and communication activation cost.  
2. **Method:** We propose a differentiable topology adaptation mechanism with budget regularization and residual guidance (AC-DSGF).  
3. **Validation:** We show comparable coordination performance with substantially lower communication activation (SCAM) under varying swarm scales and budgeted deployments."""

    if old_intro in text:
        text = text.replace(old_intro, new_intro)
    else:
        print("WARN: intro block not found exactly")

    # Prop.3 soften
    old_p3 = """#### Prop.3 — Topology Optimality Interpretation

Define a task-aware edge utility \(u_{ij}=\\Delta R_{ij}\) (marginal task contribution of link \(i\\!\\to\\!j\)). Learned scores \(s_{ij}\) (and gates \(g_{ij}=\\sigma(s_{ij})\)) are interpreted as a surrogate for the constrained selection problem
\\[
\\max_{g}\\sum_{ij}g_{ij}u_{ij}
\\quad\\mathrm{s.t.}\\quad
\\sum_{ij}g_{ij}\\le B,\\quad g_{ij}\\in[0,1].
\\]
Thus Stage~2 realizes *task-aware edge selection under budget*, rather than post-hoc magnitude pruning of a fixed graph."""

    new_p3 = """#### Prop.3 — Interpretation of Adaptive Topology Learning

Training minimizes a soft-budget surrogate \(\\mathbb{E}[R-\\lambda_c C]\) with differentiable gates \(g_{ij}=f_\\theta(h_i,h_j,\\cdot)\). We do **not** claim an explicit combinatorial solver for
\\[
\\max_{g}\\sum_{ij}g_{ij}u_{ij}
\\quad\\mathrm{s.t.}\\quad
\\sum_{ij}g_{ij}\\le B.
\\]
Instead, the learned gating mechanism can be **interpreted as an approximate, induced** solution to a budget-constrained edge-selection problem: Stage~2 ranks candidates by learned scores and applies budgeted selection, rather than post-hoc magnitude pruning of a fixed graph."""

    if old_p3 in text:
        text = text.replace(old_p3, new_p3)
    else:
        print("WARN: prop3 not found exactly")

    # Hard Top-K narrative
    old_topk = """Under matched \\(|E_i|\\le K\\), Distance and AC are comparable (both above Random). Nearest-neighbor geometry is a strong prior in navigation; learned scores remain competitive while additionally enabling the soft SCAM regime (Table~1), where activation mass is orders of magnitude lower than dense baselines. The edge-importance ablation (Fig.~12) further shows that high-\\(g\\) edges carry disproportionate task contribution beyond uniform sparsification."""

    new_topk = """Under strict hard Top-\\(K\\) deployment, AC-DSGF achieves **comparable** performance with heuristic strategies (Distance slightly higher; both above Random), demonstrating that the learned topology does **not** rely on excessive communication redundancy. The primary advantage of AC-DSGF is not “best \\(K\\) neighbors under a hard mask,” but learning **adaptive soft sparsity patterns** under soft budgets (Table~1 / Fig.~13), where SCAM is orders of magnitude lower than dense baselines while coordination remains competitive. Fig.~12 further shows task-aligned edge importance beyond uniform sparsification."""

    if old_topk in text:
        text = text.replace(old_topk, new_topk)
    else:
        print("WARN: topk narrative not found")

    # Fig.13 section strengthen as budget-performance curve
    text = text.replace(
        "**Fig. 13** Adaptive topology selection under different communication budgets (operating points on a frozen checkpoint). We do not claim Success leadership; rather:",
        "**Fig. 13** Budget–performance operating curve (frozen policy): SCAM vs Success under soft / budget / Top-\\(K\\) interventions. This visualizes the soft-budget trade-off surface induced by AC-DSGF without claiming a multi-\\(\\lambda_c\\) retrain sweep. We do not claim Success leadership; rather:",
    )

    # Conclusion tweak
    text = text.replace(
        "We present **AC-DSGF**, a constraint-aware adaptive communication topology optimization framework for multi-UAV swarms.",
        "We present **AC-DSGF**, a framework for learning adaptive communication sparsity patterns under soft budgets for multi-UAV swarms.",
    )

    # Keywords
    text = text.replace(
        "**Keywords:** constraint-aware communication topology; multi-UAV swarm; budgeted graph optimization; multi-agent RL",
        "**Keywords:** adaptive communication topology; soft budget constraints; UAV swarm coordination; communication activation; multi-agent RL",
    )
    return text


def make_pareto_fig():
    rows = list(csv.DictReader(TAB.open(encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {
        "AC-soft": "#0b6e4f",
        "AC-budget50%": "#2a9d8f",
        "AC-budget10%": "#40916c",
        "AC-Top1": "#52b788",
        "AC-Top2": "#74c69d",
        "AC-Top3": "#95d5b2",
        "AC-open(g=A)": "#9b2226",
        "AC-random": "#e09f3e",
        "DSGF(ref)": "#b35c00",
        "GAT(ref)": "#6c757d",
    }
    for r in rows:
        name = r["point"]
        x = float(r["soft_mass"])
        y = 100 * float(r["success"])
        ax.scatter(
            x,
            y,
            s=90 if name.startswith("AC-") else 70,
            c=colors.get(name, "#333"),
            edgecolors="#1a1a1a",
            zorder=3,
            label=name,
        )
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("SCAM $C_s$ (log)")
    ax.set_ylabel("Success (%)")
    ax.set_title(
        "Budget–performance operating curve (frozen AC policy)",
        loc="left",
        fontweight="bold",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="lower left", ncol=2, framealpha=0.92)
    fig.tight_layout()
    for dest in (FIG / "Fig13_comm_pareto.png", FIG_CN / "Fig13_comm_pareto.png"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Fig13")


def main():
    text = EN.read_text(encoding="utf-8")
    text2 = patch_en(text)
    EN.write_text(text2, encoding="utf-8")
    print("EN patched", len(text2))
    make_pareto_fig()


if __name__ == "__main__":
    main()
