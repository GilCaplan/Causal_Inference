"""Render the preliminary causal DAG for the military-service / income study."""
import graphviz

g = graphviz.Digraph("military_income_dag", format="png")
g.attr(rankdir="LR", splines="spline", nodesep="0.4", ranksep="0.6")
g.attr("node", shape="ellipse", style="filled", fontname="Helvetica",
       fillcolor="#eaeaea")

g.node("MIL", "Military Service\n(MIL) — treatment",
       fillcolor="#cde4ff", shape="box")
g.node("PINCP", "Personal Income > $50k\n(PINCP) — outcome",
       fillcolor="#cfeccf", shape="box")

# Pre-treatment background confounders (in ACS)
for n, lbl in [
    ("AGEP", "Age (AGEP)"),
    ("SEX", "Sex (SEX)"),
    ("RAC1P", "Race (RAC1P)"),
    ("HISP", "Hispanic origin (HISP)"),
    ("POBP", "Place of birth (POBP)"),
    ("NATIVITY", "Nativity / Citizenship\n(NATIVITY, CIT)"),
]:
    g.node(n, lbl)

# Unobserved / partly-observed background confounders
for n, lbl in [
    ("FAM_SES", "Family SES /\nrural background (unobs.)"),
    ("PRE_EDU", "Pre-service education\n& ability (unobs.)"),
    ("HEALTH0", "Pre-service health\n(unobs.)"),
]:
    g.node(n, lbl, style="dashed,filled", fillcolor="#fff4c2")

# Post-treatment mediators (caused by MIL, then cause PINCP)
for n, lbl in [
    ("SCHL", "Education attained\n(SCHL) — GI Bill mediator"),
    ("COW", "Class of worker\n(COW) — federal pref."),
    ("OCCP", "Occupation (OCCP)"),
    ("WKHP", "Hours worked (WKHP)"),
    ("DIS", "Disability /\nVA rating (DIS, DRAT)"),
    ("MAR", "Marital status (MAR)"),
]:
    g.node(n, lbl, fillcolor="#f3d9ff")

# Background -> Treatment (selection into service)
for src in ["AGEP", "SEX", "RAC1P", "HISP", "POBP", "NATIVITY",
            "FAM_SES", "PRE_EDU", "HEALTH0"]:
    g.edge(src, "MIL")

# Background -> Outcome (confounding paths)
for src in ["AGEP", "SEX", "RAC1P", "HISP", "NATIVITY",
            "FAM_SES", "PRE_EDU", "HEALTH0"]:
    g.edge(src, "PINCP")

# Treatment -> Mediators -> Outcome
for med in ["SCHL", "COW", "OCCP", "WKHP", "DIS", "MAR"]:
    g.edge("MIL", med)
    g.edge(med, "PINCP")

# Direct treatment effect (the causal estimand of interest)
g.edge("MIL", "PINCP", color="#1a6dff", penwidth="2.0")

out = g.render("causal_graph", cleanup=True)
print(f"Wrote {out}")
