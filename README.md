# Scalable Multi-Cellular Networks (SMCN) — Reproduction & Research Extensions

This repository contains my **reproduction, analysis, and extensions** of the paper:

**Topological Blindspots: Understanding and Extending Topological Deep Learning Through the Lens of Expressivity (ICLR 2025 Oral)**

The original work introduces **Scalable Multi-Cellular Networks (SMCN)** for learning over combinatorial complexes.  The original paper can be found here : https://arxiv.org/abs/2408.05486
This repository goes beyond reproduction and focuses on **understanding, benchmarking, and extending SMCN from a research perspective**.

---

## Overview

Topological Deep Learning (TDL) extends graph learning to **higher-order structures** such as cells, cochains, and incidence relations.

SMCN improves upon prior methods (e.g., HOMP) by:
- enabling **multi-rank message passing**
- capturing **higher-order interactions**
- improving **expressivity over combinatorial complexes**

This repository aims to:
- reproduce the original results
- analyze **expressivity limits**
- study **oversquashing and long-range dependencies**
- design **new benchmarks and experiments**

---

## Research Objectives

### 1. Reproducibility
- Reproduce synthetic benchmarks (e.g., torus datasets)
- Validate performance on real-world datasets (e.g., ZINC)

### 2. Expressivity Analysis
- Study topological invariants:
  - Betti numbers
  - Diameter / cross-diameter
  - Euler characteristic
- Compare SMCN with:
  - HOMP
  - Standard GNNs

### 3. Oversquashing & Long-Range Dependencies
- Convert combinatorial complexes → relational graphs
- Evaluate on:
  - NeighborMatch
  - RingTransfer
- Analyze information flow across distant cells

### 4. New Experiments
- Euler characteristic prediction
- Robustness to lifting strategies
- Ablations on multi-rank message passing

### 5. Scalability
- Study computational cost of higher-order interactions
- Explore efficient approximations of SMCN layers

---

## Installation

```bash
pip install -r requirements.txt