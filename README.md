# ⚽ Tactical Phase Clustering & Agentic Scouting Engine

> **GPU-accelerated spatiotemporal clustering and multi-agent tactical reporting for high-frequency football tracking data.**

## 📖 Overview
Modern elite football (e.g., the Premier League) generates millions of spatiotemporal data points per match via 25fps tracking cameras. Identifying distinct tactical phases (e.g., high-press, low-block transitions) from this massive, unstructured coordinate data is computationally expensive and difficult to interpret.

This project solves both bottlenecks through a **Hybrid Cloud-HPC Pipeline**:
1. **The HPC Layer:** Bypasses standard Python libraries by utilizing custom **C++/CUDA kernels** to parallelize the mathematical distance calculations required for K-Means clustering across millions of tactical frames.
2. **The Agentic Layer:** Utilizes a **LangGraph** multi-agent orchestration workflow to analyze the resulting mathematical centroids and automatically generate natural-language tactical scouting briefs.

## 🛠️ Tech Stack
*   **Data Pipeline:** Python, Pandas, NumPy
*   **High-Performance Computing:** C++, CUDA, NVIDIA Tesla T4 (via Google Colab)
*   **AI Orchestration:** LangGraph, LangChain, OpenAI GPT-4o
*   **Data Source:** Metrica Sports Open Tracking Dataset (105x68m pitch, normalized coordinates)

## 📂 Repository Structure
```text
tactical-clustering-engine/
├── data/                     # (Ignored in Git)
│   ├── raw/                  # Metrica CSVs (Raw Tracking Data)
│   └── processed/            # Parsed feature matrices ready for GPU
├── src/
│   ├── data_pipeline/        # Feature Engineering (Centroid & Spread)
│   │   └── parse_metrica.py
│   ├── hpc_core/             # C++/CUDA Parallel Clustering
│   │   ├── kmeans_kernel.cu
│   │   └── cluster_runner.cpp
│   └── agentic_layer/        # LangGraph & LLM Orchestration
│       └── tactical_agent.py
├── notebooks/                # Quick EDA and visual sanity checks
├── README.md
└── requirements.txt