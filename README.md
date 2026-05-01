# ⚽ Tactical Phase Clustering & Agentic Scouting Engine

> **A Hybrid HPC-AI Pipeline for Automated Football Formation Detection and Strategic Analysis.**

## 📖 Project Overview
This project transforms raw, unstructured spatiotemporal tracking data into high-level tactical intelligence. It processes millions of data points to identify recurring team formations and generates professional scouting briefs using a multi-agent orchestration layer.

The system analyzes **145,007 frames** of tracking data (sampled at 25fps) by extracting **22-dimensional feature vectors**—representing the normalized X and Y coordinates for all 11 players simultaneously.

---

## 🛠️ Technical Architecture

### 1. The HPC Layer (CUDA/C++)
Heavy geometric computations are offloaded to an **NVIDIA Tesla T4 GPU** to handle the computational load of high-frequency sports data.
*   **Custom K-Means Kernel:** Parallelized Euclidean distance calculations across 22 dimensions.
*   **Performance:** Thousands of parallel threads calculate tactical centroids near-instantaneously, optimizing the workflow for real-time or batch-match analysis.
*   **Output:** Generates `tactical_centroids.csv`, representing the mathematical "DNA" of each identified tactical phase.

### 2. High-Resolution Data Engineering
The engine utilizes **Formation Normalization** to ensure tactical patterns are recognized regardless of pitch location.
*   **Geometric Invariance:** By subtracting the team centroid from each player's coordinate, the system recognizes a "4-3-3" shape whether the team is in a high press or a low block.
*   **Feature Set:** Comprehensive 11-player relative coordinates ($Player1\_X, Player1\_Y \dots Player11\_Y$).

### 3. Agentic Orchestration (LangGraph + Llama 3.1)
The "Voice" of the system is built using **LangGraph** and **Groq**, running the **Llama-3.1-8b-instant** model.
*   **The Interpreter:** Translates raw GPU centroids into recognized football formations and geometric patterns.
*   **The Scout:** Formats the interpretation into an actionable, professional Markdown brief for coaching and technical staff.

---

## 📊 Real-World GPU Output: Tactical Scouting Brief

Based on the **22-dimensional cluster centroids** calculated by the CUDA engine, the Agentic Layer identified the following strategic phases:

### **Phase 1: Transitional 4-3-3 / 4-4-2**
*   **Tactical Style:** Quick transitions utilizing pacey wingers to attack flanks.
*   **Key Detail:** The engine identified a stable four-man backline with a double-pivot midfield structure.

### **Phase 2: Defensive Counter-Attack**
*   **Tactical Style:** Deep defensive block with a focus on catching opponents off-guard during transition.
*   **Key Detail:** Formation density increases centrally to deny space between the lines.

### **Phase 3: Strategic Tactical Shift (3-5-2 / 3-4-3)**
*   **Tactical Style:** A major tactical pivot to a "Target Man" setup.
*   **Key Detail:** The GPU identified a shift to three central defenders and wide wing-backs, intended for greater midfield control and direct play.

---

## 📂 Project Structure
```text
tactical-clustering-engine/
├── data/
│   ├── raw/                  # Metrica tracking data (Sample Game 1)
│   └── processed/            # 22-column formation matrices & GPU Centroids
├── src/
│   ├── data_pipeline/        # Coordinate normalization & Metrica parsing
│   ├── hpc_core/             # C++/CUDA Parallel Clustering Engine
│   └── agentic_layer/        # LangGraph & Llama 3.1 Scouting Agent
└── README.md
