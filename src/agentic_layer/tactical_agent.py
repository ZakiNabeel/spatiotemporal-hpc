from typing import TypedDict
import os
import pandas as pd

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

class TacticalState(TypedDict):
    raw_centroids: dict
    tactical_interpretation: str
    final_report: str

def interpreter(state: TacticalState):
    # Initialize the updated Groq model to llama-3.1-8b-instant
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
    # System prompt updated for 11-player formations
    prompt = ChatPromptTemplate.from_template(
        "You are an expert Premier League tactical analyst.\n"
        "You are analyzing 11-player relative coordinate sets. Recognize these geometric patterns "
        "as standard football formations (e.g., 4-3-3, 3-5-2) and describe their tactical intent.\n"
        "The coordinate center (0,0) is the team centroid.\n\n"
        "Given the following 22-dimensional raw centroid data (X, Y for 11 players) from our GPU clusters:\n"
        "{raw_centroids}\n\n"
        "Translate these coordinates into a recognized formation and describe the likely tactical style "
        "(e.g., attacking width, compact defensive block, double pivot)."
    )
    chain = prompt | llm
    response = chain.invoke({"raw_centroids": state["raw_centroids"]})
    
    return {"tactical_interpretation": response.content}

def scout(state: TacticalState):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
    prompt = ChatPromptTemplate.from_template(
        "You are the Head Scout for a Premier League team, reporting directly to the manager.\n"
        "Based on the following tactical formation interpretation derived from our tracking data:\n"
        "{tactical_interpretation}\n\n"
        "Format this interpretation into a professional Markdown scouting report. "
        "Make it actionable, clear, and cleanly structured."
    )
    chain = prompt | llm
    response = chain.invoke({"tactical_interpretation": state["tactical_interpretation"]})
    
    return {"final_report": response.content}

def build_tactical_agent():
    workflow = StateGraph(TacticalState)
    
    workflow.add_node("interpreter", interpreter)
    workflow.add_node("scout", scout)
    
    workflow.set_entry_point("interpreter")
    workflow.add_edge("interpreter", "scout")
    workflow.add_edge("scout", END)
    
    return workflow.compile()

def load_centroids_from_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    raw_centroids = {}
    for i, row in df.iterrows():
        coords = [round(val, 2) for val in row.values]
        raw_centroids[f"Cluster {i}"] = f"[{', '.join(map(str, coords))}]"
    return raw_centroids

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not found in environment. Please set it before running.")
        
    app = build_tactical_agent()
    
    csv_path = "data/processed/tactical_centroids.csv"
    if os.path.exists(csv_path):
        print(f"Loading real cluster data from {csv_path}...")
        gpu_output = load_centroids_from_csv(csv_path)
    else:
        print(f"Error: {csv_path} not found.")
        print("Please run the C++/CUDA engine or download the file from Colab and place it in data/processed/.")
        exit(1)
    
    initial_state = {
        "raw_centroids": gpu_output,
        "tactical_interpretation": "",
        "final_report": ""
    }
    
    try:
        print("Running Groq Tactical Agent Workflow (Llama 3.1)...")
        result = app.invoke(initial_state)
        print("\n--- FINAL SCOUTING BRIEF ---\n")
        print(result["final_report"])
        
        # Export the result to a text file
        output_file = "data/processed/tactical_report.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["final_report"])
        print(f"\nReport successfully exported to {output_file}")
        
    except Exception as e:
        print(f"Error executing agent: {e}")
