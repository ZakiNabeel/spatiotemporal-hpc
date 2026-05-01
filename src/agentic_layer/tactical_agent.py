from typing import TypedDict
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

class TacticalState(TypedDict):
    raw_centroids: dict
    tactical_interpretation: str
    final_report: str

def interpreter(state: TacticalState):
    # Initialize the Groq model
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.2)
    
    # Emphasizing pitch dimensions and coordinate center
    prompt = ChatPromptTemplate.from_template(
        "You are an expert Premier League tactical analyst.\n"
        "We have clustered player tracking data on a football pitch with dimensions 105x68m.\n"
        "The center spot of the pitch is at coordinates (0,0).\n"
        "Positive X values indicate the opponent's half, while negative X values indicate the team's own half.\n\n"
        "Given the following raw centroid data (X, Y) and team spread from our GPU clusters:\n"
        "{raw_centroids}\n\n"
        "Translate these raw metrics into football tactical terminology (e.g., 'High Press', 'Mid Block', 'Low Block'). "
        "Provide a concise and accurate tactical interpretation."
    )
    chain = prompt | llm
    response = chain.invoke({"raw_centroids": state["raw_centroids"]})
    
    return {"tactical_interpretation": response.content}

def scout(state: TacticalState):
    # Initialize the Groq model
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.2)
    
    prompt = ChatPromptTemplate.from_template(
        "You are the Head Scout for a Premier League team, reporting directly to the manager.\n"
        "Based on the following tactical interpretation derived from our tracking data:\n"
        "{tactical_interpretation}\n\n"
        "Format this interpretation into a professional Markdown scouting report. "
        "Make it actionable, clear, and cleanly structured."
    )
    chain = prompt | llm
    response = chain.invoke({"tactical_interpretation": state["tactical_interpretation"]})
    
    return {"final_report": response.content}

def build_tactical_agent():
    workflow = StateGraph(TacticalState)
    
    # Add nodes
    workflow.add_node("interpreter", interpreter)
    workflow.add_node("scout", scout)
    
    # Define edges and entry/exit points
    workflow.set_entry_point("interpreter")
    workflow.add_edge("interpreter", "scout")
    workflow.add_edge("scout", END)
    
    return workflow.compile()

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not found in environment. Please set it before running.")
        
    app = build_tactical_agent()
    
    # Mock GPU output data to test logic immediately
    mock_gpu_output = {
        "Phase 1 Cluster": {"centroid_x": 18.5, "centroid_y": 2.1, "spread_x": 35.0, "spread_y": 40.0},
        "Phase 2 Cluster": {"centroid_x": -15.0, "centroid_y": -5.0, "spread_x": 25.0, "spread_y": 30.0}
    }
    
    initial_state = {
        "raw_centroids": mock_gpu_output,
        "tactical_interpretation": "",
        "final_report": ""
    }
    
    try:
        print("Running Groq Tactical Agent Workflow...")
        result = app.invoke(initial_state)
        print("\n--- FINAL SCOUTING BRIEF ---\n")
        print(result["final_report"])
    except Exception as e:
        print(f"Error executing agent: {e}")
