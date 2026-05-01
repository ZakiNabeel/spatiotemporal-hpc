from typing import TypedDict
import os

# Note: In a production environment, you should handle imports robustly.
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class TacticalState(TypedDict):
    centroid_x: float
    spread: float
    tactical_description: str
    final_report: str

def interpretation_node(state: TacticalState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    prompt = ChatPromptTemplate.from_template(
        "You are an expert football tactical analyst.\n"
        "Given the team's average Centroid X: {centroid_x}m (where 0 is center circle, positive is opponent's half) "
        "and Team Spread: {spread}m,\n"
        "provide a short, concise tactical description (e.g., 'High Press, Compact Shape')."
    )
    chain = prompt | llm
    response = chain.invoke({
        "centroid_x": state["centroid_x"], 
        "spread": state["spread"]
    })
    
    return {"tactical_description": response.content}

def reporting_node(state: TacticalState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "You are a Head Scout for a Premier League team.\n"
        "Using the following tactical interpretation from our data model: '{tactical_description}',\n"
        "generate a brief Markdown scouting brief summarizing the team's shape and likely approach. "
        "Keep it professional and action-oriented."
    )
    chain = prompt | llm
    response = chain.invoke({
        "tactical_description": state["tactical_description"]
    })
    
    return {"final_report": response.content}

def build_tactical_agent():
    workflow = StateGraph(TacticalState)
    
    # Add nodes
    workflow.add_node("interpretation", interpretation_node)
    workflow.add_node("reporting", reporting_node)
    
    # Define edges
    workflow.set_entry_point("interpretation")
    workflow.add_edge("interpretation", "reporting")
    workflow.add_edge("reporting", END)
    
    return workflow.compile()

if __name__ == "__main__":
    # Ensure you have your API key set in your environment
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment.")
        
    app = build_tactical_agent()
    
    # Example state: highly advanced centroid (+15.2m into opponent half), large spread
    initial_state = {
        "centroid_x": 15.2,
        "spread": 30.5,
        "tactical_description": "",
        "final_report": ""
    }
    
    try:
        print("Running Tactical Agent Workflow...")
        result = app.invoke(initial_state)
        print("\n--- FINAL SCOUTING BRIEF ---\n")
        print(result["final_report"])
    except Exception as e:
        print(f"Error executing agent: {e}")
