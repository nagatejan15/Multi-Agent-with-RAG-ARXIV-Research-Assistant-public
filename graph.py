import os
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated
from operator import add
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from tools.arxiv_tool import arxiv_research_tool
from tools.llm_tools import (
    internet_search_tool,
    math_tool,
    code_assistant_tool,
    get_current_date_tool,
    get_weather_tool,
)

load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

#  Add all tools to the list 
tools = [
    arxiv_research_tool,
    internet_search_tool,
    math_tool,
    code_assistant_tool,
    get_current_date_tool,
    get_weather_tool,
]

# Model 
model = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-1.5-flash", temperature=0.1)
model_with_tools = model.bind_tools(tools)

# AgentState 
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]

# Node
def call_model(state: AgentState) -> AgentState:
    """Invokes the model with the current conversation state."""
    print(" CALLING MODEL ")
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# Conditional Edge
def should_continue(state: AgentState):
    """Determines the next step after the model is called."""
    print(" CHECKING FOR TOOL CALLS ")
    last_message = state['messages'][-1]
    if getattr(last_message, 'tool_calls', None):
        print(" ROUTING TO TOOL ")
        return "continue"
    print(" ROUTING TO END ")
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")

# Conditional Edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    print("Multi-tool Agent is ready. Type 'exit' or 'stop' to end the conversation.")
    
    config = {"configurable": {"thread_id": "user-1"}}
    is_first_message = True

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "stop"]:
            print("\n Conversation Ended ")
            break
        
        inputs = {"messages": [HumanMessage(content=user_query)]}
        
        if is_first_message:
            inputs["messages"].insert(0, HumanMessage(content="""
You are an expert assistant with multiple tools. Your goal is to provide accurate, well-sourced answers.

Here are your tools and when to use them:

1.  **`get_current_date_tool`**: Use this to get the current date.
2.  **`get_weather_tool`**: Use this to get the real-time weather for a location.
3.  **`arxiv_research_tool`**: Use this FIRST for research questions (e.g., science, technology, academic topics).
4.  **`internet_search_tool`**: Use this for general knowledge questions or if the arXiv tool finds nothing.
5.  **`math_tool`**: Use this for mathematical calculations.
6.  **`code_assistant_tool`**: Use this for coding questions.

**Workflow and Sourcing:**
- Always use a tool to answer a question. Do not answer from your own knowledge.
- For research questions, try `arxiv_research_tool` first. If it fails, then use `internet_search_tool`.
- When you provide an answer, you MUST state where the information came from (e.g., "Source: arXiv", "Source: Internet", "Source: Model"). For weather or date, just provide the information from the tool.

Let's begin.
"""))
            is_first_message = False
        
        final_state = app.invoke(inputs, config=config)
        
        final_response = final_state['messages'][-1]

        if final_response and final_response.content:
            print(f"\n Agent Response \n{final_response.content}\n\n")
        else:
            # If the tool output is the final answer, which we will print from the state, rather than model's output.
            # last tool output in the messages.
            tool_output = None
            for msg in reversed(final_state['messages']):
                if msg.type == 'tool':
                    tool_output = msg.content
                    break
            if tool_output:
                 print(f"\n Agent Response \n{tool_output}\n\n")
            else:
                print("\n Agent did not produce a textual response. ")