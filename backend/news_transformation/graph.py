from langgraph.graph import Graph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI

# 1. Create analysis node


def analyze_articles(state):
    # Extract patterns from input-output pairs
    pass

# 2. Create transformation node


def transform_articles(state):
    # Apply learned transformations to new articles
    llm = ChatOpenAI()
    messages = [
        SystemMessage(
            "You transform articles based on these patterns: [patterns]"),
        HumanMessage(f"Transform these articles: {state['input_articles']}")
    ]
    response = llm.invoke(messages)
    return {"output_articles": response.content}


# 3. Build the graph
workflow = Graph()
workflow.add_node("analyze", analyze_articles)
workflow.add_node("transform", transform_articles)
workflow.add_edge("analyze", "transform")
workflow.set_entry_point("analyze")
