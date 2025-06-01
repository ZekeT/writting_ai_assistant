from langgraph.graph import Graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. Define your few-shot examples
examples = [
    {"input": "Article 1 about X...", "output": "Processed version of X..."},
    # ... add all 10 examples
]

# 2. Create a prompt template with instructions + examples
prompt = ChatPromptTemplate.from_messages([
    ("system", """You transform news articles based on these patterns:
        {examples}
        Follow the same approach for new articles."""),
    ("user", "Transform these: {input_articles}")
])

# 3. LangGraph nodes


def extract_patterns(state):
    # Optional: Use an LLM to summarize patterns from examples
    llm = ChatOpenAI()
    response = llm.invoke(
        "Summarize key transformation patterns from these examples: {examples}")
    return {"patterns": response}


def transform_articles(state):
    llm = ChatOpenAI()
    chain = prompt | llm
    return {"output": chain.invoke({"input_articles": state["articles"], "examples": examples})}


# 4. Build the graph
workflow = Graph()
workflow.add_node("extract_patterns", extract_patterns)
workflow.add_node("transform", transform_articles)
workflow.add_edge("extract_patterns", "transform")
workflow.set_entry_point("extract_patterns")
