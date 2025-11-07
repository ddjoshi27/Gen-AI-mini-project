from langchain.agents.react.agent import create_react_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain.agents import initialize_agent
from langchain_core.tools import Tool
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.agents.agent_types import AgentType

CHROMA_PERSIST_DIR = "ChromaDB Store"
OLLAMA_LLM_MODEL = "phi3:mini"  # Use the model you pulled for generation
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
llm = OllamaFunctions(model=OLLAMA_LLM_MODEL)

def initialize_llm():
    """Initializes the Ollama LLM for reasoning and generation."""
    system_prompt = (
        "You are a meticulous AI agent. "
        "You MUST strictly adhere to the ReAct format: Thought, then Action, then Action Input. "
        "DO NOT include any extra text outside of these specific labels. "
        "The Action Input MUST be present after the Action."
    )
    # --- PREREQUISITE: Run 'ollama run phi3:mini' in a separate terminal ---
    return Ollama(model=OLLAMA_LLM_MODEL, temperature=0.1, system=system_prompt)

def create_retriever_tool(vectorstore):
    """Creates a LangChain Tool from the ChromaDB Retriever."""
    retriever = vectorstore.as_retriever(k=2) # Retrieve top 4 most relevant chunks
    
    # Define the tool function that the agent can call
    def retrieve_context(query: str) -> str:
        """Use this tool to search the educational content database for relevant text chunks."""
        docs = retriever.invoke(query)
        # Format the retrieved documents into a single string context
        context = "\n\n".join([f"--- SOURCE: {doc.metadata.get('source', 'N/A')} ---\n{doc.page_content}" for doc in docs])
        return context

    return Tool(
        name="Educational_Content_Retriever",
        func=retrieve_context,
        description="ALWAYS use this tool first to retrieve course content before summarizing or generating Q&A. Input should be the user's topic query."
    )

def create_generation_tools(llm):
    """
    Creates the three specialized generation tools using the LLM and specific prompts.
    The agent will pass the retrieved context to these tools.
    """
    # 1. Summarizer Tool
    summary_prompt = ChatPromptTemplate.from_template(
        "You are an expert tutor. Create a concise, bulleted summary for a student based on this retrieved context: {context}"
    )
    summarizer_chain = summary_prompt | llm
    summarizer_tool = Tool(
        name="Study_Guide_Summarizer",
        func=lambda context: summarizer_chain.invoke({"context": context}),
        description="Use ONLY this tool to generate a comprehensive study summary from retrieved context."
    )

    # 2. Q&A Tool
    qa_prompt = ChatPromptTemplate.from_template(
        "You are a test designer. Based ONLY on the following context, create 5 unique question-answer pairs (format as Q: ... A: ...): {context}"
    )
    qa_chain = qa_prompt | llm
    qa_tool = Tool(
        name="QA_Generator",
        func=lambda context: qa_chain.invoke({"context": context}),
        description="Use ONLY this tool to generate structured question-answer pairs for self-testing from retrieved context."
    )

    # 3. Flashcard Tool
    flashcard_prompt = ChatPromptTemplate.from_template(
        "You are a flashcard creator. Based ONLY on the following context, create 5 simple flashcards (format as FRONT: ... BACK: ...): {context}"
    )
    flashcard_chain = flashcard_prompt | llm
    flashcard_tool = Tool(
        name="Flashcard_Generator",
        func=lambda context: flashcard_chain.invoke({"context": context}),
        description="Use ONLY this tool to generate simple, front/back flashcards from retrieved context."
    )

    return [summarizer_tool, qa_tool, flashcard_tool]

def create_agent_executor():
    """Builds the final LangChain Agent Executor."""
    # 1. Load the ChromaDB
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    
    # 2. Initialize LLM
    llm = initialize_llm()

    # 3. Define Tools
    retriever_tool = create_retriever_tool(vectorstore)
    generation_tools = create_generation_tools(llm)
    all_tools = [retriever_tool] + generation_tools

    # 4. Create Agent
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    
    # 5. Create Executor
    agent_executor = initialize_agent(
        tools=all_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, 
        handle_parsing_errors=True
    )
    return agent_executor