import streamlit as st
from agent_core import create_agent_executor
from langchain_core.globals import set_debug
import os

# Set global debug flag for LangChain logs in the terminal
set_debug(False)

# --- Streamlit Application ---
st.set_page_config(page_title="Gen AI Study Guide Agent")
st.title("🤖 Automated Study Guide Generator")
st.caption("Powered by Phi-3 Mini, LangChain, and ChromaDB")

# Initialize agent_executor once using Streamlit's cache
@st.cache_resource
def get_agent_executor():
    """Initializes and caches the agent executor."""
    try:
        return create_agent_executor()
    except Exception as e:
        st.error(f"Error initializing AI agent. Ensure Ollama is running (phi3:mini and nomic-embed-text) and the knowledge base has been created.")
        st.error(f"Error details: {e}")
        return None

agent_executor = get_agent_executor()

if agent_executor:
    # --- User Input Form ---
    with st.form("study_guide_form"):
        topic_query = st.text_input(
            "Enter the Topic to Generate a Guide For:",
            value="Deep Learning"
        )
        output_type = st.selectbox(
            "Select Output Type:",
            ("Summary", "Q&A", "Flashcards")
        )
        
        submitted = st.form_submit_button("Generate Study Guide")

    # --- Agent Execution ---
    if submitted and topic_query:
        # Determine which tool the agent should prioritize based on the dropdown
        if output_type == "Summary":
            prompt = f"Generate a study summary of 3 lines for the topic: {topic_query}"
        elif output_type == "Q&A":
            prompt = f"Generate 2 Q&A pairs for the topic: {topic_query}"
        elif output_type == "Flashcards":
            prompt = f"Generate 2 flashcards (FRONT: BACK:) for the topic: {topic_query}"
        
        st.subheader(f"✅ Generating {output_type} for '{topic_query}'...")
        
        with st.spinner("Agent is retrieving context and generating output..."):
            try:
                # Invoke the agent
                result = agent_executor.invoke({"input": prompt})
                
                # Display the result
                st.success("Generation Complete!")
                st.markdown("---")
                st.markdown(result['output'])
                
            except Exception as e:
                st.error("An error occurred during agent execution. Check the terminal for LangChain logs if 'set_debug(True)' is enabled.")
                st.error(f"Execution Error: {e}")