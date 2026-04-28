import streamlit as st
import os
import tempfile

from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from PyPDF2 import PdfReader

# ------------------- API KEY -------------------
os.environ["GROQ_API_KEY"] = "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"

# ------------------- TOOLS -------------------

wikipedia = WikipediaAPIWrapper()
arxiv = ArxivAPIWrapper()
search = DuckDuckGoSearchRun()

@tool
def web_loader(url: str) -> str:
    """Load content from a given URL."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    return f"Tool Used: Web Loader\n{docs[0].page_content[:1000]}"

@tool
def pdf_reader(file_path: str) -> str:
    """Read a PDF file and return extracted text."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return f"Tool Used: PDF Reader\n{text[:1000]}"

@tool
def wikipedia_search(query: str) -> str:
    """Search information from Wikipedia."""
    return f"Tool Used: Wikipedia\n{wikipedia.run(query)}"

@tool
def arxiv_search(query: str) -> str:
    """Search research papers from Arxiv."""
    return f"Tool Used: Arxiv\n{arxiv.run(query)}"

@tool
def open_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    return f"Tool Used: DuckDuckGo\n{search.run(query)}"

# ------------------- LLM -------------------

llm = ChatGroq(model="llama-3.1-8b-instant")

# ------------------- AGENT -------------------

agent = initialize_agent(
    tools=[web_loader, pdf_reader, wikipedia_search, arxiv_search, open_search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ------------------- UI -------------------

st.set_page_config(page_title="AI Smart Search Assistant", layout="centered")

st.title("🔍 AI Smart Search Assistant")

# -------- PDF Upload --------
st.subheader("📂 Upload PDF (optional)")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

pdf_path = None
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
    st.success("PDF uploaded successfully!")

# -------- Query Input --------
st.subheader("💬 Ask your question")
user_input = st.text_input("Enter your query:")

if st.button("Search"):

    if user_input:
        with st.spinner("Thinking..."):

            try:
                # If PDF uploaded → force agent to use it
                if pdf_path:
                    user_input = f"Use the PDF at {pdf_path} and answer: {user_input}"

                response = agent.run(user_input)

                # -------- Extract Tool Used --------
                tool_used = "Not detected"
                if "Tool Used:" in response:
                    tool_used = response.split("Tool Used:")[1].split("\n")[0]

                clean_output = response.replace("**", "")

                # -------- Display --------
                st.subheader("📌 Answer:")
                st.write(clean_output)

                st.subheader("🛠 Tool Used:")
                st.info(tool_used)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a query.")
