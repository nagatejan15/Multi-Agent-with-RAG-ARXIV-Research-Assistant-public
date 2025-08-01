import os
from dotenv import load_dotenv
import arxiv
from llama_index.core import Document
from langchain.tools import tool
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.query_engine import CitationQueryEngine

load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Model and Embeddings Setup
Settings.llm = Gemini(api_key=os.getenv("GOOGLE_API_KEY"),model_name="models/gemini-2.5-flash")
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")

def fetch_arxiv_papers(query: str, max_results: int = 5) -> list:
    """
    Fetches papers from arXiv based on a query and returns them as LlamaIndex Document objects.
    """
    print(f"Searching arXiv for query: '{query}'")
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = list(search.results())
    if not results:
        print("No papers found on arXiv for the given query.")
        return []

    print(f"Found {len(results)} papers.")

    paper_documents = []
    for result in results:
        doc_text = f"Title: {result.title}\n\nAbstract: {result.summary}"
        document = Document(
            text=doc_text,
            metadata={
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "pdf_url": result.pdf_url,
                "published_date": str(result.published.date())
            }
        )
        paper_documents.append(document)

    return paper_documents

@tool
def arxiv_research_tool(research_query: str) -> str:
    """
    Performs a full RAG pipeline on arXiv data for a given research query.
    Use this tool to answer questions about cutting-edge research, technology,
    scientific concepts, and academic topics. The input should be a clear,
    specific research question or topic.
    """
    paper_documents = fetch_arxiv_papers(query=research_query)

    if not paper_documents:
        return "I could not find any relevant papers on arXiv for your query. Please try a different topic."

    # In-memory index
    print("Creating in-memory RAG index from fetched papers...")
    temp_index = VectorStoreIndex.from_documents(paper_documents)

    # Citation query engine to include sources
    print("Creating citation query engine...")
    query_engine = CitationQueryEngine.from_args(
        temp_index,
        similarity_top_k=3,
        citation_chunk_size=512,
    )

    print("Querying the RAG pipeline...")
    response = query_engine.query(research_query)

    answer = str(response)
    sources = response.source_nodes

    if sources:
        answer += "\n\nSOURCES:"
        for i, source in enumerate(sources):
            # Access the metadata we stored in the Document object
            metadata = source.node.metadata
            answer += f"\n\n[{i+1}] Title: {metadata.get('title', 'N/A')}"
            
            # Format authors into a readable string
            authors = ", ".join(metadata.get('authors', []))
            if authors:
                answer += f"\n   - Authors: {authors}"
            
            answer += f"\n   - Source URL: {metadata.get('pdf_url', 'N/A')}"

    return answer