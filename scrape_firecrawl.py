from langchain_core.documents import Document
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import START, StateGraph
from langchain import hub
from typing_extensions import List, TypedDict

from dotenv import load_dotenv
import os

load_dotenv()

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
POSTGRES_USERNAME = os.getenv("POSTGRES_USERNAME")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_CONNECTION_STRING = f"postgresql+psycopg2://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@localhost:5432/scraper"
COLLECTION_NAME = "website"


def main(args):
    COLLECTION_NAME = args.website.split(".")[1]
    print(COLLECTION_NAME)
    # Initialize Ollama embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    # Load existing vector store
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONNECTION_STRING,
        use_jsonb=True,
    )
    if not args.scrape:
        vector_store = PGVector.from_existing_index(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection=PG_CONNECTION_STRING,
            use_jsonb=True,
        )
    else:
        # Scrape website
        loader = FireCrawlLoader(
            api_key=FIRECRAWL_API_KEY,
            api_url="http://localhost:3002",
            url=args.website,
            mode="scrape",
        )

        pages = []
        for doc in loader.lazy_load():
            pages.append(doc)
            if len(pages) >= 10:
                break
                # do some paged operation, e.g.
                # index.upsert(page)
                pages = []

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        chunks = text_splitter.split_documents(pages)

        # Create PostgreSQL vector store
        vector_store = PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection=PG_CONNECTION_STRING,
            pre_delete_collection=True,  # Set to False if you want to append to existing collection
        )

    # Initialize Ollama LLM for RAG
    # llm = OllamaLLM(model="mistral:latest")
    llm = OllamaLLM(model="deepseek-r1:1.5b")

    # Prompts and graphs
    prompt = hub.pull("rlm/rag-prompt")

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response = llm.invoke(messages)
        return {"answer": response}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    result = graph.invoke({"question": args.query})
    print(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, default="https://www.scrapegraphai.com")
    parser.add_argument("--scrape", action="store_true")
    parser.add_argument(
        "--query",
        type=str,
        default="Describe the paid plans and designated clientele of the company.",
    )
    args = parser.parse_args()
    main(args)
