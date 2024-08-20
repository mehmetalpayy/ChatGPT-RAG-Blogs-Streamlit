from langchain_openai import ChatOpenAI
from helper import fetch_and_prepare_documents, create_rag_chain, fetch_medium_article
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# URLs for Medium articles
urls = [
    "https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c",
    "https://medium.com/@myscale/how-does-a-retrieval-augmented-generation-system-work-9e5d12b4a80f",
    "https://medium.com/@nielspace/enhancing-language-models-through-retrieval-augmented-generation-8fccc0cdac07",
    "https://medium.com/@ranadevrat/understanding-langchain-agents-a-beginners-guide-8a87708dc48e",
    "https://saurabhharak.medium.com/understanding-langchain-agents-with-example-5a92503d0cc2",
    "https://vijaykumarkartha.medium.com/beginners-guide-to-creating-ai-agents-with-langchain-eaa5c10973e6",
    "https://medium.com/@siladityaghosh/intro-to-llm-agents-with-langchain-beyond-simple-prompts-4ee1edd00225"
]

# Fetch and prepare documents
docs = [Document(page_content=fetch_medium_article(url), metadata={"source": url}) for url in urls]

vectorstore, embedding = fetch_and_prepare_documents(urls)

# Define the language model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define the RAG chain
chain = create_rag_chain(vectorstore, llm)
