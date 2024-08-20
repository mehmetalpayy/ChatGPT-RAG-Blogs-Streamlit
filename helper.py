import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain import hub


def fetch_medium_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find('article').get_text()
    return content


def fetch_and_prepare_documents(urls):
    docs = [Document(page_content=fetch_medium_article(url), metadata={"source": url}) for url in urls]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    splits = text_splitter.split_documents(docs)

    if not splits:
        raise ValueError("No document splits found. Please check your document loader and splitter.")

    ids = [str(i) for i in range(len(splits))]
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, ids=ids)
    return vectorstore, embedding


def create_rag_chain(vectorstore, llm):
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(vectorstore.as_retriever(), llm, prompt=prompt_template)
    prompt = hub.pull("rlm/rag-prompt")
    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain


def format_docs(data):
    return "\n\n".join(doc.page_content for doc in data)
