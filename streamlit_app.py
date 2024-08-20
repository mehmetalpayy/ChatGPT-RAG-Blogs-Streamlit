import streamlit as st
from helper import fetch_and_prepare_documents, fetch_medium_article, create_rag_chain
from rag import chain as llama_chain, urls as default_urls
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def main():
    st.title("Retrieval-Augmented Generation (RAG) System")

    # Sidebar for example URL input
    st.sidebar.header("Example URL")
    if st.sidebar.button("Load Example Documents"):
        try:
            # Fetch and prepare documents using default URLs
            docs = [Document(page_content=fetch_medium_article(url), metadata={"source": url}) for url in default_urls]
            vectorstore, embedding = fetch_and_prepare_documents(default_urls)

            # Define the model
            llm = ChatOpenAI(model="gpt-3.5-turbo")

            # Initialize the RAG chain
            chain_instance = create_rag_chain(vectorstore, llm)
            st.session_state["vector_db"] = vectorstore
            st.session_state["chain_instance"] = chain_instance

            st.success("Default documents loaded and RAG chain created successfully!")
        except ValueError as e:
            st.error(f"Error loading default documents: {str(e)}")

    # Sidebar for additional URL input
    st.sidebar.header("Optional URL")
    url_input = st.sidebar.text_area(
        "Enter additional Medium Article URLs (one per line)",
        value=""  # empty for user input
    )

    urls = [url.strip() for url in url_input.split('\n') if url.strip()]

    if st.sidebar.button("Load Optional Documents"):
        if urls:
            try:
                # Fetch and prepare documents using additional URLs
                docs = [Document(page_content=fetch_medium_article(url), metadata={"source": url}) for url in urls]
                vectorstore, embedding = fetch_and_prepare_documents(urls)

                # Define the language model
                llm = ChatOpenAI(model="gpt-3.5-turbo")

                # Update existing vector database if it exists
                if "vector_db" in st.session_state:
                    st.session_state["vector_db"].update(vectorstore)
                else:
                    st.session_state["vector_db"] = vectorstore

                # Update chain with the new vectorstore and llm
                st.session_state["chain_instance"] = create_rag_chain(st.session_state["vector_db"], llm)

                st.success("Additional documents loaded and RAG chain updated successfully!")
            except ValueError as e:
                st.error(f"Error loading additional documents: {str(e)}")
        else:
            st.error("Please enter URLs to load additional documents.")

    st.header("Ask a Question")

    question = st.text_input("Enter your question:")
    if st.button("Submit"):
        if "chain_instance" not in st.session_state:
            st.error("Please load documents first.")
        else:
            # Retrieve the chain instance from session state
            chain_instance = st.session_state["chain_instance"]
            # Use the chain's `invoke` method for processing
            try:
                response = chain_instance.invoke({"question": question})
                st.write("Generated Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error during query processing: {str(e)}")


if __name__ == "__main__":
    main()
