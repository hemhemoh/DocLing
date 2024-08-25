import streamlit as st
import pandas as pd
import pdfplumber
from haystack import Document
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.cohere.text_embedder import CohereTextEmbedder
from haystack_integrations.components.embedders.cohere.document_embedder import CohereDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.cohere import CohereGenerator
import os
from io import StringIO

# Initialize the Cohere API key
os.environ["COHERE_API_KEY"] = "hEfcl4uOqse4sCyqEnmQ1Q5OnI9UzdO8e9rSHYL3"

st.title("Document-based Question Answering")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

if uploaded_file:
    documents = []

    # Process CSV file
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
        data.rename(columns={"anwer-command-r": "answer"}, inplace=True)

        for index, doc in data.iterrows():
            if isinstance(doc["answer"], str) and len(doc["answer"]):
                ques = doc["question"].encode().decode()
                ans = doc["answer"].encode().decode()

                documents.append(
                    Document(
                        content="Question: " + ques + "\nAnswer: " + ans
                    )
                )

    # Process PDF file
    elif uploaded_file.name.endswith('.pdf'):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                # Simple heuristic to split questions and answers
                lines = text.split("\n")
                for i in range(0, len(lines), 2):
                    if i + 1 < len(lines):
                        ques = lines[i]
                        ans = lines[i + 1]
                        documents.append(
                            Document(
                                content="Question: " + ques + "\nAnswer: " + ans
                            )
                        )

    # Initialize the document store
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    # Embed the documents
    document_embedder = CohereDocumentEmbedder(model="embed-multilingual-v2.0")
    documents_with_embeddings = document_embedder.run(documents)["documents"]
    document_store.write_documents(documents_with_embeddings, policy=DuplicatePolicy.SKIP)

    # Define the pipeline components
    template = """
        Given the following information, answer the question with MAXIMUM ACCURACY.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{ query }}?
         
        """

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", CohereTextEmbedder(model="embed-multilingual-v2.0"))
    query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    query_pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    query_pipeline.add_component("llm", CohereGenerator())
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder", "llm")

    # Allow the user to ask questions
    query = st.text_input("Ask a question based on the uploaded document:")
    if query:
        result = query_pipeline.run({"text_embedder": {"text": query}})
        st.write("Answer:", result["llm"]["replies"][0])

else:
    st.info("Please upload a PDF or CSV file to proceed.")
