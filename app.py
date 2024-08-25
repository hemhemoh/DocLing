import pandas as pd
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

# API_KEY => Ensure to store the user pasted API_KEY into the environment

os.environ["COHERE_API_KEY"] = "hEfcl4uOqse4sCyqEnmQ1Q5OnI9UzdO8e9rSHYL3"
data = pd.read_csv("data-consolidated.csv")
data.rename(columns={"anwer-command-r": "answer"})


documents = []
for index, doc in data.iterrows():
    if isinstance(doc["answer"], str) and len(doc["answer"]):
        ques = doc["question"].encode().decode()
        ans = doc["answer"].encode().decode()

        documents.append(
            Document(
                content="Question: " + ques + "\nAnswer: " + ans
            )
        )


document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
document_embedder = CohereDocumentEmbedder(model="embed-multilingual-v2.0")
documents_with_embeddings = document_embedder.run(documents)["documents"]
document_store.write_documents(documents_with_embeddings, policy=DuplicatePolicy.SKIP)


template = """
            Given the following information, answer the question.

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


# DocumentStore update script (To be called after the a document is being uploaded in the frontend)
def store_update_script(text):
    doc = Document(content=text, meta="custom-data")
    document_with_embedding = document_embedder.run(doc)["documents"]
    document_store.write_documents(documents=document_with_embedding, policy=DuplicatePolicy.SKIP)


# Pipeline definition script (To be called to get the answer to the questions asked by the user)
def query_pipeline_script(query):
    result = query_pipeline.run({"text_embedder":{"text": query}})
    return result["llm"]["replies"][0]


