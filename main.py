from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- Config ----------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSV_PATH = "./walmart_products.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

app = FastAPI()

# ---------------------- Data & Model ----------------------
class SimpleVectorStore:
    def __init__(self):
        self.documents, self.embeddings, self.metadatas, self.ids = [], [], [], []

    def add(self, documents, embeddings, metadatas, ids):
        self.documents += documents
        self.embeddings += embeddings
        self.metadatas += metadatas
        self.ids += ids

    def query(self, query_embedding, n_results=5, where=None):
        filtered_indices = [
            i for i, metadata in enumerate(self.metadatas)
            if all(metadata.get(k) == v for k, v in (where or {}).items())
        ]
        if not filtered_indices:
            return {"documents": [[]], "distances": [[]]}
        filtered_embeddings = [self.embeddings[i] for i in filtered_indices]
        similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:n_results]
        return {
            "documents": [[self.documents[filtered_indices[i]] for i in top_indices]],
            "distances": [[1 - similarities[i] for i in top_indices]]
        }

def load_data():
    df = pd.read_csv(CSV_PATH)
    docs = [
        f"Product: {row.name}, Brand: {row.brand}, Category: {row.category}, "
        f"Price: ₹{row.price}, Discount: {row.discount}%, Description: {row.description}, "
        f"Stock: {row.stock_quantity}, Store: {row.store_location}"
        for _, row in df.iterrows()
    ]
    metas = [
        {"brand": row.brand, "category": row.category, "store": row.store_location}
        for _, row in df.iterrows()
    ]
    return df, docs, metas

model = SentenceTransformer(EMBEDDING_MODEL)
df, documents, metadatas = load_data()
vector_store = SimpleVectorStore()
vector_store.add(documents, model.encode(documents).tolist(), metadatas, [f"prod_{i}" for i in range(len(documents))])

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant for Walmart store data.
Use the following context to answer the user's question.
Answer in 1-2 lines by greeting the customer with meow.

Context:
{context}

Question:
{question}

Answer:
"""
)

rag_chain = prompt_template | llm

# ---------------------- API Schema ----------------------
class QueryRequest(BaseModel):
    question: str
    store: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    matched_products: List[str]

# ---------------------- API Endpoint ----------------------
@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    try:
        # Encode user query
        query_emb = model.encode([req.question])[0]
        where_filter = {"store": req.store} if req.store else None
        results = vector_store.query(query_embedding=query_emb, where=where_filter)
        context_chunks = results["documents"][0]
        if not context_chunks:
            return QueryResponse(answer="Meow! Sorry, no matching products found.", matched_products=[])

        context = "\n".join(context_chunks)
        response = rag_chain.invoke({
            "context": context,
            "question": req.question
        })
        answer = response.content if hasattr(response, 'content') else str(response)

        return QueryResponse(answer=answer, matched_products=context_chunks)
    except Exception as e:
        return QueryResponse(answer=f"Error: {str(e)}", matched_products=[])

@app.get("/")
def root():
    return {"status": "Walmart Product API is live."}
