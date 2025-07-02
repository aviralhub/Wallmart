import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ------------------ Load env ------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ------------------ Constants ------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CSV_PATH = "./walmart_products.csv"

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Walmart Product Assistant", page_icon="🛒", layout="wide")

# ------------------ Simple Vector Store ------------------
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

# ------------------ Caching ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_data
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

@st.cache_resource
def setup_vector_store():
    df, docs, metas = load_data()
    model = load_model()
    embeddings = model.encode(docs).tolist()
    vs = SimpleVectorStore()
    vs.add(docs, embeddings, metas, [f"prod_{i}" for i in range(len(docs))])
    return vs

@st.cache_resource
def setup_llm():
    if not GOOGLE_API_KEY:
        st.error("Google API Key missing!")
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

@st.cache_resource
def setup_rag_chain():
    llm = setup_llm()
    prompt = PromptTemplate(
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
    return prompt | llm

# ------------------ Main App ------------------
def main():
    st.title("🛒 Walmart Product Assistant")
    st.markdown("Ask me anything about Walmart products!")

    if not os.path.exists(CSV_PATH):
        st.error("CSV file not found!")
        return
    if not GOOGLE_API_KEY:
        st.error("Google API Key not found in environment!")
        return

    df, _, _ = load_data()
    model = load_model()
    vs = setup_vector_store()
    rag = setup_rag_chain()

    stores = df['store_location'].unique().tolist()
    with st.sidebar:
        selected_store = st.selectbox("🏬 Select a store:", stores)
        st.metric("Total Products", len(df[df['store_location'] == selected_store]))

    st.header(f"💬 Ask about {selected_store} store")

    user_question = st.text_input("Your question:", "")
    if st.button("Ask") and user_question:
        with st.spinner("Searching..."):
            query_embedding = model.encode([user_question])[0]
            results = vs.query(query_embedding, where={"store": selected_store})
            chunks = results["documents"][0]
            if not chunks:
                st.warning("No results found.")
                return
            context = "\n".join(chunks)
            response = rag.invoke({"context": context, "question": user_question})
            answer = response.content if hasattr(response, 'content') else str(response)
            st.success("✅ Answer:")
            st.write(answer)

            with st.expander("📄 Product Matches"):
                for i, doc in enumerate(chunks, 1):
                    st.markdown(f"**Match {i}:** {doc}")
                    st.divider()

if __name__ == "__main__":
    main()
