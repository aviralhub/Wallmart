import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ------------------- CONFIG -------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CSV_PATH = "./walmart_products.csv"

# ------------------- API KEY -------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

# ------------------- STREAMLIT CONFIG -------------------
st.set_page_config(
    page_title="Walmart Product Assistant",
    page_icon="🛒",
    layout="wide"
)

# ------------------- VECTOR STORE CLASS -------------------
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []

    def add(self, documents, embeddings, metadatas, ids):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def query(self, query_embedding, n_results=5, where=None):
        filtered_indices = []
        for i, metadata in enumerate(self.metadatas):
            if not where or all(metadata.get(k) == v for k, v in where.items()):
                filtered_indices.append(i)

        if not filtered_indices:
            return {"documents": [[]], "distances": [[]]}

        filtered_embeddings = [self.embeddings[i] for i in filtered_indices]
        similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:n_results]

        return {
            "documents": [[self.documents[filtered_indices[i]] for i in top_indices]],
            "distances": [[1 - similarities[i] for i in top_indices]]
        }

# ------------------- CACHE FUNCTIONS -------------------
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"❌ Failed to load embedding model:\n{e}")
        raise e

@st.cache_data
def load_data():
    try:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"{CSV_PATH} not found")
        df = pd.read_csv(CSV_PATH)
        documents = [
            f"Product: {row.name}, Brand: {row.brand}, Category: {row.category}, "
            f"Price: ₹{row.price}, Discount: {row.discount}%, Description: {row.description}, "
            f"Stock: {row.stock_quantity} units, Store: {row.store_location}"
            for _, row in df.iterrows()
        ]
        metadatas = [
            {"brand": str(row.brand), "category": str(row.category), "store": str(row.store_location)}
            for _, row in df.iterrows()
        ]
        return df, documents, metadatas
    except Exception as e:
        st.error(f"❌ Failed to load data:\n{e}")
        raise e

@st.cache_resource
def setup_vector_store():
    df, documents, metadatas = load_data()
    model = load_model()
    embeddings = model.encode(documents).tolist()
    vector_store = SimpleVectorStore()
    vector_store.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"prod_{i}" for i in range(len(documents))]
    )
    return vector_store, df, model

@st.cache_resource
def setup_llm():
    if not GOOGLE_API_KEY:
        st.error("❌ GOOGLE_API_KEY is missing")
        raise ValueError("GOOGLE_API_KEY not found")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

@st.cache_resource
def setup_rag_chain(llm):
    template = """
    You are a helpful assistant for Walmart store data.
    Use the following context to answer the user's question.
    Answer in 1-2 lines by greeting the customer with meow.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    return prompt | llm

# ------------------- MAIN APP -------------------
def main():
    st.title("🛒 Walmart Product Assistant")
    st.markdown("Ask me anything about Walmart products!")

    with st.expander("🔧 Debug Information"):
        st.write(f"GOOGLE_API_KEY: {'✅' if GOOGLE_API_KEY else '❌'}")
        st.write(f"CSV exists: {'✅' if os.path.exists(CSV_PATH) else '❌'}")

    try:
        with st.spinner("🔄 Loading components..."):
            vector_store, df, model = setup_vector_store()
            llm = setup_llm()
            rag_chain = setup_rag_chain(llm)

        stores = df['store_location'].unique().tolist()

        with st.sidebar:
            st.header("🏪 Store Selection")
            selected_store = st.selectbox("Choose a store:", stores)

            st.header("📊 Store Stats")
            store_df = df[df['store_location'] == selected_store]
            st.metric("Total Products", len(store_df))
            st.metric("Categories", store_df['category'].nunique())
            st.metric("Brands", store_df['brand'].nunique())

        st.header(f"💬 Chat with {selected_store} Store Assistant")
        col1, col2 = st.columns(2)
        if col1.button("📱 Samsung phones under ₹25000"):
            st.session_state.sample_question = "What Samsung smartphones do you have under ₹25000?"
        if col2.button("💻 Electronics on sale"):
            st.session_state.sample_question = "What electronics do you have with good discounts?"

        user_question = st.text_input("Ask your question:", value=st.session_state.get("sample_question", ""))

        if st.button("Ask", type="primary") and user_question:
            with st.spinner("🔍 Fetching products..."):
                query_embedding = model.encode([user_question])[0]
                results = vector_store.query(query_embedding, n_results=5, where={"store": selected_store})
                context_chunks = results["documents"][0]

                if not context_chunks:
                    st.warning("❌ No matching products found.")
                else:
                    context = "\n".join(context_chunks)
                    response = rag_chain.invoke({
                        "context": context,
                        "question": user_question
                    })
                    answer = getattr(response, 'content', str(response))

                    st.success("✅ Found matching products!")
                    st.write(answer)

                    with st.expander("📋 Product Details"):
                        for i, chunk in enumerate(context_chunks, 1):
                            st.markdown(f"**Product {i}:** {chunk}")
                            st.divider()

        st.header(f"🛍️ Sample Products from {selected_store}")
        for _, product in df[df['store_location'] == selected_store].head(5).iterrows():
            col1, col2, col3 = st.columns([2, 1, 1])
            col1.subheader(product['name'])
            col1.write(f"**Brand:** {product['brand']}")
            col1.write(f"**Category:** {product['category']}")
            col1.write(f"**Description:** {product['description'][:100]}...")
            col2.metric("Price", f"₹{product['price']}")
            col2.write(f"**Discount:** {product['discount']}%")
            col3.metric("Stock", f"{product['stock_quantity']} units")
            st.divider()

    except Exception as e:
        st.error("❌ The app crashed:")
        st.code(str(e))

if __name__ == "__main__":
    main()
