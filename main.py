import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ------------------- CONFIG -------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CSV_PATH = "./walmart_products.csv"

# Get API key with better error handling
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except:
        pass  # Will handle error in main app

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
        # Filter by metadata if specified
        if where:
            filtered_indices = []
            for i, metadata in enumerate(self.metadatas):
                match = True
                for key, value in where.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_indices.append(i)
        else:
            filtered_indices = list(range(len(self.documents)))
        
        if not filtered_indices:
            return {"documents": [[]], "distances": [[]]}
        
        # Calculate similarities
        filtered_embeddings = [self.embeddings[i] for i in filtered_indices]
        similarities = cosine_similarity([query_embedding], filtered_embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = {
            "documents": [[self.documents[filtered_indices[i]] for i in top_indices]],
            "distances": [[1 - similarities[i] for i in top_indices]]
        }
        
        return results

# ------------------- CACHE FUNCTIONS -------------------
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_data
def load_data():
    try:
        if not os.path.exists(CSV_PATH):
            return None, None, None
            
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
        st.error(f"Failed to load data: {e}")
        return None, None, None

@st.cache_resource
def setup_vector_store():
    try:
        data_result = load_data()
        if data_result[0] is None:
            return None
            
        df, documents, metadatas = data_result
        model = load_model()
        if model is None:
            return None
        
        embeddings = model.encode(documents).tolist()
        
        vector_store = SimpleVectorStore()
        vector_store.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"prod_{i}" for i in range(len(documents))]
        )
        
        return vector_store
    except Exception as e:
        st.error(f"Failed to setup vector store: {e}")
        return None

@st.cache_resource
def setup_llm():
    try:
        if not GOOGLE_API_KEY:
            st.error("❌ Google API Key not found!")
            return None
            
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
    except Exception as e:
        st.error(f"Failed to setup LLM: {e}")
        return None

@st.cache_resource
def setup_rag_chain():
    try:
        llm = setup_llm()
        if llm is None:
            return None
            
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
        # Use the new RunnableSequence instead of deprecated LLMChain
        return prompt | llm
    except Exception as e:
        st.error(f"Failed to setup RAG chain: {e}")
        return None

# ------------------- MAIN APP -------------------
def main():
    st.title("🛒 Walmart Product Assistant")
    st.markdown("Ask me anything about Walmart products!")
    
    # Add debug info
    with st.expander("🔧 Debug Information"):
        st.write(f"Google API Key configured: {'✅' if GOOGLE_API_KEY else '❌'}")
        st.write(f"CSV file exists: {'✅' if os.path.exists(CSV_PATH) else '❌'}")
    
    # Check prerequisites
    if not GOOGLE_API_KEY:
        st.error("❌ Google API Key not found! Please set it in Streamlit secrets or environment variables.")
        st.stop()
    
    if not os.path.exists(CSV_PATH):
        st.error("❌ CSV file 'walmart_products.csv' not found!")
        st.stop()
    
    # Initialize components with progress
    try:
        with st.spinner("🔄 Loading AI models... (This may take 1-2 minutes on first run)"):
            model = load_model()
            if model is None:
                st.stop()
            st.success("✅ Sentence transformer loaded")
            
        with st.spinner("📊 Loading product data..."):
            data_result = load_data()
            if data_result[0] is None:
                st.stop()
            df, _, _ = data_result
            st.success(f"✅ Loaded {len(df)} products")
            
        with st.spinner("🤖 Setting up AI assistant..."):
            vector_store = setup_vector_store()
            if vector_store is None:
                st.stop()
            rag_chain = setup_rag_chain()
            if rag_chain is None:
                st.stop()
            st.success("✅ AI assistant ready")
        
        # Get unique store locations
        stores = df['store_location'].unique().tolist()
        
        # Sidebar for store selection
        with st.sidebar:
            st.header("🏪 Store Selection")
            selected_store = st.selectbox("Choose a store:", stores, index=0)
            
            st.header("📊 Store Stats")
            store_df = df[df['store_location'] == selected_store]
            st.metric("Total Products", len(store_df))
            st.metric("Categories", store_df['category'].nunique())
            st.metric("Brands", store_df['brand'].nunique())
        
        # Main chat interface
        st.header(f"💬 Chat with {selected_store} Store Assistant")
        
        # Sample questions
        st.markdown("**Try asking:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📱 Samsung phones under ₹25000"):
                st.session_state.sample_question = "What Samsung smartphones do you have under ₹25000?"
        with col2:
            if st.button("💻 Electronics on sale"):
                st.session_state.sample_question = "What electronics do you have with good discounts?"
        
        # Chat input
        default_question = st.session_state.get('sample_question', '')
        user_question = st.text_input("Ask your question:", value=default_question, placeholder="What products do you have in electronics?")
        
        if st.button("Ask", type="primary") and user_question:
            with st.spinner("🔍 Searching for products..."):
                try:
                    # Get query embedding
                    query_embedding = model.encode([user_question])[0]
                    
                    # Search in vector store
                    results = vector_store.query(
                        query_embedding=query_embedding,
                        n_results=5,
                        where={"store": selected_store}
                    )
                    
                    context_chunks = results["documents"][0]
                    
                    if not context_chunks:
                        st.warning(f"No products found in {selected_store} store for your query. Try a different store or question.")
                        return
                    
                    context = "\n".join(context_chunks)
                    
                    # Generate answer using the new invoke method
                    with st.spinner("🤖 Generating response..."):
                        response = rag_chain.invoke({
                            "context": context,
                            "question": user_question
                        })
                        # Extract content from the response
                        answer = response.content if hasattr(response, 'content') else str(response)
                    
                    # Display results
                    st.success("Found matching products!")
                    st.write("**Answer:**")
                    st.write(answer)
                    
                    # Show context in expander
                    with st.expander("📋 Product Details Found"):
                        for i, chunk in enumerate(context_chunks, 1):
                            st.write(f"**Product {i}:**")
                            st.write(chunk)
                            st.divider()
                    
                    st.info(f"Found {len(context_chunks)} matching products")
                    
                    # Clear the sample question
                    if 'sample_question' in st.session_state:
                        del st.session_state.sample_question
                    
                except Exception as e:
                    st.error(f"❌ Error processing your question: {str(e)}")
                    st.write("**Debug info:**")
                    st.code(str(e))
        
        # Display sample products
        st.header(f"🛍️ Sample Products from {selected_store}")
        store_df = df[df['store_location'] == selected_store].head(5)
        
        if len(store_df) == 0:
            st.warning(f"No products found for {selected_store} store")
        else:
            for _, product in store_df.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.subheader(product['name'])
                        st.write(f"**Brand:** {product['brand']}")
                        st.write(f"**Category:** {product['category']}")
                        st.write(f"**Description:** {product['description'][:100]}...")
                    
                    with col2:
                        st.metric("Price", f"₹{product['price']}")
                        st.write(f"**Discount:** {product['discount']}%")
                    
                    with col3:
                        st.metric("Stock", f"{product['stock_quantity']} units")
                    
                    st.divider()
        
    except Exception as e:
        st.error(f"❌ Error initializing the app: {str(e)}")
        st.write("**Full error details:**")
        st.code(str(e))
        
        # Debug information
        st.write("**Debug checklist:**")
        st.write(f"- Google API Key: {'✅' if GOOGLE_API_KEY else '❌ Missing'}")
        st.write(f"- CSV file: {'✅' if os.path.exists(CSV_PATH) else '❌ Missing'}")
        st.write("**Possible solutions:**")
        st.write("1. Add GOOGLE_API_KEY to Streamlit secrets")
        st.write("2. Ensure walmart_products.csv is in the repository")
        st.write("3. Check internet connection for model downloads")

if __name__ == "__main__":
    main()