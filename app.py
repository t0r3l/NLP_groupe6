"""
Streamlit interface for RAG Historian - African Civilizations
"""
import streamlit as st
from pathlib import Path
import re

from src_rag import models
from src_rag.models import calc_semantic_similarity

# Configuration
ROOT = Path(__file__).resolve().parent
WIKI_FOLDER = ROOT / "data" / "raw" / "wikipedia_pages"


@st.cache_resource
def load_rag():
    """Load and cache the RAG model with best configuration."""
    wiki_files = list(WIKI_FOLDER.glob("*.txt"))
    rag = models.RAG(
        chunk_size=256,
        small2big=True,
        add_metadata=True,
        embedding="miniLM"
    )
    rag.load_wikipedia_files(wiki_files)
    return rag


def calc_reply_accuracy(answer: str, context_chunks: list[str]) -> float:
    """Calculate similarity between answer and context (reply accuracy)."""
    if not context_chunks:
        return 0.0
    # Combine context chunks
    context = " ".join([clean_chunk_content(c) for c in context_chunks[:3]])
    # Use shared similarity function from evaluate.py
    return calc_semantic_similarity(answer, context)


def extract_metadata(chunk: str) -> dict:
    """Extract metadata from chunk prefix [Entity | Region | Period]."""
    match = re.match(r'^\[([^\]]+)\]', chunk)
    if match:
        parts = [p.strip() for p in match.group(1).split('|')]
        return {
            "entity": parts[0] if len(parts) > 0 else "",
            "region": parts[1] if len(parts) > 1 else "",
            "period": parts[2] if len(parts) > 2 else "",
        }
    return {"entity": "", "region": "", "period": ""}


def clean_chunk_content(chunk: str) -> str:
    """Remove metadata prefix from chunk."""
    return re.sub(r'^\[[^\]]+\]\n?', '', chunk)


# Page config
st.set_page_config(
    page_title="RAG Historian - Civilisations Africaines",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #D4A574;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #8B7355;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metadata-card {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #D4A574;
    }
    .chunk-content {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.9rem;
        margin-top: 10px;
    }
    .answer-box {
        background: linear-gradient(135deg, #2D2D2D 0%, #1E1E1E 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #D4A574;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌍 RAG Historian</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explorez les civilisations africaines précoloniales</p>', unsafe_allow_html=True)

# Load model
with st.spinner("Chargement du modèle RAG..."):
    rag = load_rag()

# Sidebar with config info
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("""
    **Modèle actuel:**
    - Embedding: `miniLM`
    - Chunk size: `256`
    - Small2Big: `✓`
    - Métadonnées: `✓`
    """)
    
    st.divider()
    
    st.header("📊 Corpus")
    st.markdown(f"""
    - **131** pages Wikipedia
    - **~37,000** tokens
    - Civilisations africaines précoloniales
    """)
    
    st.divider()
    
    st.header("💡 Exemples de questions")
    example_questions = [
        "Qui a fondé l'Empire du Mali ?",
        "Quelle était la capitale de l'Empire du Ghana ?",
        "Comment s'appelaient les guerrières du Dahomey ?",
        "Quel roi a adopté le christianisme à Aksoum ?",
    ]
    for q in example_questions:
        if st.button(q, key=q):
            st.session_state.question = q

# Main input
question = st.text_input(
    "🔍 Posez votre question sur les civilisations africaines:",
    value=st.session_state.get("question", ""),
    placeholder="Ex: Qui a fondé l'Empire du Mali ?"
)

if st.button("Rechercher", type="primary") or question:
    if question:
        with st.spinner("Recherche en cours..."):
            # Get context chunks
            context_chunks = rag._get_context(question)
            
            # Get answer
            answer = rag.reply(question)
            
            # Calculate reply accuracy
            reply_accuracy = calc_reply_accuracy(answer, context_chunks)
        
        # Display answer with accuracy
        st.markdown("### 📜 Réponse")
        
        col_answer, col_accuracy = st.columns([4, 1])
        
        with col_answer:
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        
        with col_accuracy:
            # Color based on accuracy
            if reply_accuracy >= 0.7:
                color = "#4CAF50"  # Green
            elif reply_accuracy >= 0.5:
                color = "#FFC107"  # Yellow
            else:
                color = "#F44336"  # Red
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 0.9rem; color: #888;">Reply Accuracy</div>
                <div style="font-size: 2rem; font-weight: bold; color: {color};">{reply_accuracy:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display sources
        st.markdown("### 📚 Sources utilisées")
        
        for i, chunk in enumerate(context_chunks[:5]):
            metadata = extract_metadata(chunk)
            content = clean_chunk_content(chunk)
            
            with st.expander(f"**Source {i+1}**: {metadata['entity']}" if metadata['entity'] else f"Source {i+1}", expanded=(i==0)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if metadata['entity']:
                        st.metric("🏛️ Entité", metadata['entity'])
                
                with col2:
                    if metadata['region']:
                        st.metric("🗺️ Région", metadata['region'])
                
                with col3:
                    if metadata['period']:
                        st.metric("📅 Période", metadata['period'])
                
                st.markdown("**Extrait:**")
                st.markdown(f'<div class="chunk-content">{content[:500]}{"..." if len(content) > 500 else ""}</div>', unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    RAG Historian - Projet NLP | Configuration optimale: chunk_size=256, small2big=True, add_metadata=True
</div>
""", unsafe_allow_html=True)

