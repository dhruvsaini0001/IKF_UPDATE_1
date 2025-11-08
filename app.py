"""Streamlit UI for LinkedIn RAG Agent"""

import streamlit as st
import os
import sys
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
import json

load_dotenv()

# Import functions from simple_rag.py
sys.path.append(str(Path(__file__).parent))
from simple_rag import (
    VECTOR_STORE_DIR, 
    OUTPUT_DIR, 
    client,
    get_embeddings,
    mmr_selection,
    check_plagiarism,
    load_memory,
    save_memory
)

# Get LLM model from environment
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Page config
st.set_page_config(
    page_title="LinkedIn Post Generator - IKF",
    page_icon="✍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .snippet-box {
        background-color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .generated-post {
        background-color: black;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">✍️ LinkedIn Post Generator</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Index selection and info
with st.sidebar:
    st.header("⚙️ Settings")
    
    # List available indexes
    available_indexes = []
    if VECTOR_STORE_DIR.exists():
        available_indexes = [d.name for d in VECTOR_STORE_DIR.iterdir() if d.is_dir() and (d / "index.faiss").exists()]
    
    if not available_indexes:
        st.error("❌ No indexes found. Please build an index first using CLI.")
        st.code("python simple_rag.py build data/posts/ikf.txt ikf")
        st.stop()
    
    index_name = st.selectbox("📂 Select Index", available_indexes)
    
    st.markdown("---")
    
    # Load and display index info
    index_dir = VECTOR_STORE_DIR / index_name
    index = faiss.read_index(str(index_dir / "index.faiss"))
    
    st.metric("📊 Total Vectors", index.ntotal)
    st.metric("📏 Dimension", index.d)
    
    st.markdown("---")
    
    # Memory info
    if st.button("🧠 View Memory"):
        memory = load_memory(index_name)
        with st.expander("Memory Details", expanded=True):
            st.json(memory)

# Main content - Three inputs
col1, col2, col3 = st.columns(3)

with col1:
    person_name = st.text_input(
        "👤 Person/Brand Name",
        value="IKF Team",
        help="Name of the person or brand posting"
    )

with col2:
    company = st.text_input(
        "🏢 Company/Context",
        value="IKF Creative Agency",
        help="Company name or additional context"
    )

with col3:
    topic = st.text_input(
        "💡 Topic",
        value="Content marketing trends 2025",
        help="Main topic for the LinkedIn post"
    )

# Optional bullets
st.markdown("### 📝 Key Points (Optional)")
bullet1 = st.text_input("Bullet 1", placeholder="e.g., Focus on authentic storytelling")
bullet2 = st.text_input("Bullet 2", placeholder="e.g., Leverage video content")
bullet3 = st.text_input("Bullet 3", placeholder="e.g., Data-driven strategy")

# Combine inputs
bullets = [b for b in [bullet1, bullet2, bullet3] if b.strip()]
full_topic = topic
if bullets:
    full_topic += "\n\nKey points:\n" + "\n".join(f"- {b}" for b in bullets)

st.markdown("---")

# Generate button
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    generate_button = st.button("🚀 Generate Post", type="primary", use_container_width=True)

# Generation logic
if generate_button:
    if not person_name or not topic:
        st.error("⚠️ Please fill in Person Name and Topic")
    else:
        with st.spinner("🔍 Retrieving relevant examples..."):
            # Load index and chunks
            with open(index_dir / "texts.txt", 'r', encoding='utf-8') as f:
                chunks = f.read().split('\n---\n')
            
            # Load memory
            memory = load_memory(index_name)
            
            # Search with MMR
            query_emb = get_embeddings([full_topic], input_type="query")
            k_candidates = 15
            distances, indices = index.search(query_emb, min(k_candidates, index.ntotal))
            
            # Get candidates
            candidate_chunks = [chunks[idx] for idx in indices[0]]
            candidate_embs = get_embeddings(candidate_chunks, input_type="passage")
            
            # Apply MMR (select 5 diverse examples)
            selected_indices = mmr_selection(query_emb[0], candidate_embs, indices[0], num_to_select=5, relevance_weight=0.7)
            context = [chunks[idx] for idx in selected_indices]
        
        # Show retrieved snippets
        st.markdown("### 📚 Retrieved Snippets (Used for Style Matching)")
        for i, snippet in enumerate(context, 1):
            with st.expander(f"Snippet {i}", expanded=(i == 1)):
                st.markdown(f'<div class="snippet-box">{snippet}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Generate post
        with st.spinner("✍️ Generating LinkedIn post..."):
            context_text = "\n\n".join(f"Example {i+1}: {c}" for i, c in enumerate(context))
            
            # Build memory context
            memory_context = ""
            if memory.get("preferred_hashtags"):
                memory_context += f"\nPreferred hashtags: {', '.join(memory['preferred_hashtags'])}"
            if memory.get("recurring_themes"):
                memory_context += f"\nRecurring themes: {', '.join(memory['recurring_themes'])}"
            if memory.get("banned_phrases"):
                memory_context += f"\nAvoid these phrases: {', '.join(memory['banned_phrases'])}"
            
            system_prompt = f"""You are a LinkedIn post writer for {person_name}. Write authentic, professional posts.

STRICT FORMAT REQUIREMENTS:
1. Write exactly 120-220 words
2. Structure: Hook → Perspective → Concrete illustration → Reflection
3. NO emojis allowed
4. End with keywords in square brackets: [keyword1, keyword2, keyword3]
5. Add maximum 4 hashtags on a new line starting with hashtag#
6. Optional: Add signature line if appropriate{memory_context}

Example format:
[Hook that grabs attention...]

[Perspective or insight...]

[Concrete example or illustration...]

[Reflection or call to action...]

[keyword1, keyword2, keyword3, keyword4, keyword5]

hashtag#Hashtag1 hashtag#Hashtag2 hashtag#Hashtag3 hashtag#Hashtag4"""
            
            llm_model = LLM_MODEL
            temperature = float(os.getenv("TEMPERATURE", "0.7"))
            
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Write a LinkedIn post for {person_name} about: {full_topic}\n\nStyle examples from past posts:\n{context_text}\n\nWrite the post now (120-220 words):"}
                ],
                temperature=temperature
            )
            
            post = response.choices[0].message.content.strip()
            
            # Anti-plagiarism check (pass context chunks directly)
            is_plagiarized, matched_snippet = check_plagiarism(post, context)
            
            # Word count
            word_count = len(post.split('[')[0].split())
        
        # Display generated post
        st.markdown("### ✅ Generated Post")
        st.markdown(f'<div class="generated-post">{post}</div>', unsafe_allow_html=True)
        
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("📊 Word Count", f"{word_count}")
        with col_m2:
            st.metric("🎯 Target", "120-220")
        with col_m3:
            status = "✅ In Range" if 120 <= word_count <= 220 else "⚠️ Out of Range"
            st.metric("📏 Status", status)
        with col_m4:
            st.metric("🔍 Snippets", len(context))
        
        # Plagiarism check result
        if is_plagiarized:
            matched_text = matched_snippet[:100] if matched_snippet else 'Unknown'
            st.markdown(f'<div class="warning-box">⚠️ <strong>WARNING:</strong> Potential plagiarism detected!<br>Matched snippet: {matched_text}...</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ <strong>Anti-plagiarism check:</strong> PASSED</div>', unsafe_allow_html=True)
        
        # Note: prior_outputs removed - all generation history stored in evaluation_{index_name}.json only
        
        # Save to evaluation JSON
        evaluation_file = OUTPUT_DIR / f"evaluation_{index_name}.json"
        
        # Load existing evaluations
        existing_evaluations = []
        if evaluation_file.exists():
            try:
                with open(evaluation_file, 'r', encoding='utf-8') as f:
                    existing_evaluations = json.load(f)
            except:
                existing_evaluations = []
        
        # Add new streamlit generation
        new_entry = {
            "source": "streamlit",
            "topic": topic,
            "person": person_name,
            "company": company,
            "bullets": bullets if bullets else None,
            "generated_post": post,
            "word_count": word_count,
            "plagiarism_check": "PASSED" if not is_plagiarized else "FAILED",
            "retrieved_snippets": len(context),
            "timestamp": str(Path(__file__).stat().st_mtime)
        }
        
        existing_evaluations.append(new_entry)
        
        # Save updated evaluations
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(existing_evaluations, f, indent=2, ensure_ascii=False)
        
        # Save to text file
        output_file = OUTPUT_DIR / f"{index_name}_post_streamlit.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(post)
            f.write(f"\n\n--- Metadata ---\n")
            f.write(f"Person: {person_name}\n")
            f.write(f"Company: {company}\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Word count: {word_count}\n")
            f.write(f"Retrieved snippets: {len(context)} (with MMR diversity)\n")
            f.write(f"Plagiarism check: {'FAILED' if is_plagiarized else 'PASSED'}\n")
        
        st.success(f"💾 Post saved to: {output_file.name} & evaluation_{index_name}.json")
        
        # Download button
        st.download_button(
            label="📥 Download Post",
            data=post,
            file_name=f"{index_name}_linkedin_post.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>LinkedIn RAG Agent | Powered by NVIDIA NIM & FAISS | Built with ❤️ by IKF</small>
</div>
""", unsafe_allow_html=True)
