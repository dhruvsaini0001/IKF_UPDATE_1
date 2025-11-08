"""
LinkedIn RAG Agent - Generate personalized posts that match your writing style
"""

import os
import sys
import json
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === SETUP DIRECTORIES ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
POSTS_DIR = DATA_DIR / "posts"           # Where your training posts are stored
VECTOR_STORE_DIR = DATA_DIR / "vector_store"  # Where FAISS indexes are saved
MEMORY_DIR = DATA_DIR / "memory"         # Style preferences and constraints
OUTPUT_DIR = BASE_DIR / "outputs"        # Generated posts saved here

# Create directories if they don't exist
for directory in [POSTS_DIR, VECTOR_STORE_DIR, MEMORY_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# === OPENAI CLIENT ===
# Works with both OpenAI and NVIDIA NIM (via OPENAI_BASE_URL)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", None)
)

# ============================================================================
# STEP 1: LOAD & PREPARE DATA
# ============================================================================

def load_posts(file_path):
    """
    Load LinkedIn posts from a text file.
    
    Posts should be separated by '---' and can include optional metadata:
        date: YYYY-MM-DD
        link: https://...
        [ keywords, in, brackets ]
    
    Returns:
        posts: List of post text strings
        metadata: List of dictionaries with post_id, date, link, and keywords
    """
    import re
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    posts = []
    metadata = []
    post_id = 1
    
    # Split by --- delimiter
    for post_block in content.split('---'):
        if not post_block.strip():
            continue
        
        lines = post_block.strip().split('\n')
        post_info = {
            'post_id': post_id,
            'date': None,
            'link': None,
            'keywords': []
        }
        post_lines = []
        
        # Separate metadata from post text
        for line in lines:
            if line.startswith('date:'):
                post_info['date'] = line.split('date:', 1)[1].strip()
            elif line.startswith('link:'):
                post_info['link'] = line.split('link:', 1)[1].strip()
            elif line.strip().startswith('[') and line.strip().endswith(']'):
                # Extract keywords from brackets
                keywords_text = line.strip()[1:-1]  # Remove [ and ]
                # Split by comma and clean
                keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
                post_info['keywords'] = keywords
            else:
                post_lines.append(line)
        
        posts.append('\n'.join(post_lines).strip())
        metadata.append(post_info)
        post_id += 1
    
    return posts, metadata


def chunk_posts(posts):
    """
    Production-grade text chunking with token-aware sentence combining.
    
    Strategy:
    1. Split posts into paragraphs (preserve structure)
    2. Use NLTK for accurate sentence tokenization (handles Dr., Mr., etc.)
    3. Combine consecutive sentences until ~350 tokens (max 512)
    4. Filter out fragments <20 words
    5. Optimize for embedding model token limits
    
    This approach:
    - Preserves writing style and paragraph boundaries
    - Ensures chunks fit within embedding model limits (512 tokens)
    - Maximizes context per chunk (better retrieval quality)
    - Uses production-grade sentence splitting
    
    Args:
        posts: List of post text strings
    
    Returns:
        List of text chunks ready for embedding
    """
    # Ensure NLTK punkt tokenizer is available
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("📦 Downloading NLTK punkt tokenizer (one-time setup)...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
    except ImportError:
        print("⚠️  NLTK not installed. Falling back to simple sentence splitting.")
        print("   Install with: pip install nltk")
        nltk = None
    
    chunks = []
    
    # Token estimation: ~1 token per 4 characters (conservative estimate)
    # Target: ~350 tokens (1400 chars), Max: 512 tokens (2048 chars)
    TARGET_TOKENS = 350
    MAX_TOKENS = 512
    CHARS_PER_TOKEN = 4
    TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN  # 1400
    MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN        # 2048
    
    for post in posts:
        # STEP 1: Split by paragraphs (preserve natural structure)
        paragraphs = [p.strip() for p in post.split('\n\n') if p.strip()]
        
        # Fallback: if no double-newlines, split by single newlines
        if not paragraphs:
            paragraphs = [p.strip() for p in post.split('\n') if p.strip()]
        
        for paragraph in paragraphs:
            word_count = len(paragraph.split())
            char_count = len(paragraph)
            
            # CASE 1: Short paragraph - keep as-is
            # (Under target size, maintains style and context)
            if char_count <= TARGET_CHARS:
                if word_count >= 20:  # Filter fragments
                    chunks.append(paragraph)
                continue
            
            # CASE 2: Long paragraph - split into sentences and recombine
            # Use NLTK for accurate sentence boundary detection
            if nltk:
                try:
                    sentences = nltk.sent_tokenize(paragraph)
                except:
                    # Fallback if NLTK fails
                    sentences = paragraph.replace('!', '.').replace('?', '.').split('.')
                    sentences = [s.strip() for s in sentences if s.strip()]
            else:
                # Simple fallback without NLTK
                sentences = paragraph.replace('!', '.').replace('?', '.').split('.')
                sentences = [s.strip() for s in sentences if s.strip()]
            
            # STEP 2: Combine sentences into token-aware chunks
            current_chunk = []
            current_chars = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_chars = len(sentence)
                sentence_words = len(sentence.split())
                
                # Skip fragments (too short to be meaningful)
                if sentence_words < 20:
                    continue
                
                # Check if adding this sentence would exceed MAX_TOKENS
                if current_chars + sentence_chars + 1 > MAX_CHARS:
                    # Save current chunk if it has content
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text.split()) >= 20:
                            chunks.append(chunk_text)
                    
                    # Start new chunk with this sentence
                    current_chunk = [sentence]
                    current_chars = sentence_chars
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_chars += sentence_chars + 1  # +1 for space
            
            # Don't forget the last chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= 20:
                    chunks.append(chunk_text)
    
    return chunks


# ============================================================================
# STEP 2: EMBEDDINGS (Convert text to vectors)
# ============================================================================

def get_embeddings(texts, input_type="passage"):
    """
    Convert text into numerical vectors (embeddings) using OpenAI API.
    
    Args:
        texts: List of strings to embed
        input_type: "passage" for storing, "query" for searching
    
    Returns:
        NumPy array of shape (num_texts, embedding_dimension)
        Default dimension: 1536 for text-embedding-3-small
    """
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # NVIDIA models need extra parameters
    extra_params = {}
    if "nvidia" in model.lower():
        extra_params = {"input_type": input_type, "truncate": "END"}
    
    response = client.embeddings.create(
        input=texts,
        model=model,
        extra_body=extra_params
    )
    
    # Convert to NumPy array for efficient computation
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype='float32')


# ============================================================================
# STEP 3: MMR ALGORITHM (Select diverse examples)
# ============================================================================

def mmr_selection(query_embedding, candidate_embeddings, candidate_indices, 
                  num_to_select=5, relevance_weight=0.7):
    """
    Maximal Marginal Relevance: Select diverse yet relevant examples.
    
    The Problem:
        Simple top-k search often returns very similar results.
        Example: All 5 results might say "AI transforms marketing"
    
    The Solution:
        MMR balances relevance with diversity.
        Formula: MMR = λ × Relevance - (1-λ) × Max_Similarity
        
    Args:
        query_embedding: The search query vector
        candidate_embeddings: All potential result vectors
        candidate_indices: Original indices of candidates
        num_to_select: How many to select (default: 5)
        relevance_weight: λ parameter, 0.7 means 70% relevance, 30% diversity
    
    Returns:
        List of selected indices (diverse + relevant)
    """
    selected_indices = []
    selected_embeddings = []
    
    # Normalize all vectors for cosine similarity calculation
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    
    # Calculate how relevant each candidate is to the query
    relevance_scores = np.dot(candidate_norms, query_norm.T).flatten()
    
    # STEP 1: Pick the most relevant item first
    most_relevant_idx = np.argmax(relevance_scores)
    selected_indices.append(candidate_indices[most_relevant_idx])
    selected_embeddings.append(candidate_embeddings[most_relevant_idx])
    
    # STEP 2: Pick remaining items using MMR formula
    remaining_indices = list(range(len(candidate_indices)))
    remaining_indices.remove(most_relevant_idx)
    
    while len(selected_indices) < num_to_select and remaining_indices:
        mmr_scores = []
        
        for idx in remaining_indices:
            # How relevant is this to the query?
            relevance = relevance_scores[idx]
            
            # How similar is this to what we've already selected?
            similarities_to_selected = [
                np.dot(candidate_norms[idx], selected / np.linalg.norm(selected))
                for selected in selected_embeddings
            ]
            max_similarity = max(similarities_to_selected) if similarities_to_selected else 0
            
            # MMR score: Reward relevance, penalize similarity to already-selected
            mmr_score = relevance_weight * relevance - (1 - relevance_weight) * max_similarity
            mmr_scores.append((mmr_score, idx))
        
        # Select the item with the best MMR score
        _, best_idx = max(mmr_scores)
        selected_indices.append(candidate_indices[best_idx])
        selected_embeddings.append(candidate_embeddings[best_idx])
        remaining_indices.remove(best_idx)
    
    return selected_indices


# ============================================================================
# STEP 4: MEMORY SYSTEM (Style preferences)
# ============================================================================

def load_memory(index_name):
    """
    Load style preferences and constraints from JSON file.
    
    Memory contains:
        - preferred_hashtags: Hashtags to use consistently
        - recurring_themes: Topics/phrases that define brand voice
        - banned_phrases: Overused jargon to avoid
    
    This helps maintain consistency across all generated posts.
    """
    memory_file = MEMORY_DIR / f"{index_name}_memory.json"
    
    # Default empty memory
    default = {
        "preferred_hashtags": [],
        "recurring_themes": [],
        "banned_phrases": []
    }
    
    if memory_file.exists():
        try:
            with open(memory_file, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception:
            return default
    
    return default


def save_memory(index_name, memory_data):
    """Save updated memory back to JSON file."""
    memory_file = MEMORY_DIR / f"{index_name}_memory.json"
    
    with open(memory_file, 'w', encoding='utf-8') as file:
        json.dump(memory_data, file, indent=2, ensure_ascii=False)


# ============================================================================
# STEP 5: BUILD INDEX (One-time setup)
# ============================================================================

def build_index(posts_file, index_name):
    """
    Build a FAISS vector index from your LinkedIn posts.
    
    This is a ONE-TIME setup per person/brand:
        1. Load posts from file
        2. Break into chunks (preserving style)
        3. Convert chunks to embeddings
        4. Store in FAISS for fast similarity search
        5. Save texts and metadata for later retrieval
    
    Args:
        posts_file: Path to your posts.txt file
        index_name: Name for this index (e.g., "ikf", "john_doe")
    """
    print(f"\n🔄 Loading posts from: {posts_file}")
    posts, metadata = load_posts(posts_file)
    print(f"✅ Loaded {len(posts)} posts")
    
    print("\n🔄 Breaking posts into chunks (preserving style)...")
    chunks = chunk_posts(posts)
    print(f"✅ Created {len(chunks)} chunks")
    
    print("\n🔄 Converting chunks to embeddings...")
    embeddings = get_embeddings(chunks)
    embedding_dimension = embeddings.shape[1]
    print(f"✅ Generated {len(embeddings)} embeddings (dimension: {embedding_dimension})")
    
    print("\n🔄 Building FAISS index...")
    # IndexFlatL2 = exact search using L2 distance (good for <10K vectors)
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings)
    print(f"✅ Index built with {index.ntotal} vectors")
    
    # Save everything
    save_dir = VECTOR_STORE_DIR / index_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, str(save_dir / "index.faiss"))
    
    # Save original text chunks
    with open(save_dir / "texts.txt", 'w', encoding='utf-8') as file:
        file.write('\n---\n'.join(chunks))
    
    # Save metadata
    with open(save_dir / "metadata.json", 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2)
    
    print(f"\n✅ SUCCESS! Index saved to: {save_dir}")
    print(f"   - index.faiss: FAISS vector index")
    print(f"   - texts.txt: Original text chunks")
    print(f"   - metadata.json: Post metadata")


# ============================================================================
# STEP 6: GENERATE POST (The main function!)
# ============================================================================

def generate_post(index_name, person_name, topic):
    """
    Generate a LinkedIn post that matches your writing style.
    
    This is where the magic happens:
        1. Load your pre-built FAISS index
        2. Search for similar posts to the topic
        3. Use MMR to select 5 diverse examples
        4. Feed those examples to the LLM as style guide
        5. Generate a new post matching that style
        6. Check for plagiarism
        7. Save the result
    
    Args:
        index_name: Which index to use (e.g., "ikf")
        person_name: Author name (e.g., "IKF Team")
        topic: What to write about (e.g., "AI in marketing")
    """
    # === LOAD INDEX ===
    print(f"\n🔄 Loading index: {index_name}")
    index_dir = VECTOR_STORE_DIR / index_name
    index = faiss.read_index(str(index_dir / "index.faiss"))
    
    with open(index_dir / "texts.txt", 'r', encoding='utf-8') as file:
        chunks = file.read().split('\n---\n')
    
    # Load metadata with keywords
    with open(index_dir / "metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"✅ Loaded {index.ntotal} vectors from index")
    
    # === LOAD STYLE PREFERENCES ===
    memory = load_memory(index_name)
    
    # === STEP 1: SEARCH ===
    print(f"\n🔍 Searching for posts similar to: '{topic}'")
    query_embedding = get_embeddings([topic], input_type="query")
    
    # Get 15 candidates (we'll narrow down to 5 with MMR)
    num_candidates = 15
    distances, indices = index.search(query_embedding, min(num_candidates, index.ntotal))
    
    # === STEP 2: GET EMBEDDINGS FOR CANDIDATES ===
    candidate_texts = [chunks[idx] for idx in indices[0]]
    candidate_embeddings = get_embeddings(candidate_texts, input_type="passage")
    
    # === STEP 3: APPLY MMR FOR DIVERSITY ===
    print("🎯 Applying MMR to select diverse examples...")
    selected_indices = mmr_selection(
        query_embedding[0], 
        candidate_embeddings, 
        indices[0], 
        num_to_select=5, 
        relevance_weight=0.7  # 70% relevance, 30% diversity
    )
    context_examples = [chunks[idx] for idx in selected_indices]
    
    # Extract keywords from selected posts' metadata
    selected_keywords = []
    for idx in selected_indices:
        # Find the metadata entry for this chunk
        for meta in metadata:
            if meta.get('keywords'):
                selected_keywords.extend(meta['keywords'])
    # Get unique keywords
    unique_keywords = list(set(selected_keywords))[:15]  # Top 15 keywords
    
    print(f"✅ Selected {len(context_examples)} diverse examples")
    print(f"🏷️  Extracted {len(unique_keywords)} relevant keywords from metadata\n")
    
    # === SHOW WHAT WE FOUND ===
    print("=" * 60)
    print("📚 RETRIEVED EXAMPLES (These guide the writing style):")
    print("=" * 60)
    for i, example in enumerate(context_examples, 1):
        preview = example[:150] + "..." if len(example) > 150 else example
        print(f"\nExample {i}:\n{preview}")
    
    print("\n" + "=" * 60)
    print("🏷️  EXTRACTED KEYWORDS (From selected posts):")
    print("=" * 60)
    print(", ".join(unique_keywords))
    print("\n" + "=" * 60)
    
    # === STEP 4: BUILD THE PROMPT ===
    print("\n✍️  Generating post...")
    
    # Prepare examples for the LLM
    context_text = "\n\n".join(f"Example {i+1}: {ex}" for i, ex in enumerate(context_examples))
    
    # Add style preferences from memory
    memory_instructions = ""
    if memory.get("preferred_hashtags"):
        memory_instructions += f"\nPreferred hashtags: {', '.join(memory['preferred_hashtags'])}"
    if memory.get("recurring_themes"):
        memory_instructions += f"\nRecurring themes: {', '.join(memory['recurring_themes'])}"
    if memory.get("banned_phrases"):
        memory_instructions += f"\nAvoid these phrases: {', '.join(memory['banned_phrases'])}"
    
    # Add keyword context from metadata
    keyword_context = ""
    if unique_keywords:
        keyword_context = f"\n\nCONTEXT-AWARE KEYWORDS (Use 3-5 of these to shape tone & vocabulary):\n{', '.join(unique_keywords)}"
    
    # System prompt: How to write
    system_prompt = f"""You are a LinkedIn post writer for {person_name}. 
Write authentic, professional posts that match their unique style.

STRICT FORMAT RULES:
1. Length: 120-220 words exactly
2. Structure: Hook → Insight → Example → Reflection
3. NO emojis allowed
4. End with keywords in brackets: [keyword1, keyword2, keyword3]
5. Add maximum 4 hashtags on new line: hashtag#Tag1 hashtag#Tag2
6. Optional signature if appropriate{memory_instructions}{keyword_context}

Example structure:
[Attention-grabbing opening...]

[Your perspective or insight...]

[Concrete example or story...]

[Reflection or call to action...]

[keyword1, keyword2, keyword3, keyword4, keyword5]

hashtag#Topic1 hashtag#Topic2 hashtag#Topic3 hashtag#Topic4"""
    
    # User prompt: What to write
    user_prompt = f"""Write a LinkedIn post about: {topic}

Study these examples to match the writing style:

{context_text}

Now write the post (120-220 words):"""
    
    # === STEP 5: CALL THE LLM ===
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    
    generated_post = response.choices[0].message.content.strip()
    
    # === STEP 6: VALIDATE ===
    print("\n🔍 Running quality checks...")
    
    # Check for plagiarism
    is_plagiarized, matched_text = check_plagiarism(generated_post, context_examples)
    
    # Count words (exclude keywords section)
    text_without_keywords = generated_post.split('[')[0]
    word_count = len(text_without_keywords.split())
    
    # === SHOW RESULTS ===
    print("\n" + "=" * 60)
    print("✨ GENERATED POST:")
    print("=" * 60)
    print(generated_post)
    print("=" * 60)
    
    print(f"\n📊 Quality Metrics:")
    print(f"   • Word count: {word_count} words (target: 120-220)")
    print(f"   • Plagiarism check: {'⚠️  FAILED' if is_plagiarized else '✅ PASSED'}")
    
    if is_plagiarized:
        print(f"   • Matched text: {matched_text[:100]}...")
    
    # === STEP 7: SAVE ===
    output_file = OUTPUT_DIR / f"{index_name}_post.txt"
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(generated_post)
        file.write(f"\n\n--- Metadata ---\n")
        file.write(f"Author: {person_name}\n")
        file.write(f"Topic: {topic}\n")
        file.write(f"Word count: {word_count}\n")
        file.write(f"Examples used: {len(context_examples)} (MMR-selected)\n")
        file.write(f"Plagiarism check: {'FAILED' if is_plagiarized else 'PASSED'}\n")
    
    print(f"\n💾 Saved to: {output_file}")


# ============================================================================
# STEP 7: PLAGIARISM DETECTION
# ============================================================================

def check_plagiarism(generated_post, source_examples, threshold=25):
    """
    Check if the generated post copied text from training examples.
    
    Algorithm:
        - Look for sequences of 25+ consecutive words that match exactly
        - This catches direct copying while allowing similar ideas/phrasing
    
    Why 25 words?
        - 10 words: Too short (common phrases)
        - 50 words: Too long (rare exact matches)
        - 25 words: Sweet spot for detecting real copying
    
    Args:
        generated_post: The newly generated text
        source_examples: The training examples used
        threshold: Number of consecutive words to check (default: 25)
    
    Returns:
        (is_plagiarized, matched_snippet) tuple
    """
    # Clean up the generated post
    post_text = generated_post.lower()
    post_text = post_text.split('[')[0]  # Remove keywords section
    post_words = post_text.split()
    
    # Check each source example
    for source in source_examples:
        source_words = source.lower().split()
        
        # Sliding window through generated post
        for i in range(len(post_words) - threshold + 1):
            generated_sequence = ' '.join(post_words[i:i + threshold])
            
            # Sliding window through source
            for j in range(len(source_words) - threshold + 1):
                source_sequence = ' '.join(source_words[j:j + threshold])
                
                # Found an exact match!
                if generated_sequence == source_sequence:
                    return True, source
    
    # No plagiarism detected
    return False, None


def generate_without_rag(person_name, topic):
    """Generate post WITHOUT RAG for comparison."""
    system_prompt = """You are a LinkedIn post writer for IKF. Write authentic, professional posts.

STRICT FORMAT REQUIREMENTS:
1. Write exactly 120-220 words
2. Structure: Hook → Perspective → Concrete illustration → Reflection
3. NO emojis allowed
4. End with keywords in square brackets: [keyword1, keyword2]
5. Add maximum 4 hashtags on a new line starting with hashtag#"""
    
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a LinkedIn post for {person_name} about: {topic}"}
        ],
        temperature=temperature
    )
    
    return response.choices[0].message.content.strip()


def evaluate(index_name, person_name, topics):
    """Run evaluation: compare with and without RAG."""
    import json
    
    print("\n" + "="*60)
    print("🧪 EVALUATION: With RAG vs Without RAG")
    print("="*60)
    
    # Load index
    index_dir = VECTOR_STORE_DIR / index_name
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "texts.txt", 'r', encoding='utf-8') as f:
        chunks = f.read().split('\n---\n')
    
    results = []
    
    for topic in topics:
        print(f"\n📝 Topic: {topic}")
        print("-" * 60)
        
        # With RAG (k=5 with MMR)
        print("\n[WITH RAG]")
        query_emb = get_embeddings([topic], input_type="query")
        
        # Get candidates for MMR
        k_candidates = 15
        distances, indices = index.search(query_emb, min(k_candidates, index.ntotal))
        candidate_chunks = [chunks[idx] for idx in indices[0]]
        candidate_embs = get_embeddings(candidate_chunks, input_type="passage")
        
        # Apply MMR (k=5)
        selected_indices = mmr_selection(query_emb[0], candidate_embs, indices[0], k=5, lambda_param=0.7)
        context = [chunks[idx] for idx in selected_indices]
        
        print(f"Retrieved {len(context)} diverse examples (MMR)")
        
        # Generate with RAG
        context_text = "\n\n".join(f"Example {i+1}: {c}" for i, c in enumerate(context))
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        system_prompt = """You are a LinkedIn post writer for IKF. Write authentic, professional posts in IKF's style.
Format: 120-220 words, NO emojis, keywords in brackets, max 4 hashtags."""
        
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Write a LinkedIn post for {person_name} about: {topic}\n\nStyle examples:\n{context_text}\n\nWrite the post:"}
            ],
            temperature=temperature
        )
        post_with_rag = response.choices[0].message.content.strip()
        is_plagiarized, _ = check_plagiarism(post_with_rag, context)
        
        # Without RAG
        print("\n[WITHOUT RAG]")
        post_without_rag = generate_without_rag(person_name, topic)
        
        results.append({
            'topic': topic,
            'with_rag': post_with_rag,
            'without_rag': post_without_rag,
            'plagiarism_check': 'PASSED' if not is_plagiarized else 'FAILED',
            'retrieved_snippets': len(context)
        })
        
        print(f"\nWith RAG preview: {post_with_rag[:100]}...")
        print(f"Without RAG preview: {post_without_rag[:100]}...")
        print(f"Plagiarism check: {results[-1]['plagiarism_check']}")
    
    # Save results (append to existing evaluations)
    output_file = OUTPUT_DIR / f"evaluation_{index_name}.json"
    
    # Load existing results if file exists
    existing_results = []
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except:
            existing_results = []
    
    # Append new results
    all_results = existing_results + results
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Evaluation complete! {len(results)} new results added. Total: {len(all_results)} evaluations in {output_file}")
    return results


def main():
    if len(sys.argv) < 2:
        print("""
Simple LinkedIn RAG Agent

Commands:
  python simple_rag.py build <posts_file> <index_name>
  python simple_rag.py generate <index_name> <person_name> <topic>
  python simple_rag.py evaluate <index_name> <person_name> <topic1> [topic2] [topic3]
  python simple_rag.py memory <index_name> [show|set]
  python simple_rag.py analytics <index_name>

Examples:
  python simple_rag.py build data/posts/ikf.txt ikf
  python simple_rag.py generate ikf "John Doe" "AI in marketing"
  python simple_rag.py evaluate ikf "IKF Team" "SEO strategy" "Brand storytelling"
  python simple_rag.py memory ikf show
  python simple_rag.py analytics ikf
""")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "build" and len(sys.argv) == 4:
        build_index(sys.argv[2], sys.argv[3])
    elif cmd == "generate" and len(sys.argv) == 5:
        generate_post(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "evaluate" and len(sys.argv) >= 5:
        topics = sys.argv[4:]
        evaluate(sys.argv[2], sys.argv[3], topics)
    elif cmd == "memory" and len(sys.argv) >= 3:
        manage_memory_cli(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "show")
    elif cmd == "analytics" and len(sys.argv) == 3:
        show_analytics(sys.argv[2])
    else:
        print("Invalid command or arguments")


def manage_memory_cli(index_name, action):
    """Manage memory settings via CLI."""
    import json
    
    if action == "show":
        memory = load_memory(index_name)
        print("\n" + "="*60)
        print(f"📝 MEMORY for {index_name}")
        print("="*60)
        print(json.dumps(memory, indent=2, ensure_ascii=False))
        print("="*60)
    
    elif action == "set":
        memory = load_memory(index_name)


def show_analytics(index_name):
    """Display thematic analytics from metadata keywords."""
    import json
    from collections import Counter
    
    index_dir = VECTOR_STORE_DIR / index_name
    
    # Load metadata
    with open(index_dir / "metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print("\n" + "="*80)
    print(f"📊 THEMATIC ANALYTICS - {index_name.upper()}")
    print("="*80)
    
    # Collect all keywords
    all_keywords = []
    keyword_by_post = {}
    
    for meta in metadata:
        post_id = meta.get('post_id', 'unknown')
        keywords = meta.get('keywords', [])
        all_keywords.extend(keywords)
        keyword_by_post[post_id] = keywords
    
    # Count keyword frequency
    keyword_counts = Counter(all_keywords)
    total_keywords = len(all_keywords)
    unique_keywords = len(keyword_counts)
    
    print(f"\n📈 OVERVIEW:")
    print(f"   • Total Posts: {len(metadata)}")
    print(f"   • Total Keywords: {total_keywords}")
    print(f"   • Unique Keywords: {unique_keywords}")
    print(f"   • Avg Keywords/Post: {total_keywords/len(metadata):.1f}")
    
    # Top themes
    print(f"\n🏆 TOP 15 THEMES (Most Frequent Keywords):")
    print("-" * 80)
    for i, (keyword, count) in enumerate(keyword_counts.most_common(15), 1):
        percentage = (count / len(metadata)) * 100
        bar = "█" * int(percentage / 5)
        print(f"{i:2}. {keyword:40} | Count: {count:2} | {bar} {percentage:.0f}%")
    
    # Keyword distribution by post
    print(f"\n📋 KEYWORD DISTRIBUTION BY POST:")
    print("-" * 80)
    for post_id, keywords in sorted(keyword_by_post.items()):
        print(f"\nPost #{post_id} ({len(keywords)} keywords):")
        print(f"   {', '.join(keywords[:8])}")
        if len(keywords) > 8:
            print(f"   {', '.join(keywords[8:])}")
    
    # Topic clustering (simple grouping)
    print(f"\n🎯 TOPIC CLUSTERS (Keyword Categories):")
    print("-" * 80)
    
    # Define topic categories
    topics = {
        "SEO & Digital Marketing": ["SEO", "search engine optimization", "organic growth", "online visibility", "digital marketing", "digital campaign"],
        "Branding & Storytelling": ["brand experience", "brand strategy", "luxury storytelling", "storytelling with impact", "visual storytelling", "people-centric branding"],
        "Content Strategy": ["content creation", "platform-specific content", "reel marketing", "influencer marketing", "content marketing"],
        "Marketing Trends": ["2025 digital trends", "marketing in 2025", "future of marketing", "new audience reach"],
        "Technology & Innovation": ["AI", "technology", "innovation", "digital transformation"]
    }
    
    for category, keywords_list in topics.items():
        matches = [kw for kw in all_keywords if any(term in kw.lower() for term in [k.lower() for k in keywords_list])]
        if matches:
            percentage = (len(matches) / total_keywords) * 100
            print(f"\n{category}:")
            print(f"   Coverage: {len(matches)}/{total_keywords} keywords ({percentage:.1f}%)")
            print(f"   Keywords: {', '.join(list(set(matches))[:5])}")
    
    print("\n" + "="*80)
    print("✅ Analytics complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
