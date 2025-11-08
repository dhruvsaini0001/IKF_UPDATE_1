# LinkedIn RAG Agent 🚀

An intelligent LinkedIn post generator that learns and replicates your unique writing style using Retrieval-Augmented Generation (RAG) with FAISS vector search and NVIDIA NIM.

## 🌟 Features

- **Style-Matching AI**: Analyzes your past LinkedIn posts to generate new content that sounds authentically like you
- **Vector Search with FAISS**: Lightning-fast semantic search to find relevant writing examples
- **MMR Algorithm**: Maximal Marginal Relevance ensures diverse, non-repetitive examples
- **Anti-Plagiarism Detection**: Automatic checks to prevent copying from source material
- **Memory System**: Maintains brand voice consistency with preferred hashtags and themes
- **Streamlit UI**: User-friendly web interface for easy post generation
- **Keyword Extraction**: Automatically extracts and uses relevant keywords from historical posts
- **Evaluation Mode**: Compare RAG vs non-RAG generated content
- **Analytics Dashboard**: Visualize themes and keyword patterns in your content

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [CLI Commands](#cli-commands)
  - [Streamlit UI](#streamlit-ui)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Compatibility](#api-compatibility)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

The system uses a **RAG (Retrieval-Augmented Generation)** pipeline:

```
Your LinkedIn Posts
        ↓
    Chunking (350-token smart chunks)
        ↓
    Embeddings (text-embedding-3-small / nvidia/nv-embedqa-e5-v5)
        ↓
    FAISS Vector Store (IndexFlatL2)
        ↓
    Query → Semantic Search → MMR Selection
        ↓
    5 Diverse Examples → LLM Prompt
        ↓
    Generated Post (matches your style)
```

**Key Components:**

1. **Vector Database**: FAISS for efficient similarity search
2. **Embeddings**: NVIDIA NIM or OpenAI embeddings
3. **LLM**: Meta Llama 3.1 (via NVIDIA NIM) or GPT-4
4. **MMR Algorithm**: Balances relevance with diversity (70% relevance, 30% diversity)
5. **Memory System**: JSON-based style preferences storage
6. **Plagiarism Checker**: Detects 25+ consecutive word matches

## 💻 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Step 1: Clone or Download

```bash
cd d:\ikf_update_1
```

### Step 2: Install Dependencies

```bash
pip install streamlit numpy faiss-cpu openai python-dotenv nltk
```

**Package Breakdown:**
- `streamlit`: Web UI framework
- `numpy`: Numerical computations
- `faiss-cpu`: Vector similarity search (use `faiss-gpu` for GPU acceleration)
- `openai`: API client (compatible with OpenAI and NVIDIA NIM)
- `python-dotenv`: Environment variable management
- `nltk`: Natural language processing for sentence tokenization

### Step 3: Download NLTK Data (First Run Only)

The system will automatically download required NLTK data on first use, but you can pre-download:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## ⚙️ Configuration

### Environment Variables

Create or modify `.env` file in the project root:

```env
# API Configuration (Choose one)

# Option 1: NVIDIA NIM (FREE)
OPENAI_API_KEY=nvapi-your-key-here
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=meta/llama-3.1-8b-instruct
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5

# Option 2: OpenAI
OPENAI_API_KEY=sk-your-openai-key
# OPENAI_BASE_URL=  # Leave blank for OpenAI
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# RAG Configuration
CHUNK_SIZE=200
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=5
TEMPERATURE=0.7

# Post Generation Parameters
MIN_POST_LENGTH=120
MAX_POST_LENGTH=220
MAX_HASHTAGS=4

# Anti-Plagiarism
MAX_CONSECUTIVE_WORDS=25
```

### Getting API Keys

**NVIDIA NIM (Free Option):**
1. Visit [NVIDIA NIM](https://build.nvidia.com/)
2. Sign up for free account
3. Generate API key from dashboard

**OpenAI:**
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create account and add payment method
3. Generate API key from API settings

## 🚀 Usage

### CLI Commands

#### 1. Build Index (First Time Setup)

Convert your LinkedIn posts into a searchable vector database:

```bash
python simple_rag.py build data/posts/ikf.txt ikf
```

**Parameters:**
- `data/posts/ikf.txt`: Path to your posts file
- `ikf`: Index name (used to identify this collection)

**Input File Format (`data/posts/ikf.txt`):**
```
Your first LinkedIn post content here...

[keyword1, keyword2, keyword3]

---

Your second post...

[another keyword, more keywords]

---
```

Optional metadata per post:
```
date: 2024-11-08
link: https://linkedin.com/post/...

Post content here...

[keywords here]

---
```

#### 2. Generate Post

Create a new LinkedIn post matching your style:

```bash
python simple_rag.py generate ikf "IKF Team" "AI in digital marketing"
```

**Parameters:**
- `ikf`: Index name to use
- `"IKF Team"`: Author/brand name
- `"AI in digital marketing"`: Topic to write about

**Output:**
- Generated post printed to console
- Saved to `outputs/ikf_post.txt`
- Added to `outputs/evaluation_ikf.json`

#### 3. Evaluate (Compare RAG vs No RAG)

Test RAG effectiveness with multiple topics:

```bash
python simple_rag.py evaluate ikf "IKF Team" "SEO trends 2025" "Content marketing" "Brand storytelling"
```

**Output:**
- Side-by-side comparison of RAG vs non-RAG posts
- Metrics: word count, plagiarism check, retrieval quality
- Saved to `outputs/evaluation_ikf.json`

#### 4. View Memory Settings

Display stored style preferences:

```bash
python simple_rag.py memory ikf show
```

**Memory Structure:**
```json
{
  "preferred_hashtags": ["#DigitalMarketing", "#ContentStrategy"],
  "recurring_themes": ["authentic storytelling", "data-driven"],
  "banned_phrases": ["synergy", "think outside the box"]
}
```

#### 5. Analytics Dashboard

View keyword trends and thematic analysis:

```bash
python simple_rag.py analytics ikf
```

**Displays:**
- Top 15 most frequent keywords
- Keyword distribution per post
- Topic clusters and categories
- Coverage statistics

### Streamlit UI

Launch the web interface for easier interaction:

```bash
streamlit run app.py
```

**Features:**
- 📂 Select from available indexes
- 👤 Input person/brand name
- 🏢 Add company context
- 💡 Enter topic with optional bullet points
- 🔍 View retrieved snippets used for style matching
- ✍️ Generate post with one click
- 📊 Real-time metrics (word count, plagiarism check)
- 📥 Download generated post
- 🧠 View memory settings

**UI Workflow:**
1. Select your index from sidebar
2. Fill in person name, company, and topic
3. Optionally add 3 key points
4. Click "Generate Post"
5. Review retrieved snippets and generated content
6. Check metrics and plagiarism status
7. Download or copy the post

## 📁 Project Structure

```
ikf_update_1/
│
├── app.py                      # Streamlit web UI
├── simple_rag.py              # Core RAG engine & CLI
├── .env                       # Configuration (API keys)
├── README.md                  # This file
│
├── data/
│   ├── posts/
│   │   └── ikf.txt           # Your LinkedIn posts (input)
│   │
│   ├── vector_store/
│   │   └── ikf/              # FAISS index for 'ikf'
│   │       ├── index.faiss   # Vector database
│   │       ├── texts.txt     # Original text chunks
│   │       └── metadata.json # Post metadata & keywords
│   │
│   └── memory/
│       └── ikf_memory.json   # Style preferences
│
└── outputs/
    ├── ikf_post.txt          # Latest generated post (CLI)
    ├── ikf_post_streamlit.txt # Latest from Streamlit
    └── evaluation_ikf.json   # All generation history
```

## 🔧 How It Works

### 1. Index Building

**Input**: Your LinkedIn posts in a text file

**Process:**
1. **Load Posts**: Parse text file, extract metadata (dates, links, keywords)
2. **Smart Chunking**: 
   - Split by paragraphs (preserves structure)
   - Use NLTK for accurate sentence tokenization
   - Combine sentences up to ~350 tokens (max 512)
   - Filter fragments <20 words
3. **Generate Embeddings**: Convert chunks to vectors (1536-dim for OpenAI, 1024-dim for NVIDIA)
4. **Build FAISS Index**: IndexFlatL2 for exact L2 distance search
5. **Save**: Store index, texts, and metadata

**Why This Matters**: Good chunking preserves your writing style while fitting within model limits.

### 2. Post Generation

**Input**: Topic to write about

**Process:**

1. **Query Embedding**: Convert topic to vector
2. **Candidate Retrieval**: Get 15 most similar chunks from FAISS
3. **MMR Selection**: 
   - Select most relevant chunk first
   - Then iteratively select chunks that are:
     - **Relevant** to the query (70% weight)
     - **Diverse** from already selected (30% weight)
   - Result: 5 diverse, relevant examples
4. **Keyword Extraction**: Pull keywords from selected posts' metadata
5. **Prompt Construction**:
   ```
   System: You are writing for [Person]. Follow these rules...
   Memory: Use these hashtags, avoid these phrases...
   Keywords: [extracted from similar posts]
   Examples: [5 MMR-selected snippets]
   User: Write about [topic]
   ```
6. **LLM Generation**: Temperature 0.7 for creativity
7. **Quality Checks**:
   - Word count validation (120-220 words)
   - Anti-plagiarism check (25+ word sequences)
8. **Save**: Store in outputs with metadata

**Formula Details:**

MMR Score = λ × Relevance - (1-λ) × MaxSimilarity

Where:
- λ = 0.7 (relevance weight)
- Relevance = cosine similarity to query
- MaxSimilarity = highest similarity to already-selected chunks

### 3. Anti-Plagiarism

**Algorithm:**
- Sliding window through generated post (25-word sequences)
- Compare against all source examples
- Flag if exact match found
- **Why 25 words?**: Balance between false positives and detection

**Example:**
```
Generated: "Content marketing drives engagement through authentic storytelling..."
Source: "Content marketing drives engagement through authentic storytelling..."
                    ↑_______________25 words_______________↑
Result: ⚠️ PLAGIARISM DETECTED
```

### 4. Memory System

Maintains consistency across generations:

```json
{
  "preferred_hashtags": ["#BrandStrategy", "#ContentMarketing"],
  "recurring_themes": ["data-driven decisions", "authentic engagement"],
  "banned_phrases": ["leverage synergies", "paradigm shift"]
}
```

**Usage**: Automatically injected into LLM system prompt.

## 🔌 API Compatibility

The system works with any OpenAI-compatible API:

### Supported Providers

| Provider | LLM Models | Embedding Models | Cost |
|----------|-----------|------------------|------|
| **NVIDIA NIM** | meta/llama-3.1-8b-instruct<br>meta/llama-3.1-70b-instruct | nvidia/nv-embedqa-e5-v5 | **FREE** |
| **OpenAI** | gpt-4o-mini<br>gpt-4o<br>gpt-3.5-turbo | text-embedding-3-small<br>text-embedding-3-large | Paid |
| **Azure OpenAI** | Same as OpenAI | Same as OpenAI | Paid |
| **Local (Ollama)** | llama3, mistral, etc. | Custom | Free |

### Configuration Examples

**NVIDIA NIM (Recommended for Free Tier):**
```env
OPENAI_API_KEY=nvapi-xxx
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=meta/llama-3.1-8b-instruct
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
```

**OpenAI:**
```env
OPENAI_API_KEY=sk-xxx
# OPENAI_BASE_URL=  # Leave blank
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

**Azure OpenAI:**
```env
OPENAI_API_KEY=your-azure-key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
LLM_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002
```

## 📚 Examples

### Example 1: Initial Setup

```bash
# Step 1: Create posts file
# Add your LinkedIn posts to data/posts/my_posts.txt

# Step 2: Build index
python simple_rag.py build data/posts/my_posts.txt my_brand

# Output:
# ✅ Loaded 10 posts
# ✅ Created 45 chunks
# ✅ Generated 45 embeddings (dimension: 1024)
# ✅ Index built with 45 vectors
```

### Example 2: Generate Single Post

```bash
python simple_rag.py generate my_brand "John Doe" "Future of AI in healthcare"

# Output:
# 📚 RETRIEVED EXAMPLES (5 diverse snippets shown)
# 🏷️ EXTRACTED KEYWORDS: healthcare innovation, patient care, digital health...
# ✨ GENERATED POST: [your styled post]
# 📊 Word count: 185 words
# ✅ Plagiarism check: PASSED
# 💾 Saved to: outputs/my_brand_post.txt
```

### Example 3: Evaluation Mode

```bash
python simple_rag.py evaluate my_brand "Jane Smith" "Content strategy 2025" "Social media trends"

# Output shows side-by-side comparison:
# [WITH RAG] - matches your style
# [WITHOUT RAG] - generic style
# Results saved to outputs/evaluation_my_brand.json
```

### Example 4: Using Streamlit UI

```bash
streamlit run app.py

# Open browser to http://localhost:8501
# Select index: my_brand
# Enter: Person="Jane Smith", Topic="AI trends"
# Click "Generate Post"
# Review snippets and generated content
# Download post
```

## 🐛 Troubleshooting

### Issue: "No indexes found"

**Cause**: Haven't built an index yet

**Solution**:
```bash
python simple_rag.py build data/posts/your_posts.txt your_name
```

### Issue: "API key not found"

**Cause**: Missing or incorrect `.env` file

**Solution**:
1. Create `.env` in project root
2. Add `OPENAI_API_KEY=your-key`
3. Restart application

### Issue: "NLTK punkt tokenizer not found"

**Cause**: Missing NLTK data

**Solution**:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Issue: Slow generation (>30 seconds)

**Cause**: Using large LLM model or slow API

**Solution**:
- Switch to faster model: `LLM_MODEL=gpt-4o-mini` or `meta/llama-3.1-8b-instruct`
- Reduce `TOP_K_RETRIEVAL` in `.env`
- Use `faiss-gpu` instead of `faiss-cpu`

### Issue: Posts don't match my style

**Cause**: Insufficient training data or poor examples

**Solution**:
1. Add more posts to training file (20+ recommended)
2. Check retrieved snippets - are they relevant?
3. Adjust `relevance_weight` in `mmr_selection()` (increase for more relevance)
4. Update memory file with preferred phrases/hashtags

### Issue: Plagiarism warnings on original content

**Cause**: Threshold too sensitive (25 words)

**Solution**:
Edit `simple_rag.py`, line in `check_plagiarism()`:
```python
def check_plagiarism(generated_post, source_examples, threshold=30):  # Increase from 25
```

## 🎯 Best Practices

### For Best Results:

1. **Quality Training Data**: 
   - Use 20+ well-written posts
   - Include variety (different topics, lengths)
   - Remove irrelevant metadata

2. **Effective Prompts**:
   ```
   ✅ Good: "AI-driven personalization in e-commerce for small businesses"
   ❌ Bad: "AI"
   ```

3. **Memory Maintenance**:
   - Update `banned_phrases` to avoid clichés
   - Set `preferred_hashtags` for consistency
   - Define `recurring_themes` for brand voice

4. **Regular Evaluation**:
   - Run evaluation mode monthly
   - Compare RAG vs non-RAG quality
   - Adjust parameters based on results

## 📊 Performance Metrics

Typical performance (with 50 training posts):

| Metric | Value |
|--------|-------|
| Index Build Time | 10-30 seconds |
| Query Response Time | 2-5 seconds |
| Post Generation Time | 5-15 seconds |
| Memory Usage | ~200 MB |
| Storage (per index) | ~10 MB |

## 🤝 Contributing

This is a personal/organizational project. For improvements:

1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit pull request with clear description

## 📄 License

Proprietary - IKF Creative Agency

## 🙏 Acknowledgments

- **NVIDIA NIM**: Free API access for embeddings and LLMs
- **FAISS**: Meta's efficient similarity search library
- **OpenAI**: API compatibility standard
- **Streamlit**: Rapid UI development framework

## 📞 Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review `.env` configuration
- Verify API key validity
- Check `outputs/` directory for detailed logs

---

**Built with ❤️ by IKF Creative Agency**

*Powered by NVIDIA NIM, FAISS, and RAG technology*
