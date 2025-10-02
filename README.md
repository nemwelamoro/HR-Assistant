# HR AI Knowledge Base Setup

This project processes HR documents from multiple Supabase storage buckets and creates a searchable knowledge base with vector embeddings using Google Gemini.

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure Environment
Edit the `.env` file with your credentials:
```
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_service_role_key
GOOGLE_API_KEY=your_google_api_key
```

### 3. Update Database Schema for Gemini
Run the SQL in `schema_update_gemini.sql` in your Supabase SQL Editor to update for 768-dimensional embeddings. **Not a must** 

### 4. Set Up Database Functions
Run the SQL in `supabase_functions.sql` in your Supabase SQL Editor.

### 4. Process Your Documents
```powershell
python process_hr_documents.py
```

## 📁 File Structure

- `knowledge_base.py` - Main client for Supabase and OpenAI integration
- `document_processor.py` - Document parsing and text extraction
- `process_hr_documents.py` - Main script to process all documents
- `supabase_functions.sql` - Database functions for vector search
- `.env` - Environment configuration
- `requirements.txt` - Python dependencies

## 🔧 What It Does

1. **Connects to Multiple Buckets** - Downloads all files from your HR storage buckets:
   - `hr_policies` - Company policies and procedures
   - `hr-docs` - General HR documents  
   - `hr-templates` - Document templates
   - `hr-reports` - Reports and analytics
2. **Extracts Text** - Supports PDF, DOCX, and TXT files
3. **Creates Chunks** - Splits documents into searchable pieces (500 tokens each)
4. **Generates Embeddings** - Uses Google Gemini to create 768-dimensional vector embeddings
5. **Stores in Database** - Saves to our `kb_article` and `kb_chunk` tables
6. **Enables Search** - Vector similarity search across all content with bucket-aware tagging

## 📊 Usage Example

After processing, you can search your knowledge base:

```python
from knowledge_base import HRKnowledgeBaseClient

# Initialize client
kb = HRKnowledgeBaseClient()

# Search for information
results = kb.search_similar_chunks("What is the leave policy?")

# Print results
for result in results:
    print(f"Source: {result['article_title']}")
    print(f"Similarity: {result['similarity']:.2f}")
    print(f"Content: {result['content']}")
    print("---")
```

## 🎯 Expected Output

When you run `process_hr_documents.py`, you'll see:
- Files being processed one by one
- Text extraction statistics
- Embedding generation progress  
- Final summary with totals

The script will create searchable chunks from all your HR documents and enable semantic search across your entire knowledge base.

## 🔍 Testing

The script automatically tests search functionality with sample queries like:
- "What is the company leave policy?"
- "How do I request time off?"
- "Performance review guidelines"

## ⚡ Performance

- Processes documents in batches for efficiency
- Skips already-processed files on re-runs
- Handles various document formats automatically
- Provides detailed logging and error handling