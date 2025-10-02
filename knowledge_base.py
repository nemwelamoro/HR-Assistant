import os
import logging
from typing import List, Dict, Optional
from supabase import create_client, Client
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRKnowledgeBaseClient:
    """Client for managing HR knowledge base with vector search"""
    
    def __init__(self):
        """Initialize Supabase and Gemini clients"""
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 500))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
        self.batch_size = int(os.getenv("BATCH_SIZE", 10))
        
        # Storage buckets
        self.buckets = {
            'policies': os.getenv("BUCKET_POLICIES", "hr_policies"),
            'docs': os.getenv("BUCKET_DOCS", "hr-docs"), 
            'templates': os.getenv("BUCKET_TEMPLATES", "hr-templates"),
            'reports': os.getenv("BUCKET_REPORTS", "hr-reports")
        }
        
        logger.info("HR Knowledge Base Client initialized with Gemini")
    
    def list_storage_files(self, bucket_type: str = "policies") -> List[Dict]:
        """List all files in the specified HR storage bucket"""
        bucket_name = self.buckets.get(bucket_type, bucket_type)
        try:
            result = self.supabase.storage.from_(bucket_name).list()
            logger.info(f"Found {len(result)} files in bucket '{bucket_name}'")
            return result
        except Exception as e:
            logger.error(f"Error listing files from bucket '{bucket_name}': {e}")
            return []
    
    def list_all_storage_files(self) -> Dict[str, List[Dict]]:
        """List files from all HR storage buckets"""
        all_files = {}
        for bucket_type, bucket_name in self.buckets.items():
            files = self.list_storage_files(bucket_type)
            all_files[bucket_type] = files
            logger.info(f"Bucket '{bucket_name}' ({bucket_type}): {len(files)} files")
        return all_files
    
    def download_file_content(self, file_path: str, bucket_type: str = "policies") -> bytes:
        """Download file content from Supabase storage"""
        bucket_name = self.buckets.get(bucket_type, bucket_type)
        try:
            result = self.supabase.storage.from_(bucket_name).download(file_path)
            logger.info(f"Downloaded file: {file_path} from {bucket_name}")
            return result
        except Exception as e:
            logger.error(f"Error downloading file '{file_path}' from {bucket_name}: {e}")
            return None
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks using Gemini"""
        try:
            embeddings = []
            for text in texts:
                # Use Gemini's embedding model
                embedding_result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(embedding_result['embedding'])
            
            logger.info(f"Generated embeddings for {len(texts)} texts using Gemini")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with Gemini: {e}")
            return []
    
    def create_article(self, title: str, content: str, file_path: str, bucket_type: str, tags: List[str] = None) -> Optional[str]:
        """Create a new knowledge base article"""
        try:
            bucket_name = self.buckets.get(bucket_type, bucket_type)
            article_data = {
                "title": title,
                "body": content,  # Using 'body' as per your schema
                "source_url": f"{bucket_name}/{file_path}",  # Store bucket + path
                "tags": tags or [],
                "is_published": True
            }
            
            result = self.supabase.table("kb_article").insert(article_data).execute()
            
            if result.data:
                article_id = result.data[0]["id"]
                logger.info(f"Created article: {title} (ID: {article_id})")
                return article_id
            else:
                logger.error(f"Failed to create article: {title}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating article '{title}': {e}")
            return None
    
    def create_chunks_with_embeddings(self, article_id: str, content: str) -> bool:
        """Create chunks and embeddings for an article"""
        try:
            # Import chunking utilities
            from document_processor import chunk_text
            
            # Create chunks
            chunks = chunk_text(content, self.chunk_size, self.chunk_overlap)
            
            # Process in batches
            for i in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[i:i + self.batch_size]
                batch_texts = [chunk['content'] for chunk in batch_chunks]
                
                # Generate embeddings for this batch
                embeddings = self.generate_embeddings(batch_texts)
                
                if not embeddings:
                    logger.error(f"Failed to generate embeddings for batch {i}")
                    continue
                
                # Prepare chunk data for database
                chunk_data = []
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_data.append({
                        "article_id": article_id,
                        "chunk_index": i + j,
                        "content": chunk['content'],
                        "embedding": embedding
                    })
                
                # Insert chunks into database
                result = self.supabase.table("kb_chunk").insert(chunk_data).execute()
                
                if result.data:
                    logger.info(f"Inserted batch {i//self.batch_size + 1}: {len(chunk_data)} chunks")
                else:
                    logger.error(f"Failed to insert chunk batch {i}")
                    return False
            
            logger.info(f"Successfully processed {len(chunks)} chunks for article {article_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing chunks for article {article_id}: {e}")
            return False
    
    def search_similar_chunks(self, query: str, limit: int = 5, threshold: float = 0.7) -> List[Dict]:
        """Search for similar chunks using vector similarity"""
        try:
            # Generate query embedding
            query_embeddings = self.generate_embeddings([query])
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            
            # Call the RPC function from your schema
            result = self.supabase.rpc('match_kb_chunks', {
                'query_embedding': query_embedding,
                'match_threshold': threshold,
                'match_count': limit
            }).execute()
            
            logger.info(f"Found {len(result.data)} similar chunks for query: {query[:50]}...")
            return result.data
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def get_article_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        try:
            # Count articles
            articles_result = self.supabase.table("kb_article").select("id", count="exact").execute()
            articles_count = articles_result.count
            
            # Count chunks
            chunks_result = self.supabase.table("kb_chunk").select("id", count="exact").execute()
            chunks_count = chunks_result.count
            
            stats = {
                "total_articles": articles_count,
                "total_chunks": chunks_count,
                "avg_chunks_per_article": chunks_count / articles_count if articles_count > 0 else 0
            }
            
            logger.info(f"Knowledge base stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}