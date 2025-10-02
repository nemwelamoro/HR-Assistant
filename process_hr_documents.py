#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Knowledge Base Processor
Processes all HR documents from Supabase storage and creates chunks with embeddings
"""

import os
import logging
from typing import List, Dict
from knowledge_base import HRKnowledgeBaseClient
from document_processor import process_document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_all_hr_documents():
    """Main function to process all HR documents from Supabase storage"""
    
    # Initialize the knowledge base client
    kb_client = HRKnowledgeBaseClient()
    
    # Get list of all files from all buckets
    logger.info("Fetching files from all Supabase storage buckets...")
    all_files = kb_client.list_all_storage_files()
    
    total_files = sum(len(files) for files in all_files.values())
    if total_files == 0:
        logger.warning("No files found in any storage bucket")
        return
    
    # Statistics tracking
    stats = {
        'processed': 0,
        'failed': 0,
        'total_chunks': 0,
        'skipped': 0,
        'by_bucket': {}
    }
    
    # Process files from each bucket
    for bucket_type, files in all_files.items():
        logger.info(f"Processing bucket: {bucket_type} ({len(files)} files)")
        stats['by_bucket'][bucket_type] = {'processed': 0, 'failed': 0, 'skipped': 0}
        
        for file_info in files:
            filename = file_info['name']
            file_path = file_info['name']
            
            logger.info(f"Processing file: {filename} from {bucket_type}")
            
            try:
                # Skip directories and non-document files
                if file_info.get('metadata', {}).get('mimetype') == 'application/x-directory':
                    logger.info(f"Skipping directory: {filename}")
                    stats['skipped'] += 1
                    stats['by_bucket'][bucket_type]['skipped'] += 1
                    continue
                
                # Check if article already exists
                existing_article = check_existing_article(kb_client, f"{kb_client.buckets[bucket_type]}/{filename}")
                if existing_article:
                    logger.info(f"Article already exists for {filename}, skipping...")
                    stats['skipped'] += 1
                    stats['by_bucket'][bucket_type]['skipped'] += 1
                    continue
                
                # Download file content
                file_content = kb_client.download_file_content(file_path, bucket_type)
                if not file_content:
                    logger.error(f"Failed to download {filename}")
                    stats['failed'] += 1
                    stats['by_bucket'][bucket_type]['failed'] += 1
                    continue
            
                # Process document
                doc_info = process_document(filename, file_content)
                
                if not doc_info['content']:
                    logger.warning(f"No text content extracted from {filename}")
                    stats['failed'] += 1
                    stats['by_bucket'][bucket_type]['failed'] += 1
                    continue
                
                # Check if file should be skipped (letterheads, minimal content, etc.)
                from document_processor import should_skip_file
                if should_skip_file(filename, doc_info['content']):
                    logger.info(f"Skipping {filename} - minimal meaningful content")
                    stats['skipped'] += 1
                    stats['by_bucket'][bucket_type]['skipped'] += 1
                    continue
                
                logger.info(f"Extracted {doc_info['word_count']} words from {filename}")
                
                # Add bucket type to tags
                doc_info['tags'].append(f"bucket-{bucket_type}")
                
                # Create article in knowledge base
                article_id = kb_client.create_article(
                    title=doc_info['title'],
                    content=doc_info['content'],
                    file_path=file_path,
                    bucket_type=bucket_type,
                    tags=doc_info['tags']
                )
                
                if not article_id:
                    logger.error(f"Failed to create article for {filename}")
                    stats['failed'] += 1
                    stats['by_bucket'][bucket_type]['failed'] += 1
                    continue
                
                # Create chunks and embeddings
                success = kb_client.create_chunks_with_embeddings(
                    article_id=article_id,
                    content=doc_info['content']
                )
                
                if success:
                    logger.info(f"✅ Successfully processed {filename}")
                    stats['processed'] += 1
                    stats['by_bucket'][bucket_type]['processed'] += 1
                    
                    # Estimate chunk count (rough calculation)
                    estimated_chunks = len(doc_info['content']) // 2000  # ~500 tokens per chunk
                    stats['total_chunks'] += estimated_chunks
                else:
                    logger.error(f"Failed to create chunks for {filename}")
                    stats['failed'] += 1
                    stats['by_bucket'][bucket_type]['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                stats['failed'] += 1
                stats['by_bucket'][bucket_type]['failed'] += 1
    
    # Print final statistics
    print_processing_stats(stats, kb_client)

def check_existing_article(kb_client: HRKnowledgeBaseClient, source_url: str) -> bool:
    """Check if an article already exists for this source URL"""
    try:
        result = kb_client.supabase.table("kb_article").select("id").eq("source_url", source_url).execute()
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"Error checking existing article for {source_url}: {e}")
        return False

def print_processing_stats(stats: Dict, kb_client: HRKnowledgeBaseClient):
    """Print processing statistics"""
    
    print("\n" + "="*60)
    print("📊 PROCESSING COMPLETE")
    print("="*60)
    print(f"✅ Successfully processed: {stats['processed']} files")
    print(f"❌ Failed to process: {stats['failed']} files") 
    print(f"⏭️  Skipped: {stats['skipped']} files")
    print(f"📄 Estimated total chunks: {stats['total_chunks']}")
    
    # Print stats by bucket
    print(f"\n📂 Results by Storage Bucket:")
    for bucket_type, bucket_stats in stats['by_bucket'].items():
        print(f"   {bucket_type.upper()}:")
        print(f"      ✅ Processed: {bucket_stats['processed']}")
        print(f"      ❌ Failed: {bucket_stats['failed']}")
        print(f"      ⏭️  Skipped: {bucket_stats['skipped']}")
    
    # Get current knowledge base stats
    kb_stats = kb_client.get_article_stats()
    if kb_stats:
        print(f"\n📚 Knowledge Base Stats:")
        print(f"   Total articles: {kb_stats['total_articles']}")
        print(f"   Total chunks: {kb_stats['total_chunks']}")
        print(f"   Avg chunks per article: {kb_stats['avg_chunks_per_article']:.1f}")
    
    print("="*60)

def test_search_functionality(kb_client: HRKnowledgeBaseClient):
    """Test the search functionality with sample queries"""
    
    print("\n🔍 Testing Search Functionality")
    print("-"*30)
    
    test_queries = [
        "What is the company leave policy?",
        "How do I request time off?",
        "What are the performance review guidelines?", 
        "Employee benefits information",
        "Recruitment and hiring process"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = kb_client.search_similar_chunks(query, limit=3)
        
        if results:
            for i, result in enumerate(results, 1):
                similarity = result.get('similarity', 0)
                title = result.get('article_title', 'Unknown')
                content = result.get('content', '')[:100] + "..."
                
                print(f"  {i}. [{similarity:.2f}] {title}")
                print(f"     {content}")
        else:
            print("  No results found")

if __name__ == "__main__":
    print("🚀 Starting HR Knowledge Base Processing")
    print("="*50)
    
    try:
        # Process all documents
        process_all_hr_documents()
        
        # Test search functionality
        kb_client = HRKnowledgeBaseClient()
        test_search_functionality(kb_client)
        
        print("\n✅ All processing complete!")
        
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        logger.error(f"Main processing error: {e}")