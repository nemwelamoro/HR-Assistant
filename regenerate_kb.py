# -*- coding: utf-8 -*-
"""
Script to regenerate knowledge base with improved quality
"""
import logging
from knowledge_base import HRKnowledgeBaseClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Regenerate knowledge base"""
    try:
        kb = HRKnowledgeBaseClient()
        
        print("🔄 Regenerating Knowledge Base...")
        print("=" * 50)
        
        # Clear existing data
        print("1. Clearing existing articles and chunks...")
        kb.supabase.table('kb_chunk').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        kb.supabase.table('kb_article').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        
        print("2. Knowledge base cleared!")
        print("3. Run process_hr_documents.py to regenerate with improved quality")
        
    except Exception as e:
        logger.error(f"Error regenerating knowledge base: {e}")

if __name__ == "__main__":
    main()