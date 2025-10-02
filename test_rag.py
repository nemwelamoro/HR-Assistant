# -*- coding: utf-8 -*-
"""
Test script for the HR RAG system
"""
import os
from rag_engine import HRRAGEngine

def test_rag_system():
    """Test the RAG system with various queries"""
    
    # Initialize RAG engine
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        return
    
    rag = HRRAGEngine(api_key=gemini_api_key)
    
    # Test queries
    test_queries = [
        "What is the company's leave policy?",
        "How do I request time off?",
        "What are the performance review procedures?",
        "What benefits do employees get?",
        "What is the dress code policy?",
        "How do I return company equipment?",
        "What happens during the recruitment process?",
        "What should I do if I'm leaving the company?",
        "How long is the probationary period?",
        "What is completely unrelated to HR policies?"  # Test edge case
    ]
    
    print("🤖 Testing HR RAG System")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 50)
        
        response = rag.ask(query, include_sources=True)
        
        print(f"Confidence: {response['confidence_level']}")
        print(f"Score: {response['confidence_score']:.2f}")
        print(f"Sources: {response['sources_used']}")
        print(f"Coverage: {response['coverage']}")
        print()
        print("Answer:")
        print(response['answer'])
        
        if response.get('chunks'):
            print(f"\nSources Used ({len(response['chunks'])}):")
            for chunk in response['chunks']:
                print(f"  • {chunk.get('article_title', 'Unknown')} (Similarity: {chunk.get('similarity', 0):.2f})")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    test_rag_system()