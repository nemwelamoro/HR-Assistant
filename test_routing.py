from query_router import HRQueryRouter
import os

def test_improved_routing():
    """Test the improved query routing"""
    
    router = HRQueryRouter(gemini_api_key=os.getenv("GEMINI_API_KEY"))
    
    test_queries = [
        # Should be DATA queries
        ("Show me current headcount breakdown", "data_query", "headcount"),
        ("What is our attrition rate?", "data_query", "attrition"),
        ("Appraisal completion status", "data_query", "appraisals"),
        ("Give me HR dashboard summary", "data_query", "dashboard"),
        
        # Should be DOCUMENT queries (RAG)
        ("What are the performance review procedures?", "document_query", "policy"),
        ("How do I request time off?", "document_query", "policy"),
        ("What is the leave policy?", "document_query", "policy"),
        ("Performance review guidelines", "document_query", "policy"),
    ]
    
    print("🧪 Testing Improved Query Routing")
    print("=" * 60)
    
    for query, expected_type, expected_data_type in test_queries:
        query_type, data_type, metadata = router.analyze_query_intent(query)
        
        # Check if routing is correct
        routing_correct = (query_type == expected_type)
        type_icon = "✅" if routing_correct else "❌"
        
        print(f"\n{type_icon} Query: '{query}'")
        print(f"   Expected: {expected_type} -> {expected_data_type}")
        print(f"   Got: {query_type} -> {data_type}")
        print(f"   Reason: {metadata.get('reason', 'unknown')}")
        print(f"   Pattern: {metadata.get('matched_pattern', 'none')}")

if __name__ == "__main__":
    test_improved_routing()