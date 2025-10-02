# -*- coding: utf-8 -*-
"""
Advanced RAG (Retrieval-Augmented Generation) Engine for HR Knowledge Base
"""
import logging
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime
import google.generativeai as genai
from knowledge_base import HRKnowledgeBaseClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRRAGEngine:
    """Advanced RAG Engine with quality response generation"""
    
    def __init__(self, api_key: str = None):
        """Initialize RAG engine"""
        self.kb_client = HRKnowledgeBaseClient()
        
        # Initialize Gemini for generation
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Response quality thresholds
        self.min_similarity_score = 0.65  # Minimum relevance for chunks
        self.max_chunks = 5  # Maximum chunks to use for context
        self.min_chunks_for_confident_answer = 2  # Need at least 2 relevant chunks
        
        logger.info("HR RAG Engine initialized successfully")
    
    def preprocess_query(self, query: str) -> str:
        """Clean and enhance the user query"""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Add HR context if not present
        hr_keywords = ['policy', 'leave', 'benefits', 'recruitment', 'performance', 'hr', 'employee']
        if not any(keyword in query.lower() for keyword in hr_keywords):
            query = f"HR {query}"
        
        return query
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve and filter relevant chunks"""
        try:
            # Search for similar chunks
            chunks = self.kb_client.search_similar_chunks(
                query=query,
                limit=top_k
            )
            
            if not chunks:
                return []
            
            # Filter by similarity score
            relevant_chunks = [
                chunk for chunk in chunks 
                if chunk.get('similarity', 0) >= self.min_similarity_score
            ]
            
            # Sort by similarity (highest first)
            relevant_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Limit to max_chunks
            return relevant_chunks[:self.max_chunks]
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def analyze_context_quality(self, chunks: List[Dict], query: str) -> Dict:
        """Analyze the quality of retrieved context"""
        if not chunks:
            return {
                'confidence': 0.0,
                'coverage': 'none',
                'source_diversity': 0,
                'recommendation': 'insufficient_context'
            }
        
        # Calculate average similarity
        avg_similarity = sum(chunk.get('similarity', 0) for chunk in chunks) / len(chunks)
        
        # Check source diversity (different articles)
        unique_articles = len(set(chunk.get('article_title', '') for chunk in chunks))
        
        # Determine coverage based on chunk content
        total_words = sum(len(chunk.get('content', '').split()) for chunk in chunks)
        
        # Calculate confidence score
        confidence = min(1.0, (
            avg_similarity * 0.4 +
            min(len(chunks) / self.min_chunks_for_confident_answer, 1.0) * 0.3 +
            min(unique_articles / 2, 1.0) * 0.2 +
            min(total_words / 500, 1.0) * 0.1
        ))
        
        # Determine coverage level
        if total_words > 800:
            coverage = 'comprehensive'
        elif total_words > 300:
            coverage = 'adequate'
        else:
            coverage = 'limited'
        
        # Make recommendation
        if confidence > 0.8:
            recommendation = 'high_confidence'
        elif confidence > 0.6:
            recommendation = 'moderate_confidence'
        elif confidence > 0.4:
            recommendation = 'low_confidence'
        else:
            recommendation = 'insufficient_context'
        
        return {
            'confidence': confidence,
            'coverage': coverage,
            'source_diversity': unique_articles,
            'recommendation': recommendation,
            'avg_similarity': avg_similarity,
            'total_words': total_words
        }
    
    def build_context_prompt(self, query: str, chunks: List[Dict], quality_analysis: Dict) -> str:
        """Build a high-quality context prompt for generation"""
        
        context_sections = []
        
        # Group chunks by article for better organization
        articles = {}
        for chunk in chunks:
            article_title = chunk.get('article_title', 'Unknown Document')
            if article_title not in articles:
                articles[article_title] = []
            articles[article_title].append(chunk)
        
        # Build context from articles
        for article_title, article_chunks in articles.items():
            context_sections.append(f"\n## {article_title}")
            for chunk in article_chunks:
                content = chunk.get('content', '').strip()
                similarity = chunk.get('similarity', 0)
                context_sections.append(f"[Relevance: {similarity:.2f}] {content}")
        
        context = "\n".join(context_sections)
        
        # Build the complete prompt
        prompt = f"""You are an expert HR assistant with access to company policies and procedures. 
Your role is to provide accurate, helpful, and professional responses to HR-related questions.

CONTEXT QUALITY ANALYSIS:
- Confidence Level: {quality_analysis['confidence']:.2f} ({quality_analysis['recommendation']})
- Information Coverage: {quality_analysis['coverage']}
- Sources Available: {quality_analysis['source_diversity']} different documents
- Average Relevance: {quality_analysis['avg_similarity']:.2f}

RELEVANT CONTEXT FROM COMPANY DOCUMENTS:
{context}

USER QUESTION: {query}

RESPONSE GUIDELINES:
1. Base your answer primarily on the provided context
2. If the context is comprehensive (confidence > 0.8), provide a detailed answer
3. If the context is limited (confidence < 0.6), acknowledge limitations and provide what you can
4. Always cite which document(s) your information comes from
5. If you cannot find relevant information in the context, clearly state this
6. Provide actionable advice when possible
7. Use a professional, helpful tone
8. Structure your response clearly with headings if covering multiple topics

RESPONSE:"""
        
        return prompt
    
    def generate_response(self, prompt: str, quality_analysis: Dict) -> str:
        """Generate a high-quality response using Gemini"""
        try:
            # Adjust generation parameters based on context quality
            if quality_analysis['confidence'] > 0.8:
                temperature = 0.3  # More deterministic for high-confidence responses
            elif quality_analysis['confidence'] > 0.6:
                temperature = 0.5  # Balanced
            else:
                temperature = 0.7  # More creative for low-confidence responses
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1000,
                    top_p=0.95,
                    top_k=40
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again or contact support."
    
    def post_process_response(self, response: str, quality_analysis: Dict) -> Dict:
        """Post-process and enhance the response"""
        
        # Add confidence indicator
        confidence_level = quality_analysis['recommendation']
        confidence_labels = {
            'high_confidence': '✅ High Confidence',
            'moderate_confidence': '⚠️ Moderate Confidence',
            'low_confidence': '❓ Low Confidence',
            'insufficient_context': '❌ Limited Information'
        }
        
        # Build final response structure
        final_response = {
            'answer': response,
            'confidence_level': confidence_labels.get(confidence_level, '❓ Unknown'),
            'confidence_score': quality_analysis['confidence'],
            'sources_used': quality_analysis['source_diversity'],
            'coverage': quality_analysis['coverage'],
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'avg_similarity': quality_analysis.get('avg_similarity', 0),
                'total_context_words': quality_analysis.get('total_words', 0),
                'recommendation': confidence_level
            }
        }
        
        return final_response
    
    def ask(self, question: str, include_sources: bool = True) -> Dict:
        """Main RAG pipeline - ask a question and get a comprehensive answer"""
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Step 1: Preprocess query
            processed_query = self.preprocess_query(question)
            
            # Step 2: Retrieve relevant chunks
            chunks = self.retrieve_relevant_chunks(processed_query)
            
            # Step 3: Analyze context quality
            quality_analysis = self.analyze_context_quality(chunks, processed_query)
            
            # Step 4: Handle insufficient context
            if quality_analysis['recommendation'] == 'insufficient_context':
                return {
                    'answer': "I don't have enough relevant information in the knowledge base to answer your question confidently. Please try rephrasing your question or contact HR directly for assistance.",
                    'confidence_level': '❌ Limited Information',
                    'confidence_score': 0.0,
                    'sources_used': 0,
                    'coverage': 'none',
                    'timestamp': datetime.now().isoformat(),
                    'chunks': chunks if include_sources else None
                }
            
            # Step 5: Build context prompt
            prompt = self.build_context_prompt(processed_query, chunks, quality_analysis)
            
            # Step 6: Generate response
            raw_response = self.generate_response(prompt, quality_analysis)
            
            # Step 7: Post-process response
            final_response = self.post_process_response(raw_response, quality_analysis)
            
            # Step 8: Add source chunks if requested
            if include_sources:
                final_response['chunks'] = chunks
            
            logger.info(f"Generated response with confidence: {final_response['confidence_score']:.2f}")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}. Please try again.",
                'confidence_level': '❌ Error',
                'confidence_score': 0.0,
                'sources_used': 0,
                'coverage': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }