# -*- coding: utf-8 -*-
"""
Advanced RAG (Retrieval-Augmented Generation) Engine for HR Knowledge Base
Enhanced with LLM-based NLP capabilities
"""
import logging
from typing import List, Dict, Optional, Tuple
import re
import json
from datetime import datetime
import google.generativeai as genai
from knowledge_base import HRKnowledgeBaseClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRRAGEngine:
    """Advanced RAG Engine with LLM-based NLP and conversational capabilities"""
    
    def __init__(self, api_key: str = None):
        """Initialize RAG engine"""
        self.kb_client = HRKnowledgeBaseClient()
        
        # Initialize Gemini for generation and NLP
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Response quality thresholds - More lenient for better UX
        self.min_similarity_score = 0.4  # Lowered from 0.65
        self.max_chunks = 5
        self.min_chunks_for_confident_answer = 1  # Lowered from 2
        
        # HR domain topics for fallback responses
        self.hr_topics = [
            'benefits', 'leave', 'policies', 'performance', 'recruitment', 
            'compensation', 'training', 'workplace', 'onboarding', 'termination'
        ]
        
        logger.info("HR RAG Engine initialized successfully")
    
    def analyze_query_with_llm(self, query: str) -> Dict:
        """Use LLM to analyze query intent, topics, and context"""
        
        # Simpler, safer prompt to avoid safety filters
        analysis_prompt = f"""Analyze this HR query and extract key information in JSON format:

Query: {query}

Provide analysis as JSON:
{{
    "primary_topic": "benefits|leave|policies|performance|recruitment|compensation|training|general",
    "key_terms": ["important", "terms", "from", "query"],
    "search_keywords": ["alternative", "search", "terms"],
    "intent": "informational|procedural|support"
}}"""

        try:
            response = self.model.generate_content(
                analysis_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Lower temperature for more consistent output
                    max_output_tokens=300
                ),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            )
            
            if not response.text:
                logger.warning("Empty response from LLM, using fallback analysis")
                return self._create_fallback_analysis(query)
                
            # Clean and parse JSON
            analysis_text = response.text.strip()
            if analysis_text.startswith('```json'):
                analysis_text = analysis_text.replace('```json', '').replace('```', '').strip()
            
            analysis = json.loads(analysis_text)
            logger.info(f"Query analysis successful: Topic={analysis.get('primary_topic')}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in LLM query analysis: {e}")
            return self._create_fallback_analysis(query)
    
    def _create_fallback_analysis(self, query: str) -> Dict:
        """Create fallback analysis when LLM fails"""
        query_lower = query.lower()
        
        # Simple keyword matching for topic detection
        topic_keywords = {
            'compensation': ['salary', 'pay', 'compensation', 'wage', 'bonus', 'raise', 'review'],
            'benefits': ['benefits', 'insurance', 'health', 'dental', 'retirement', '401k'],
            'leave': ['leave', 'vacation', 'pto', 'sick', 'time off', 'absence'],
            'performance': ['performance', 'review', 'evaluation', 'appraisal'],
            'policies': ['policy', 'policies', 'procedure', 'guidelines', 'rules']
        }
        
        detected_topic = 'general'
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_topic = topic
                break
        
        # Generate search keywords
        search_keywords = [query]
        if detected_topic != 'general':
            search_keywords.append(detected_topic)
        
        return {
            "primary_topic": detected_topic,
            "key_terms": query.split()[:5],
            "search_keywords": search_keywords,
            "intent": "informational"
        }
    
    def enhance_query_for_search(self, query: str, analysis: Dict) -> List[str]:
        """Generate multiple search variations based on LLM analysis"""
        
        search_queries = [query]  # Original query
        
        # Add suggested search keywords
        if analysis.get('search_keywords'):
            search_queries.extend(analysis['search_keywords'])
        
        # Add topic-based search
        if analysis.get('primary_topic') and analysis['primary_topic'] != 'general':
            search_queries.append(analysis['primary_topic'])
        
        # Add entity-based searches
        if analysis.get('key_entities'):
            entities_query = ' '.join(analysis['key_entities'][:3])
            search_queries.append(entities_query)
        
        return search_queries[:4]  # Limit to 4 variations
    
    def preprocess_query(self, query: str) -> str:
        """Clean and enhance the user query"""
        query = re.sub(r'\s+', ' ', query.strip())
        return query
    
    def retrieve_relevant_chunks(self, query: str, analysis: Dict = None, top_k: int = 10) -> List[Dict]:
        """Enhanced retrieval with multiple search strategies and lower thresholds - OPTIMIZED"""
        try:
            all_chunks = []
            
            # Generate comprehensive search variations (now optimized)
            search_queries = self._generate_search_variations(query, analysis)
            logger.info(f"Search variations: {search_queries}")
            
            # Progressive search with decreasing thresholds
            thresholds = [0.5, 0.3, 0.2]  # Much lower thresholds
            
            for threshold in thresholds:
                if len(all_chunks) >= top_k:
                    break
                    
                for search_query in search_queries:
                    try:
                        chunks = self.kb_client.search_similar_chunks(
                            query=search_query,
                            limit=min(top_k, 5),  # Limit per search to avoid noise
                            threshold=threshold
                        )
                        if chunks:
                            all_chunks.extend(chunks)
                            logger.info(f"Found {len(chunks)} chunks for '{search_query}' at threshold {threshold}")
                            
                            # Early stopping if we have enough good results
                            if len(all_chunks) >= top_k and threshold >= 0.3:
                                break
                                
                    except Exception as e:
                        logger.warning(f"Search failed for query '{search_query}': {e}")
                        continue
                
                if all_chunks:
                    break  # Stop if we found results
            
            if not all_chunks:
                logger.warning(f"No chunks found for any search variation of: {query}")
                return []
            
            # Remove duplicates and sort
            unique_chunks = self._deduplicate_chunks(all_chunks)
            
            # Sort by similarity but keep more results
            unique_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Return top results
            final_chunks = unique_chunks[:top_k]
            logger.info(f"Returning {len(final_chunks)} unique chunks")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def _generate_search_variations(self, query: str, analysis: Dict) -> List[str]:
        """Generate more comprehensive search variations - FIXED"""
        variations = [query]  # Always keep the original full query first
        
        if analysis:
            # Add topic-based searches
            topic = analysis.get('primary_topic', '')
            if topic and topic != 'general':
                variations.append(topic)
                variations.append(f"{topic} policy")
                variations.append(f"employee {topic}")
            
            # Add key terms but filter out stop words
            key_terms = analysis.get('key_terms', [])
            if key_terms:
                # Filter out stop words and short terms
                stop_words = {'how', 'do', 'we', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
                meaningful_terms = [term for term in key_terms if len(term) > 2 and term.lower() not in stop_words]
                
                if meaningful_terms:
                    variations.extend(meaningful_terms[:2])  # Only add 2 meaningful terms
                    variations.append(' '.join(meaningful_terms[:3]))
    
        # Add query word variations - but filter out stop words
        query_words = query.lower().split()
        meaningful_words = [word for word in query_words if len(word) > 2 and word not in {'how', 'do', 'we', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}]
        
        if len(meaningful_words) > 2:
            # Try different combinations of meaningful words only
            variations.append(' '.join(meaningful_words[:3]))
            if len(meaningful_words) > 3:
                variations.append(' '.join(meaningful_words[-3:]))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var.lower() not in seen and len(var.strip()) > 2:  # Skip very short variations
                seen.add(var.lower())
                unique_variations.append(var)
        
        # Limit to 4 most relevant variations
        return unique_variations[:4]
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks based on content similarity"""
        if not chunks:
            return []
        
        unique_chunks = []
        seen_content = set()
        
        for chunk in chunks:
            content = chunk.get('content', '')
            # Use first 200 characters as fingerprint
            content_fingerprint = content[:200].lower().strip()
            
            if content_fingerprint not in seen_content:
                seen_content.add(content_fingerprint)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def generate_conversational_fallback(self, query: str, analysis: Dict) -> Dict:
        """Generate helpful conversational response when no relevant chunks found"""
        
        fallback_prompt = f"""The user asked an HR question but no relevant information was found in the knowledge base.

User Question: "{query}"
Query Analysis: {json.dumps(analysis, indent=2)}

Generate a helpful, conversational response that:
1. Acknowledges their question empathetically
2. Explains that you don't have specific information about their exact question
3. Suggests ways they can get help (rephrasing, being more specific, contacting HR)
4. If appropriate, offers related information you might be able to help with
5. Asks clarifying questions to better understand their needs
6. Maintains a professional but friendly tone

Make it feel like talking to a helpful HR colleague, not a robotic system."""

        try:
            response = self.model.generate_content(
                fallback_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500
                )
            )
            
            return {
                'answer': response.text,
                'confidence_level': '🤝 Ready to Help',
                'confidence_score': 0.3,
                'sources_used': 0,
                'coverage': 'conversational_fallback',
                'timestamp': datetime.now().isoformat(),
                'query_analysis': analysis,
                'response_type': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return {
                'answer': f"I understand you're asking about {analysis.get('primary_topic', 'HR matters')}, but I don't have specific information about your question in my current knowledge base. Could you try rephrasing your question or providing more details? I'm here to help!",
                'confidence_level': '🤝 Ready to Help',
                'confidence_score': 0.2,
                'sources_used': 0,
                'coverage': 'basic_fallback',
                'timestamp': datetime.now().isoformat(),
                'response_type': 'fallback'
            }
    
    def enhance_low_confidence_response(self, response: str, analysis: Dict) -> str:
        """Use LLM to enhance responses when confidence is low"""
        
        enhancement_prompt = f"""Enhance this HR response to be more conversational and helpful when the AI has low confidence:

Original Response: {response}

User's Query Analysis: {json.dumps(analysis, indent=2)}

Please enhance the response by:
1. Adding a conversational opening that acknowledges uncertainty transparently
2. Keeping the original helpful information
3. Adding suggestions for follow-up questions or clarifications
4. Making it sound more human and empathetic
5. Offering alternative ways to get help

Keep the enhanced response concise but more engaging."""

        try:
            enhanced = self.model.generate_content(
                enhancement_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.6,
                    max_output_tokens=600
                )
            )
            return enhanced.text
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return response  # Return original if enhancement fails
    
    def analyze_context_quality(self, chunks: List[Dict], query: str) -> Dict:
        """Analyze the quality of retrieved context"""
        if not chunks:
            return {
                'confidence': 0.0,
                'coverage': 'none',
                'source_diversity': 0,
                'recommendation': 'insufficient_context'
            }
        
        avg_similarity = sum(chunk.get('similarity', 0) for chunk in chunks) / len(chunks)
        unique_articles = len(set(chunk.get('article_title', '') for chunk in chunks))
        total_words = sum(len(chunk.get('content', '').split()) for chunk in chunks)
        
        # More lenient confidence calculation
        confidence = min(1.0, (
            avg_similarity * 0.5 +
            min(len(chunks) / self.min_chunks_for_confident_answer, 1.0) * 0.3 +
            min(unique_articles / 1, 1.0) * 0.1 +  # Reduced requirement
            min(total_words / 300, 1.0) * 0.1  # Reduced requirement
        ))
        
        if total_words > 600:
            coverage = 'comprehensive'
        elif total_words > 200:
            coverage = 'adequate'
        else:
            coverage = 'limited'
        
        if confidence > 0.7:  # Lowered thresholds
            recommendation = 'high_confidence'
        elif confidence > 0.5:
            recommendation = 'moderate_confidence'
        elif confidence > 0.3:
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
    
    def build_context_prompt(self, query: str, chunks: List[Dict], quality_analysis: Dict, query_analysis: Dict) -> str:
        """Build enhanced context prompt with conversational instructions - OPTIMIZED"""
        
        # Build more concise context
        context_sections = []
        articles = {}
        for chunk in chunks:
            article_title = chunk.get('article_title', 'Unknown Document')
            if article_title not in articles:
                articles[article_title] = []
            articles[article_title].append(chunk)
        
        # Limit context to most relevant chunks to save tokens
        total_chunks_added = 0
        max_chunks = 8  # Reduced from unlimited
        
        for article_title, article_chunks in articles.items():
            if total_chunks_added >= max_chunks:
                break
                
            context_sections.append(f"\n## {article_title}")
            
            # Sort chunks by similarity and take the best ones
            article_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            for chunk in article_chunks[:3]:  # Max 3 chunks per article
                if total_chunks_added >= max_chunks:
                    break
                    
                content = chunk.get('content', '').strip()
                similarity = chunk.get('similarity', 0)
                
                # Truncate very long chunks to save tokens
                if len(content) > 800:
                    content = content[:800] + "..."
                
                context_sections.append(f"[Relevance: {similarity:.2f}] {content}")
                total_chunks_added += 1
        
        context = "\n".join(context_sections)
        
        # More concise prompt
        prompt = f"""You are a helpful HR assistant. Provide comprehensive, accurate answers based on company policies.

QUERY ANALYSIS:
- Topic: {query_analysis.get('primary_topic', 'general')}
- Intent: {query_analysis.get('intent', 'informational')}

CONTEXT QUALITY: {quality_analysis['confidence']:.2f} confidence, {quality_analysis['coverage']} coverage

COMPANY INFORMATION:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Provide a complete, detailed answer using the context
2. Be conversational and helpful
3. Cite sources naturally ("According to our HR Policy...")
4. Include specific steps or procedures when available
5. If information seems incomplete, acknowledge it
6. End with an offer to help further

COMPLETE RESPONSE:"""
        
        return prompt
    
    def generate_response(self, prompt: str, quality_analysis: Dict) -> str:
        """Generate response using Gemini with completion checking"""
        try:
            if quality_analysis['confidence'] > 0.7:
                temperature = 0.4
            elif quality_analysis['confidence'] > 0.5:
                temperature = 0.6
            else:
                temperature = 0.7
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1700,  # Increased
                    top_p=0.95,
                    top_k=40
                )
            )
            
            response_text = response.text
            
            # Check if response seems incomplete (ends mid-sentence)
            if response_text and not response_text.strip().endswith(('.', '!', '?', ':')):
                logger.warning("Response appears incomplete, attempting to complete...")
                # Add a completion note
                response_text += "\n\n*If you need more specific details about any part of this process, please let me know and I'll be happy to elaborate!*"
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again or contact HR support directly."
    
    def post_process_response(self, response: str, quality_analysis: Dict, query_analysis: Dict) -> Dict:
        """Post-process and enhance the response"""
        
        confidence_level = quality_analysis['recommendation']
        confidence_labels = {
            'high_confidence': '✅ High Confidence',
            'moderate_confidence': '⚠️ Moderate Confidence', 
            'low_confidence': '🤔 Lower Confidence - Happy to Clarify',
            'insufficient_context': '❌ Limited Information'
        }
        
        final_response = {
            'answer': response,
            'confidence_level': confidence_labels.get(confidence_level, '❓ Unknown'),
            'confidence_score': quality_analysis['confidence'],
            'sources_used': quality_analysis['source_diversity'],
            'coverage': quality_analysis['coverage'],
            'timestamp': datetime.now().isoformat(),
            'query_analysis': query_analysis,
            'metadata': {
                'avg_similarity': quality_analysis.get('avg_similarity', 0),
                'total_context_words': quality_analysis.get('total_words', 0),
                'recommendation': confidence_level,
                'primary_topic': query_analysis.get('primary_topic'),
                'user_intent': query_analysis.get('intent_type')
            }
        }
        
        return final_response
    
    def ask(self, question: str, include_sources: bool = True) -> Dict:
        """Enhanced RAG pipeline with LLM-based NLP and conversational capabilities"""
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Step 1: Analyze query with LLM
            query_analysis = self.analyze_query_with_llm(question)
            
            # Step 2: Preprocess query
            processed_query = self.preprocess_query(question)
            
            # Step 3: Enhanced retrieval with multiple strategies
            chunks = self.retrieve_relevant_chunks(processed_query, query_analysis)
            
            # Step 4: Handle no relevant chunks with conversational fallback
            if not chunks:
                logger.info("No relevant chunks found, generating conversational fallback")
                return self.generate_conversational_fallback(question, query_analysis)
            
            # Step 5: Analyze context quality
            quality_analysis = self.analyze_context_quality(chunks, processed_query)
            
            # Step 6: Build enhanced context prompt
            prompt = self.build_context_prompt(processed_query, chunks, quality_analysis, query_analysis)
            
            # Step 7: Generate response
            raw_response = self.generate_response(prompt, quality_analysis)
            
            # Step 8: Enhance low confidence responses
            if quality_analysis['confidence'] < 0.6:
                raw_response = self.enhance_low_confidence_response(raw_response, query_analysis)
            
            # Step 9: Post-process response
            final_response = self.post_process_response(raw_response, quality_analysis, query_analysis)
            
            # Step 10: Add source chunks if requested
            if include_sources:
                final_response['chunks'] = chunks
            
            logger.info(f"Generated response with confidence: {final_response['confidence_score']:.2f}")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in enhanced RAG pipeline: {e}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}. Please try again or contact HR support directly.",
                'confidence_level': '❌ Error',
                'confidence_score': 0.0,
                'sources_used': 0,
                'coverage': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }