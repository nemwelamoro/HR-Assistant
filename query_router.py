# -*- coding: utf-8 -*-
"""
Query Router for determining whether to use RAG or structured data queries
"""
import logging
import re
from typing import Dict, Tuple
from rag_engine import HRRAGEngine
from hr_analytics import HRAnalyticsService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRQueryRouter:
    """Routes queries to appropriate handlers (RAG vs Analytics)"""
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize router with both engines"""
        self.rag_engine = HRRAGEngine(api_key=gemini_api_key)
        self.analytics_service = HRAnalyticsService()
        
        # Data query patterns
        self.data_query_patterns = {
            'headcount': [
                r'how many employees', r'headcount', r'total employees', r'staff count',
                r'employee count', r'workforce size', r'team size'
            ],
            'attrition': [
                r'attrition rate', r'turnover', r'resignations', r'terminations',
                r'people leaving', r'quit rate', r'departure rate'
            ],
            'probation': [
                r'probation', r'probationary', r'new employees', r'probation period',
                r'probation review', r'probation status'
            ],
            'appraisals': [
                r'appraisal', r'performance review', r'evaluation', r'review status',
                r'appraisal completion', r'performance evaluation'
            ],
            'contracts': [
                r'contract expir', r'contract renewal', r'contract end', r'contract status'
            ]
        }
        
        logger.info("HR Query Router initialized")
    
    def analyze_query_intent(self, query: str) -> Tuple[str, str, Dict]:
        """
        Enhanced query analysis with better intent detection
        Returns: (query_type, data_type, metadata)
        """
        query_lower = query.lower()
        
        # PRIORITY 1: Check for explicit data request patterns (MUST match these exactly)
        explicit_data_patterns = {
            'headcount': [
                r'show me.*headcount', r'current headcount', r'how many employees do we have',
                r'employee count', r'workforce size', r'staff numbers', r'total employees'
            ],
            'attrition': [
                r'attrition rate', r'turnover rate', r'show me.*attrition', 
                r'what is our attrition', r'departure rate'
            ],
            'probation': [
                r'probation alerts', r'probation status', r'show me.*probation',
                r'upcoming probation', r'probation review alerts'
            ],
            'appraisals': [
                r'appraisal completion', r'appraisal status', r'show me.*appraisal',
                r'performance review completion', r'appraisal progress'
            ],
            'contracts': [
                r'contract expiry', r'expiring contracts', r'contract alerts',
                r'show me.*contract', r'contract renewal'
            ],
            'dashboard': [
                r'dashboard summary', r'hr summary', r'show me.*dashboard',
                r'hr dashboard', r'give me.*summary'
            ]
        }
        
        # Check explicit data patterns first
        for data_type, patterns in explicit_data_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return 'data_query', data_type, {
                        'matched_pattern': pattern,
                        'confidence': 'high',
                        'reason': 'explicit_data_request'
                    }
        
        # PRIORITY 2: Check for policy/procedure questions (should go to RAG)
        policy_patterns = [
            r'procedures?', r'process', r'how do i', r'how to', r'what is the policy',
            r'what are the.*procedures', r'guidelines', r'rules', r'requirements',
            r'how does.*work', r'what should i do'
        ]
        
        for pattern in policy_patterns:
            if re.search(pattern, query_lower):
                return 'document_query', 'policy', {
                    'matched_pattern': pattern,
                    'confidence': 'high',
                    'reason': 'policy_procedure_request'
                }
        
        # PRIORITY 3: Check for general informational questions
        info_patterns = [
            r'what is', r'explain', r'tell me about', r'information about',
            r'details about', r'help with'
        ]
        
        for pattern in info_patterns:
            if re.search(pattern, query_lower):
                return 'document_query', 'general', {
                    'matched_pattern': pattern,
                    'confidence': 'medium',
                    'reason': 'informational_request'
                }
        
        # PRIORITY 4: Fallback - simple keyword matching
        # Only route to data if it contains specific data keywords AND request indicators
        data_keywords = ['headcount', 'attrition', 'probation', 'contracts', 'dashboard']
        request_indicators = ['show', 'current', 'status', 'how many', 'total']
        
        has_data_keyword = any(keyword in query_lower for keyword in data_keywords)
        has_request_indicator = any(indicator in query_lower for indicator in request_indicators)
        
        if has_data_keyword and has_request_indicator:
            # Determine which data type
            for keyword in data_keywords:
                if keyword in query_lower:
                    return 'data_query', keyword, {
                        'matched_pattern': f'keyword:{keyword}',
                        'confidence': 'low',
                        'reason': 'keyword_fallback'
                    }
        
        # Default to document query (RAG)
        return 'document_query', 'general', {
            'confidence': 'medium',
            'reason': 'default_fallback'
        }
    
    def handle_data_query(self, query: str, data_type: str) -> Dict:
        """Handle structured data queries"""
        try:
            logger.info(f"Handling data query for type: {data_type}")
            
            if data_type == 'headcount':
                data = self.analytics_service.get_current_headcount()
                return self._format_headcount_response(query, data)
                
            elif data_type == 'attrition':
                data = self.analytics_service.get_attrition_data()
                return self._format_attrition_response(query, data)
                
            elif data_type == 'probation':
                data = self.analytics_service.get_probation_alerts()
                return self._format_probation_response(query, data)
                
            elif data_type == 'appraisals':
                data = self.analytics_service.get_appraisal_status()
                return self._format_appraisal_response(query, data)
                
            elif data_type == 'contracts':
                data = self.analytics_service.get_contract_expiry_alerts()
                return self._format_contract_response(query, data)
                
            elif data_type == 'general':
                # Get dashboard summary for general data queries
                data = self.analytics_service.get_hr_dashboard_summary()
                return self._format_summary_response(query, data)
            
            else:
                return {
                    'answer': f"I understand you're asking about {data_type}, but I'm not sure how to handle that specific data request yet. Could you try rephrasing your question?",
                    'confidence_level': '🤔 Unsure',
                    'confidence_score': 0.3,
                    'response_type': 'data_query_unsupported',
                    'data_type': data_type
                }
                
        except Exception as e:
            logger.error(f"Error handling data query: {e}")
            return {
                'answer': f"I encountered an error while retrieving that information: {str(e)}. Please try again or contact HR support.",
                'confidence_level': '❌ Error',
                'confidence_score': 0.0,
                'response_type': 'data_query_error',
                'error': str(e)
            }
    
    def _format_headcount_response(self, query: str, data: Dict) -> Dict:
        """Format headcount data into conversational response"""
        if 'error' in data:
            return {'answer': f"I had trouble retrieving headcount data: {data['error']}", 'confidence_level': '❌ Error', 'confidence_score': 0.0}
        
        total = data.get('total_headcount', 0)
        by_dept = data.get('by_department', {})
        
        response = f"📊 **Current Headcount Summary**\n\n"
        response += f"We currently have **{total} active employees** in the organization.\n\n"
        
        if by_dept:
            response += "**Department Breakdown:**\n"
            for dept, count in sorted(by_dept.items()):
                response += f"• {dept}: {count} employees\n"
        
        response += f"\n*Data last updated: {data.get('last_updated', 'Unknown')}*"
        
        return {
            'answer': response,
            'confidence_level': '✅ High Confidence',
            'confidence_score': 0.95,
            'response_type': 'data_query_headcount',
            'data': data
        }
    
    def _format_attrition_response(self, query: str, data: Dict) -> Dict:
        """Format attrition data into conversational response"""
        if 'error' in data:
            return {'answer': f"I had trouble retrieving attrition data: {data['error']}", 'confidence_level': '❌ Error', 'confidence_score': 0.0}
        
        rate = data.get('attrition_rate_percent', 0)
        period = data.get('period_months', 12)
        terminations = data.get('total_terminations', 0)
        
        response = f"📈 **Attrition Analysis ({period} months)**\n\n"
        response += f"• **Attrition Rate:** {rate}%\n"
        response += f"• **Total Departures:** {terminations} employees\n\n"
        
        by_dept = data.get('by_department', {})
        if by_dept:
            response += "**Departures by Department:**\n"
            for dept, count in sorted(by_dept.items()):
                response += f"• {dept}: {count} departures\n"
        
        response += f"\n*Analysis period: Last {period} months*"
        
        return {
            'answer': response,
            'confidence_level': '✅ High Confidence',
            'confidence_score': 0.95,
            'response_type': 'data_query_attrition',
            'data': data
        }
    
    def _format_probation_response(self, query: str, data: Dict) -> Dict:
        """Format probation alerts into conversational response"""
        if 'error' in data:
            return {'answer': f"I had trouble retrieving probation data: {data['error']}", 'confidence_level': '❌ Error', 'confidence_score': 0.0}
        
        upcoming = data.get('upcoming_reviews', [])
        overdue = data.get('overdue_reviews', [])
        total_alerts = data.get('total_alerts', 0)
        
        response = f"⏰ **Probation Review Status**\n\n"
        
        if total_alerts == 0:
            response += "✅ **Good news!** No immediate probation review alerts.\n\n"
        else:
            response += f"📋 **{total_alerts} probation review alerts require attention:**\n\n"
            
            if overdue:
                response += f"🚨 **Overdue Reviews ({len(overdue)}):**\n"
                for review in overdue[:5]:  # Limit to 5 for readability
                    days = abs(review.get('days_until_end', 0))
                    response += f"• {review.get('name', 'Unknown')} - {days} days overdue\n"
                if len(overdue) > 5:
                    response += f"• ... and {len(overdue) - 5} more\n"
                response += "\n"
            
            if upcoming:
                response += f"⏳ **Upcoming Reviews ({len(upcoming)}):**\n"
                for review in upcoming[:5]:  # Limit to 5 for readability
                    days = review.get('days_until_end', 0)
                    response += f"• {review.get('name', 'Unknown')} - {days} days remaining\n"
                if len(upcoming) > 5:
                    response += f"• ... and {len(upcoming) - 5} more\n"
        
        response += f"\n*Data last updated: {data.get('last_updated', 'Unknown')}*"
        
        return {
            'answer': response,
            'confidence_level': '✅ High Confidence',
            'confidence_score': 0.95,
            'response_type': 'data_query_probation',
            'data': data
        }
    
    def _format_appraisal_response(self, query: str, data: Dict) -> Dict:
        """Format appraisal status into conversational response"""
        if 'error' in data:
            return {'answer': f"I had trouble retrieving appraisal data: {data['error']}", 'confidence_level': '❌ Error', 'confidence_score': 0.0}
        
        if 'message' in data:
            return {'answer': f"📋 {data['message']}", 'confidence_level': '✅ High Confidence', 'confidence_score': 0.9}
        
        cycle_info = data.get('cycle_info', {})
        completion_stats = data.get('completion_stats', {})
        
        response = f"🎯 **Appraisal Status - {cycle_info.get('name', 'Current Cycle')}**\n\n"
        
        completion_rate = completion_stats.get('completion_rate_percent', 0)
        total_appraisals = completion_stats.get('total_appraisals', 0)
        completed = completion_stats.get('completed_appraisals', 0)
        pending = completion_stats.get('pending_appraisals', 0)
        
        response += f"📊 **Overall Progress:**\n"
        response += f"• **Completion Rate:** {completion_rate}%\n"
        response += f"• **Completed:** {completed}/{total_appraisals} appraisals\n"
        response += f"• **Pending:** {pending} appraisals\n"
        
        if cycle_info.get('is_overdue'):
            response += f"\n🚨 **Status:** Cycle is overdue (ended {cycle_info.get('end_date')})\n"
        else:
            response += f"\n📅 **Cycle End Date:** {cycle_info.get('end_date')}\n"
        
        by_dept = data.get('by_department', {})
        if by_dept:
            response += f"\n**Department Progress:**\n"
            for dept, stats in sorted(by_dept.items()):
                dept_rate = stats.get('completion_rate', 0)
                response += f"• {dept}: {dept_rate:.1f}% complete\n"
        
        return {
            'answer': response,
            'confidence_level': '✅ High Confidence',
            'confidence_score': 0.95,
            'response_type': 'data_query_appraisal',
            'data': data
        }
    
    def _format_contract_response(self, query: str, data: Dict) -> Dict:
        """Format contract expiry data into conversational response"""
        if 'error' in data:
            return {'answer': f"I had trouble retrieving contract data: {data['error']}", 'confidence_level': '❌ Error', 'confidence_score': 0.0}
        
        expiring = data.get('expiring_contracts', [])
        total_expiring = data.get('total_expiring', 0)
        alert_period = data.get('alert_period_days', 30)
        
        response = f"📋 **Contract Expiry Alerts (Next {alert_period} days)**\n\n"
        
        if total_expiring == 0:
            response += "✅ **Good news!** No contracts expiring in the next 30 days.\n\n"
        else:
            response += f"⚠️ **{total_expiring} contracts require attention:**\n\n"
            
            for contract in expiring[:10]:  # Limit to 10 for readability
                name = contract.get('name', 'Unknown')
                days = contract.get('days_until_expiry', 0)
                contract_type = contract.get('contract_type', 'Unknown')
                response += f"• **{name}** ({contract_type}) - {days} days remaining\n"
            
            if len(expiring) > 10:
                response += f"• ... and {len(expiring) - 10} more contracts\n"
        
        response += f"\n*Alert period: {alert_period} days ahead*"
        
        return {
            'answer': response,
            'confidence_level': '✅ High Confidence',
            'confidence_score': 0.95,
            'response_type': 'data_query_contracts',
            'data': data
        }
    
    def _format_summary_response(self, query: str, data: Dict) -> Dict:
        """Format dashboard summary into conversational response"""
        if 'error' in data:
            return {'answer': f"I had trouble retrieving HR summary data: {data['error']}", 'confidence_level': '❌ Error', 'confidence_score': 0.0}
        
        summary = data.get('summary', {})
        
        response = "📊 **HR Dashboard Summary**\n\n"
        response += f"👥 **Total Employees:** {summary.get('total_employees', 0)}\n"
        response += f"📈 **Attrition Rate:** {summary.get('attrition_rate', 0)}% (last 3 months)\n"
        response += f"🎯 **Appraisal Completion:** {summary.get('appraisal_completion', 0)}%\n"
        response += f"🚨 **Active Alerts:** {summary.get('total_alerts', 0)} (probation + contract expiry)\n"
        
        if summary.get('total_alerts', 0) > 0:
            response += f"\n💡 **Tip:** Ask me about specific areas like 'probation status' or 'contract expiry alerts' for detailed information."
        
        response += f"\n\n*Data last updated: {data.get('last_updated', 'Unknown')}*"
        
        return {
            'answer': response,
            'confidence_level': '✅ High Confidence',
            'confidence_score': 0.95,
            'response_type': 'data_query_summary',
            'data': data
        }
    
    def ask(self, question: str) -> Dict:
        """Main query handler - routes to appropriate engine"""
        try:
            logger.info(f"Routing query: {question}")
            
            # Analyze query intent
            query_type, data_type, metadata = self.analyze_query_intent(question)
            
            logger.info(f"Query routed to: {query_type} ({data_type})")
            
            if query_type == 'data_query':
                response = self.handle_data_query(question, data_type)
                response['query_type'] = 'data_query'
                response['data_type'] = data_type
                response['routing_metadata'] = metadata
                return response
            else:
                # Use RAG engine for document queries
                response = self.rag_engine.ask(question)
                response['query_type'] = 'document_query'
                response['data_type'] = data_type
                response['routing_metadata'] = metadata
                return response
                
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}. Please try again.",
                'confidence_level': '❌ Error',
                'confidence_score': 0.0,
                'query_type': 'error',
                'error': str(e)
            }