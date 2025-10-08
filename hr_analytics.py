# -*- coding: utf-8 -*-
"""
HR Analytics Service for structured data queries
Handles headcount, attrition, probation, and appraisal data
"""
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
from supabase import Client
from knowledge_base import HRKnowledgeBaseClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRAnalyticsService:
    """Service for HR analytics and structured data queries"""
    
    def __init__(self):
        """Initialize with Supabase client"""
        kb_client = HRKnowledgeBaseClient()
        self.supabase = kb_client.supabase
        logger.info("HR Analytics Service initialized")
    
    # HEADCOUNT ANALYTICS
    def get_current_headcount(self) -> Dict:
        """Get current total headcount with breakdown"""
        try:
            # Get active employees
            result = self.supabase.table('people').select(
                'id, employment_status, org_unit_id, hr_role, started_on'
            ).eq('employment_status', 'active').execute()
            
            employees = result.data
            total_count = len(employees)
            
            # Department breakdown
            dept_result = self.supabase.table('people').select(
                'org_unit_id, org_unit(name)'
            ).eq('employment_status', 'active').execute()
            
            dept_counts = {}
            for emp in dept_result.data:
                org_unit = emp.get('org_unit', {})
                dept_name = org_unit.get('name', 'Unknown') if org_unit else 'Unknown'
                dept_counts[dept_name] = dept_counts.get(dept_name, 0) + 1
            
            # Role breakdown
            role_counts = {}
            for emp in employees:
                role = emp.get('hr_role', 'employee')
                role_counts[role] = role_counts.get(role, 0) + 1
            
            return {
                'total_headcount': total_count,
                'by_department': dept_counts,
                'by_role': role_counts,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting headcount: {e}")
            return {'error': str(e)}
    
    def get_headcount_trends(self, months: int = 6) -> Dict:
        """Get headcount trends over time"""
        try:
            # Get hiring trends
            start_date = (datetime.now() - timedelta(days=months*30)).date()
            
            hires_result = self.supabase.table('people').select(
                'started_on'
            ).gte('started_on', start_date).execute()
            
            # Get termination trends (using ended_on field)
            terms_result = self.supabase.table('people').select(
                'ended_on'
            ).gte('ended_on', start_date).execute()
            
            # Process monthly data
            monthly_data = {}
            
            # Process hires
            for person in hires_result.data:
                if person.get('started_on'):
                    month = person['started_on'][:7]  # YYYY-MM format
                    if month not in monthly_data:
                        monthly_data[month] = {'hires': 0, 'terminations': 0}
                    monthly_data[month]['hires'] += 1
            
            # Process terminations
            for person in terms_result.data:
                if person.get('ended_on'):
                    month = person['ended_on'][:7]  # YYYY-MM format
                    if month not in monthly_data:
                        monthly_data[month] = {'hires': 0, 'terminations': 0}
                    monthly_data[month]['terminations'] += 1
            
            return {
                'monthly_trends': monthly_data,
                'period_months': months,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting headcount trends: {e}")
            return {'error': str(e)}
    
    # ATTRITION ANALYTICS
    def get_attrition_data(self, period_months: int = 12) -> Dict:
        """Calculate attrition rates and trends"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=period_months*30)
            
            # Get terminations in period
            terms_result = self.supabase.table('people').select(
                'id, ended_on, org_unit_id, org_unit(name)'
            ).gte('ended_on', start_date).lte('ended_on', end_date).execute()
            
            terminations = terms_result.data
            
            # Get average headcount for the period (approximate)
            current_headcount = self.get_current_headcount()['total_headcount']
            
            # Calculate attrition rate
            attrition_count = len(terminations)
            attrition_rate = (attrition_count / max(current_headcount, 1)) * 100
            
            # Department-wise attrition
            dept_attrition = {}
            for term in terminations:
                org_unit = term.get('org_unit', {})
                dept_name = org_unit.get('name', 'Unknown') if org_unit else 'Unknown'
                dept_attrition[dept_name] = dept_attrition.get(dept_name, 0) + 1
            
            return {
                'period_months': period_months,
                'total_terminations': attrition_count,
                'attrition_rate_percent': round(attrition_rate, 2),
                'by_department': dept_attrition,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating attrition: {e}")
            return {'error': str(e)}
    
    # PROBATION TRACKING
    def get_probation_alerts(self) -> Dict:
        """Get employees with upcoming or overdue probation reviews"""
        try:
            today = datetime.now().date()
            warning_date = today + timedelta(days=14)  # 2 weeks notice
            
            # Get contracts with probation periods
            contracts_result = self.supabase.table('employment_contract').select(
                '*, people(id, first_name, last_name, work_email, manager_id)'
            ).not_.is_('probation_end_date', 'null').execute()
            
            contracts = contracts_result.data
            
            upcoming_reviews = []
            overdue_reviews = []
            
            for contract in contracts:
                probation_end = datetime.strptime(contract['probation_end_date'], '%Y-%m-%d').date()
                person = contract.get('people', {})
                
                days_until_end = (probation_end - today).days
                
                contract_info = {
                    'person_id': person.get('id'),
                    'name': f"{person.get('first_name', '')} {person.get('last_name', '')}",
                    'email': person.get('work_email'),
                    'manager_id': person.get('manager_id'),
                    'probation_end_date': contract['probation_end_date'],
                    'days_until_end': days_until_end,
                    'contract_type': contract.get('contract_type')
                }
                
                if days_until_end < 0:
                    overdue_reviews.append(contract_info)
                elif days_until_end <= 14:
                    upcoming_reviews.append(contract_info)
            
            return {
                'upcoming_reviews': upcoming_reviews,
                'overdue_reviews': overdue_reviews,
                'total_alerts': len(upcoming_reviews) + len(overdue_reviews),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting probation alerts: {e}")
            return {'error': str(e)}
    
    # APPRAISAL TRACKING
    def get_appraisal_status(self) -> Dict:
        """Get current appraisal cycle status and completion rates - FIXED"""
        try:
            # Get current active appraisal cycle
            cycles_result = self.supabase.table('appraisal_cycle').select(
                '*'
            ).order('created_at', desc=True).limit(1).execute()
            
            if not cycles_result.data:
                return {
                    'message': 'No active appraisal cycles found',
                    'last_updated': datetime.now().isoformat()
                }
            
            current_cycle = cycles_result.data[0]
            cycle_id = current_cycle['id']
            
            # FIXED: Get appraisal records with explicit relationship specification
            records_result = self.supabase.table('appraisal_record').select(
                '*, people!appraisal_record_person_id_fkey(first_name, last_name, work_email, org_unit_id, org_unit(name))'
            ).eq('cycle_id', cycle_id).execute()
            
            records = records_result.data
            
            # Calculate completion stats
            total_records = len(records)
            completed_records = len([r for r in records if r.get('status') == 'completed'])
            completion_rate = (completed_records / max(total_records, 1)) * 100
            
            # Department-wise completion
            dept_completion = {}
            for record in records:
                person = record.get('people', {})
                if person:  # Check if person data exists
                    org_unit = person.get('org_unit', {})
                    dept_name = org_unit.get('name', 'Unknown') if org_unit else 'Unknown'
                    
                    if dept_name not in dept_completion:
                        dept_completion[dept_name] = {'total': 0, 'completed': 0}
                    
                    dept_completion[dept_name]['total'] += 1
                    if record.get('status') == 'completed':
                        dept_completion[dept_name]['completed'] += 1
            
            # Calculate department completion rates
            for dept in dept_completion:
                total = dept_completion[dept]['total']
                completed = dept_completion[dept]['completed']
                dept_completion[dept]['completion_rate'] = (completed / max(total, 1)) * 100
            
            # Find overdue appraisals (if cycle end date has passed)
            cycle_end = datetime.strptime(current_cycle['end_date'], '%Y-%m-%d').date()
            is_overdue = datetime.now().date() > cycle_end
            
            return {
                'cycle_info': {
                    'name': current_cycle['name'],
                    'year': current_cycle['year'],
                    'stage': current_cycle['stage'],
                    'end_date': current_cycle['end_date'],
                    'is_overdue': is_overdue
                },
                'completion_stats': {
                    'total_appraisals': total_records,
                    'completed_appraisals': completed_records,
                    'completion_rate_percent': round(completion_rate, 2),
                    'pending_appraisals': total_records - completed_records
                },
                'by_department': dept_completion,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting appraisal status: {e}")
            return {'error': str(e)}
    
    # CONTRACT TRACKING
    def get_contract_expiry_alerts(self, days_ahead: int = 30) -> Dict:
        """Get contracts expiring soon"""
        try:
            today = datetime.now().date()
            alert_date = today + timedelta(days=days_ahead)
            
            # Get contracts with end dates approaching
            contracts_result = self.supabase.table('employment_contract').select(
                '*, people(first_name, last_name, work_email, manager_id)'
            ).not_.is_('end_date', 'null').lte('end_date', alert_date).gte('end_date', today).execute()
            
            contracts = contracts_result.data
            
            expiring_contracts = []
            for contract in contracts:
                person = contract.get('people', {})
                end_date = datetime.strptime(contract['end_date'], '%Y-%m-%d').date()
                days_until_expiry = (end_date - today).days
                
                expiring_contracts.append({
                    'person_id': person.get('id'),
                    'name': f"{person.get('first_name', '')} {person.get('last_name', '')}",
                    'email': person.get('work_email'),
                    'manager_id': person.get('manager_id'),
                    'contract_type': contract.get('contract_type'),
                    'end_date': contract['end_date'],
                    'days_until_expiry': days_until_expiry
                })
            
            return {
                'expiring_contracts': expiring_contracts,
                'total_expiring': len(expiring_contracts),
                'alert_period_days': days_ahead,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting contract expiry alerts: {e}")
            return {'error': str(e)}
    
    # SUMMARY METHODS
    def get_hr_dashboard_summary(self) -> Dict:
        """Get a comprehensive HR dashboard summary"""
        try:
            headcount = self.get_current_headcount()
            attrition = self.get_attrition_data(period_months=3)  # Last 3 months
            probation = self.get_probation_alerts()
            appraisals = self.get_appraisal_status()
            contracts = self.get_contract_expiry_alerts(days_ahead=30)
            
            # Calculate key metrics
            total_alerts = (
                probation.get('total_alerts', 0) + 
                contracts.get('total_expiring', 0)
            )
            
            return {
                'summary': {
                    'total_employees': headcount.get('total_headcount', 0),
                    'attrition_rate': attrition.get('attrition_rate_percent', 0),
                    'total_alerts': total_alerts,
                    'appraisal_completion': appraisals.get('completion_stats', {}).get('completion_rate_percent', 0)
                },
                'headcount': headcount,
                'attrition': attrition,
                'probation_alerts': probation,
                'appraisal_status': appraisals,
                'contract_alerts': contracts,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard summary: {e}")
            return {'error': str(e)}