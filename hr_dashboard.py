# -*- coding: utf-8 -*-
"""
HR Dashboard Component for Live Data Visualization
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
from hr_analytics import HRAnalyticsService

class HRDashboard:
    """Live HR Dashboard with real-time analytics"""
    
    def __init__(self):
        """Initialize dashboard with analytics service"""
        self.analytics = HRAnalyticsService()
    
    def render_dashboard(self):
        """Render the complete HR dashboard"""
        st.markdown("## 📊 Live HR Dashboard")
        st.markdown("*Real-time data from your HR system*")
        
        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            auto_refresh = st.checkbox("Auto-refresh (30s)")
        
        if auto_refresh:
            # Auto-refresh every 30 seconds
            time.sleep(30)
            st.rerun()
        
        # Get dashboard data
        with st.spinner("Loading dashboard data..."):
            dashboard_data = self.get_cached_dashboard_data()
        
        if 'error' in dashboard_data:
            st.error(f"❌ Error loading dashboard: {dashboard_data['error']}")
            return
        
        # Render sections
        self.render_key_metrics(dashboard_data)
        self.render_headcount_section(dashboard_data['headcount'])
        self.render_attrition_section(dashboard_data['attrition'])
        self.render_alerts_section(dashboard_data)
        self.render_appraisal_section(dashboard_data.get('appraisal_status'))
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_cached_dashboard_data(_self):
        """Get cached dashboard data"""
        return _self.analytics.get_hr_dashboard_summary()
    
    def render_key_metrics(self, data):
        """Render key metrics cards"""
        st.markdown("### 🎯 Key Metrics")
        
        summary = data.get('summary', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_employees = summary.get('total_employees', 0)
            st.metric(
                label="👥 Total Employees",
                value=f"{total_employees}",
                help="Current active headcount"
            )
        
        with col2:
            attrition_rate = summary.get('attrition_rate', 0)
            st.metric(
                label="📈 Attrition Rate",
                value=f"{attrition_rate}%",
                delta=f"Last 3 months",
                help="Employee departure rate"
            )
        
        with col3:
            appraisal_completion = summary.get('appraisal_completion', 0)
            st.metric(
                label="🎯 Appraisal Progress",
                value=f"{appraisal_completion}%",
                help="Current cycle completion rate"
            )
        
        with col4:
            total_alerts = summary.get('total_alerts', 0)
            alert_color = "🔴" if total_alerts > 5 else "🟡" if total_alerts > 0 else "🟢"
            st.metric(
                label=f"{alert_color} Active Alerts",
                value=f"{total_alerts}",
                help="Probation + Contract expiry alerts"
            )
    
    def render_headcount_section(self, headcount_data):
        """Render headcount visualization"""
        st.markdown("---")
        st.markdown("### 👥 Headcount Analysis")
        
        if 'error' in headcount_data:
            st.error(f"Error loading headcount data: {headcount_data['error']}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Department breakdown pie chart
            dept_data = headcount_data.get('by_department', {})
            if dept_data:
                fig_dept = px.pie(
                    values=list(dept_data.values()),
                    names=list(dept_data.keys()),
                    title="Employees by Department"
                )
                fig_dept.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_dept, use_container_width=True)
        
        with col2:
            # Role breakdown bar chart
            role_data = headcount_data.get('by_role', {})
            if role_data:
                fig_role = px.bar(
                    x=list(role_data.keys()),
                    y=list(role_data.values()),
                    title="Employees by Role",
                    labels={'x': 'Role', 'y': 'Count'}
                )
                st.plotly_chart(fig_role, use_container_width=True)
    
    def render_attrition_section(self, attrition_data):
        """Render attrition analysis"""
        st.markdown("---")
        st.markdown("### 📈 Attrition Analysis")
        
        if 'error' in attrition_data:
            st.error(f"Error loading attrition data: {attrition_data['error']}")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Attrition by department
            dept_attrition = attrition_data.get('by_department', {})
            if dept_attrition:
                fig_attrition = px.bar(
                    x=list(dept_attrition.keys()),
                    y=list(dept_attrition.values()),
                    title="Departures by Department (Last 12 months)",
                    labels={'x': 'Department', 'y': 'Departures'}
                )
                st.plotly_chart(fig_attrition, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Attrition Summary")
            
            total_departures = attrition_data.get('total_terminations', 0)
            attrition_rate = attrition_data.get('attrition_rate_percent', 0)
            period = attrition_data.get('period_months', 12)
            
            st.info(f"""
            **Period:** Last {period} months  
            **Total Departures:** {total_departures}  
            **Attrition Rate:** {attrition_rate}%
            """)
    
    def render_alerts_section(self, data):
        """Render alerts and notifications"""
        st.markdown("---")
        st.markdown("### 🚨 Active Alerts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⏰ Probation Alerts")
            probation_data = data.get('probation_alerts', {})
            
            if 'error' in probation_data:
                st.error("Error loading probation data")
            else:
                total_alerts = probation_data.get('total_alerts', 0)
                upcoming = len(probation_data.get('upcoming_reviews', []))
                overdue = len(probation_data.get('overdue_reviews', []))
                
                if total_alerts == 0:
                    st.success("✅ No probation alerts")
                else:
                    st.warning(f"⚠️ {total_alerts} probation alerts")
                    st.write(f"• **Upcoming:** {upcoming} reviews")
                    st.write(f"• **Overdue:** {overdue} reviews")
                
                if st.button("View Details", key="probation_details"):
                    self.show_probation_details(probation_data)
        
        with col2:
            st.markdown("#### 📋 Contract Alerts")
            contract_data = data.get('contract_alerts', {})
            
            if 'error' in contract_data:
                st.error("Error loading contract data")
            else:
                expiring = contract_data.get('total_expiring', 0)
                
                if expiring == 0:
                    st.success("✅ No contract alerts")
                else:
                    st.warning(f"⚠️ {expiring} contracts expiring soon")
                
                if st.button("View Details", key="contract_details"):
                    self.show_contract_details(contract_data)
    
    def render_appraisal_section(self, appraisal_data):
        """Render appraisal progress"""
        st.markdown("---")
        st.markdown("### 🎯 Appraisal Progress")
        
        if not appraisal_data or 'error' in appraisal_data:
            st.info("📋 No active appraisal cycle or data unavailable")
            return
        
        if 'message' in appraisal_data:
            st.info(f"📋 {appraisal_data['message']}")
            return
        
        cycle_info = appraisal_data.get('cycle_info', {})
        completion_stats = appraisal_data.get('completion_stats', {})
        
        # Progress bar
        completion_rate = completion_stats.get('completion_rate_percent', 0)
        st.progress(completion_rate / 100)
        st.write(f"**{cycle_info.get('name', 'Current Cycle')}:** {completion_rate}% complete")
        
        col1, col2 = st.columns(2)
        
        with col1:
            completed = completion_stats.get('completed_appraisals', 0)
            total = completion_stats.get('total_appraisals', 0)
            pending = completion_stats.get('pending_appraisals', 0)
            
            st.metric("✅ Completed", f"{completed}/{total}")
            st.metric("⏳ Pending", pending)
        
        with col2:
            end_date = cycle_info.get('end_date')
            is_overdue = cycle_info.get('is_overdue', False)
            
            if is_overdue:
                st.error(f"🚨 Cycle overdue (ended {end_date})")
            else:
                st.info(f"📅 Cycle ends: {end_date}")
        
        # Department progress
        dept_data = appraisal_data.get('by_department', {})
        if dept_data:
            st.markdown("#### 📊 Progress by Department")
            
            dept_names = list(dept_data.keys())
            completion_rates = [dept_data[dept]['completion_rate'] for dept in dept_names]
            
            fig_progress = px.bar(
                x=dept_names,
                y=completion_rates,
                title="Appraisal Completion by Department",
                labels={'x': 'Department', 'y': 'Completion Rate (%)'}
            )
            fig_progress.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_progress, use_container_width=True)
    
    def show_probation_details(self, probation_data):
        """Show detailed probation information"""
        with st.expander("👥 Probation Review Details", expanded=True):
            upcoming = probation_data.get('upcoming_reviews', [])
            overdue = probation_data.get('overdue_reviews', [])
            
            if overdue:
                st.markdown("#### 🚨 Overdue Reviews")
                for review in overdue:
                    days_overdue = abs(review.get('days_until_end', 0))
                    st.error(f"**{review.get('name', 'Unknown')}** - {days_overdue} days overdue")
            
            if upcoming:
                st.markdown("#### ⏳ Upcoming Reviews")
                for review in upcoming:
                    days_remaining = review.get('days_until_end', 0)
                    st.warning(f"**{review.get('name', 'Unknown')}** - {days_remaining} days remaining")
    
    def show_contract_details(self, contract_data):
        """Show detailed contract information"""
        with st.expander("📋 Contract Expiry Details", expanded=True):
            contracts = contract_data.get('expiring_contracts', [])
            
            if contracts:
                for contract in contracts:
                    name = contract.get('name', 'Unknown')
                    days = contract.get('days_until_expiry', 0)
                    contract_type = contract.get('contract_type', 'Unknown')
                    
                    if days <= 7:
                        st.error(f"**{name}** ({contract_type}) - {days} days remaining")
                    else:
                        st.warning(f"**{name}** ({contract_type}) - {days} days remaining")