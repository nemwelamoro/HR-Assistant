# -*- coding: utf-8 -*-
"""
Role-based Streamlit web interface for HR RAG system with improved UI layout
- Employee View: Basic HR queries and self-service
- HR Personnel View: Advanced features, analytics, and management
"""
import streamlit as st
import os
from rag_engine import HRRAGEngine
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="HR Assistant - AI Powered",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .role-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .sidebar-section {
        margin: 20px 0;
        padding: 15px 0;
        border-bottom: 1px solid #e0e0e0;
    }
    .urgent-button {
        background-color: #ff4b4b !important;
        color: white !important;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def init_rag_engine():
    """Initialize RAG engine with caching"""
    if 'rag_engine' not in st.session_state:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("❌ GEMINI_API_KEY environment variable not set")
            st.stop()
        
        with st.spinner("Initializing HR Assistant..."):
            st.session_state.rag_engine = HRRAGEngine(api_key=gemini_api_key)
    
    return st.session_state.rag_engine

def display_response(response, show_metadata=False):
    """Display RAG response in a nice format"""
    
    # Main answer with better styling
    st.markdown("### 💬 Response")
    with st.container():
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #1f77b4;">
            {response['answer']}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Response metrics in a nice layout
    st.markdown("#### 📊 Response Quality")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        confidence_color = "🟢" if response['confidence_score'] > 0.8 else "🟡" if response['confidence_score'] > 0.6 else "🔴"
        st.markdown(f"""
        <div class="metric-container">
            <strong>Confidence</strong><br>
            {confidence_color} {response['confidence_level']}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <strong>Score</strong><br>
            📈 {response['confidence_score']:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <strong>Sources</strong><br>
            📚 {response['sources_used']}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <strong>Coverage</strong><br>
            📄 {response['coverage'].title()}
        </div>
        """, unsafe_allow_html=True)
    
    # Sources section (role-specific)
    if response.get('chunks') and response['sources_used'] > 0:
        st.markdown("---")
        if st.session_state.user_role == "HR Personnel":
            st.markdown("#### 📚 Detailed Sources")
            for i, chunk in enumerate(response['chunks'], 1):
                with st.expander(f"📄 Source {i}: {chunk.get('article_title', 'Unknown Document')} (Relevance: {chunk.get('similarity', 0):.2f})"):
                    st.markdown(chunk.get('content', 'No content available'))
        else:
            st.markdown("#### 📚 Information Sources")
            doc_names = list(set(chunk.get('article_title', 'Unknown Document') for chunk in response['chunks']))
            for doc in doc_names:
                st.markdown(f"📋 {doc}")
    
    # Technical metadata (HR Personnel only)
    if show_metadata and st.session_state.user_role == "HR Personnel":
        st.markdown("---")
        with st.expander("🔧 Technical Details"):
            st.json(response.get('metadata', {}))

def setup_sidebar():
    """Setup role-specific sidebar content"""
    
    with st.sidebar:
        # Role switcher at the top of sidebar
        st.markdown("## 🔄 Role Selection")
        role_options = ["Employee", "HR Personnel"]
        current_role = st.selectbox(
            "Select your role:",
            role_options,
            index=role_options.index(st.session_state.user_role),
            key="role_selector"
        )
        
        # Update role if changed
        if current_role != st.session_state.user_role:
            st.session_state.user_role = current_role
            st.session_state.chat_history = []  # Clear history on role change
            st.rerun()
        
        st.markdown("---")
        
        # Role-specific sidebar content
        if st.session_state.user_role == "Employee":
            return setup_employee_sidebar()
        else:
            return setup_hr_sidebar()

def setup_employee_sidebar():
    """Sidebar content for employees"""
    
    # Employee Self-Service Section
    st.markdown("## 👤 Employee Self-Service")
    with st.expander("🔍 What can I ask?", expanded=True):
        st.markdown("""
        • **Leave & Time Off** - Vacation, sick leave, PTO policies
                    
        • **Benefits** - Health insurance, retirement, perks
                    
        • **Policies** - Dress code, remote work, company rules 
                     
        • **Payroll** - Salary, deductions, tax information
                    
        • **Training** - Available courses, skill development
                    
        • **IT Support** - Equipment, software, troubleshooting
        """)
    
    st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
    
    # Need More Help Section
    st.markdown("## 📞 Need More Help?")
    with st.expander("📧 Contact Information"):
        st.markdown("""
        **HR Department:**
        • 📧 Email: hr@adanianlabs.io
        • 📱 Phone: (555) 123-4567
        • 💬 Teams: @HR-Support
        • 🕒 Hours: Mon-Fri 9AM-5PM
        
        **IT Support:**
        • 📧 Email: it-support@company.com
        • 📱 Phone: (555) 123-4568
        • 🎫 Ticket System: Available 24/7
        """)
    
    with st.expander("🆘 Emergency Contacts"):
        st.markdown("""
        **Security:** (555) 911-SAFE
        **Medical Emergency:** 911
        **Employee Assistance Program:** (555) 123-HELP
        """)
    
    st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
    
    # Settings Section
    st.markdown("## ⚙️ Settings")
    show_sources = st.checkbox("Show information sources", value=True, help="Display which documents were used to answer your question")
    
    # Quick Actions
    st.markdown("## ⚡ Quick Actions")
    if st.button("📝 Request Time Off", use_container_width=True):
        st.session_state.current_query = "How do I request time off?"
        st.rerun()
    
    if st.button("💰 Check Benefits", use_container_width=True):
        st.session_state.current_query = "What benefits am I entitled to?"
        st.rerun()
    
    if st.button("👔 Dress Code Info", use_container_width=True):
        st.session_state.current_query = "What is the company dress code?"
        st.rerun()
    
    return show_sources, False  # show_sources, show_metadata

def setup_hr_sidebar():
    """Sidebar content for HR personnel"""
    
    # Advanced Features Section
    st.markdown("## 🏢 Advanced Features")
    with st.expander("🎯 HR Management Tools", expanded=True):
        st.markdown("""
        • **Policy Management** - View, update, analyze policies
                    
        • **Compliance Tracking** - Legal requirements, audits
                    
        • **Analytics Dashboard** - Query patterns, response quality
                    
        • **Knowledge Base Admin** - Content management, updates
                    
        • **Employee Query Insights** - Common questions, trends
        """)
    
    st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
    
    # Knowledge Base Statistics
    st.markdown("## 📊 Knowledge Base Stats")
    try:
        rag = st.session_state.rag_engine
        stats = rag.kb_client.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Articles", stats.get('total_articles', 0))
            st.metric("🧩 Chunks", stats.get('total_chunks', 0))
        with col2:
            st.metric("📊 Avg Chunks", f"{stats.get('avg_chunks_per_article', 0):.1f}")
            
        # Session Analytics
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            st.markdown("### 📈 Session Analytics")
            total_queries = len(st.session_state.chat_history)
            avg_confidence = sum(chat['response']['confidence_score'] for chat in st.session_state.chat_history) / total_queries
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔍 Queries", total_queries)
            with col2:
                st.metric("✅ Avg Confidence", f"{avg_confidence:.2f}")
            
    except Exception as e:
        st.error(f"Could not load stats: {e}")
    
    st.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)
    
    # Settings Section
    st.markdown("## ⚙️ Advanced Settings")
    show_sources = st.checkbox("Show detailed sources", value=True, help="Display full source content and relevance scores")
    show_metadata = st.checkbox("Show response metadata", value=False, help="Display technical response details")
    
    # Advanced Options
    with st.expander("🔧 Advanced Options"):
        st.markdown("**Query Processing:**")
        enable_urgent = st.checkbox("Enable urgent query mode", value=True)
        auto_escalate = st.checkbox("Auto-escalate low confidence responses", value=False)
        
        st.markdown("**Analytics:**")
        track_queries = st.checkbox("Track query analytics", value=True)
        export_enabled = st.checkbox("Enable data export", value=True)
    
    # Quick HR Actions
    st.markdown("## ⚡ Quick HR Actions")
    if st.button("👥 Employee Onboarding", use_container_width=True):
        st.session_state.current_query = "What is the complete employee onboarding process?"
        st.rerun()
    
    if st.button("📋 Performance Reviews", use_container_width=True):
        st.session_state.current_query = "What are the performance review procedures?"
        st.rerun()
    
    if st.button("⚖️ Compliance Check", use_container_width=True):
        st.session_state.current_query = "What are our current compliance requirements?"
        st.rerun()
    
    if st.button("📊 Generate Report", use_container_width=True):
        st.session_state.current_query = "How do I generate employee reports?"
        st.rerun()
    
    return show_sources, show_metadata

def get_role_specific_questions():
    """Get sample questions based on user role"""
    
    employee_questions = [
        "What is the company's leave policy?",
        "How do I request time off?",
        "What benefits am I entitled to?",
        "What is the dress code policy?",
        "How do I access my payslip?",
        "What should I do if I'm sick?",
        "How do I update my personal information?",
        "What training opportunities are available?"
    ]
    
    hr_questions = [
        "What are the performance review procedures?",
        "How do we handle employee disciplinary actions?",
        "What is the recruitment and hiring process?",
        "What are the compliance requirements for payroll?",
        "How do we process employee terminations?",
        "What documentation is required for new hires?",
        "What are the legal requirements for employee records?",
        "How do we handle workplace harassment complaints?"
    ]
    
    if st.session_state.user_role == "Employee":
        return employee_questions
    else:
        return hr_questions

def main():
    """Main Streamlit app with improved UI layout"""
    
    # Initialize session state
    if 'user_role' not in st.session_state:
        st.session_state.user_role = "Employee"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    # Initialize RAG engine
    rag = init_rag_engine()
    
    # Setup sidebar and get settings
    show_sources, show_metadata = setup_sidebar()
    
    # Main page header
    st.markdown('<h1 class="main-header">🤖 HR Assistant - AI Powered</h1>', unsafe_allow_html=True)
    
    # Role-specific welcome section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.user_role == "Employee":
            st.markdown("""
            <div class="role-info">
                <h3>👤 Welcome, Employee!</h3>
                <p>Get instant answers to your HR questions. Ask about policies, benefits, procedures, and more.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="role-info">
                <h3>🏢 Welcome, HR Professional!</h3>
                <p>Access advanced HR management features, analytics, and comprehensive policy information.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Status indicator
        status_color = "🟢" if st.session_state.rag_engine else "🔴"
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <strong>System Status</strong><br>
            {status_color} Online
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main chat interface
    st.markdown("### 💭 Ask Your Question")
    
    # Query input form
    with st.form(key="query_form", clear_on_submit=True):
        placeholder_text = {
            "Employee": "e.g., How do I request vacation time?",
            "HR Personnel": "e.g., What is the process for employee performance reviews?"
        }
        
        query = st.text_area(
            "Type your question here:",
            placeholder=placeholder_text[st.session_state.user_role],
            value=st.session_state.current_query,
            height=100
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 2, 6])
        
        with col1:
            ask_button = st.form_submit_button("🚀 Ask Question", type="primary", use_container_width=True)
        
        with col2:
            if st.session_state.user_role == "HR Personnel":
                urgent_button = st.form_submit_button("🚨 Urgent Query", type="secondary", use_container_width=True)
            else:
                urgent_button = False
        
        with col3:
            clear_button = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)
    
    # Handle form submissions
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.current_query = ""
        st.rerun()
    
    # Process query
    if (ask_button or urgent_button) and query.strip():
        # Add urgency context for HR personnel
        if urgent_button and st.session_state.user_role == "HR Personnel":
            query = f"[URGENT] {query}"
        
        with st.spinner("🤔 Thinking... Please wait"):
            response = rag.ask(query, include_sources=show_sources)
            
            # Add role context to response
            response['user_role'] = st.session_state.user_role
            response['is_urgent'] = urgent_button
            
            # Add to chat history
            st.session_state.chat_history.append({
                'query': query,
                'response': response,
                'timestamp': datetime.now(),
                'role': st.session_state.user_role
            })
        
        st.session_state.current_query = ""
        st.rerun()
    
    # Sample questions section
    if not st.session_state.chat_history:
        st.markdown("---")
        st.markdown(f"### 🎯 Popular {st.session_state.user_role} Questions")
        st.markdown("*Click on any question to try it:*")
        
        sample_questions = get_role_specific_questions()
        
        # Create 2 columns for questions
        col1, col2 = st.columns(2)
        
        for i, question in enumerate(sample_questions):
            col_idx = i % 2
            with [col1, col2][col_idx]:
                if st.button(question, key=f"sample_{i}", use_container_width=True):
                    st.session_state.current_query = question
                    st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📝 Conversation History")
        
        # Show recent conversations
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):  # Show last 3
            chat_number = len(st.session_state.chat_history) - i
            
            # Chat header with role and urgency indicators
            role_emoji = "👤" if chat['role'] == "Employee" else "🏢"
            urgent_indicator = "🚨 " if chat['response'].get('is_urgent', False) else ""
            
            st.markdown(f"#### {role_emoji} {urgent_indicator}Question {chat_number}")
            st.markdown(f"**{chat['query']}**")
            st.caption(f"Asked at: {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ({chat['role']} View)")
            
            # Display response
            display_response(chat['response'], show_metadata)
            
            st.markdown("---")
        
        # Show more conversations if available
        if len(st.session_state.chat_history) > 3:
            with st.expander(f"📚 View All {len(st.session_state.chat_history)} Conversations"):
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    chat_number = len(st.session_state.chat_history) - i
                    role_emoji = "👤" if chat['role'] == "Employee" else "🏢"
                    
                    with st.container():
                        st.markdown(f"**{role_emoji} Q{chat_number}:** {chat['query']}")
                        st.markdown(f"**Answer:** {chat['response']['answer'][:150]}...")
                        st.markdown(f"**Confidence:** {chat['response']['confidence_level']} | **Time:** {chat['timestamp'].strftime('%H:%M:%S')}")
                        st.markdown("---")

if __name__ == "__main__":
    main()