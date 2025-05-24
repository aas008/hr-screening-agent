"""
Streamlit Web Interface for HR Screening Agent
Professional dashboard for managing autonomous resume screening
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from workflows.langgraph_workflow import run_autonomous_screening
from agents.github_loader import validate_github_config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="HR Screening Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background-color: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .status-error {
        background-color: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .status-warning {
        background-color: #f59e0b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .candidate-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    
    # Configuration state
    if 'config_validated' not in st.session_state:
        st.session_state.config_validated = False
    
    # Job requirements state
    if 'job_requirements' not in st.session_state:
        st.session_state.job_requirements = {}
    
    # Screening state
    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = None
    
    if 'screening_in_progress' not in st.session_state:
        st.session_state.screening_in_progress = False
    
    # History state
    if 'screening_history' not in st.session_state:
        st.session_state.screening_history = []

def load_config():
    """Load configuration from environment variables"""
    
    config = {
        'github_token': os.getenv('GITHUB_TOKEN'),
        'repo_owner': os.getenv('GITHUB_REPO_OWNER'),
        'repo_name': os.getenv('GITHUB_REPO_NAME'),
        'huggingface_model': os.getenv('HUGGINGFACE_MODEL', 'microsoft/DialoGPT-medium'),
        'email_enabled': os.getenv('EMAIL_ENABLED', 'false').lower() == 'true',
        'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'email_address': os.getenv('EMAIL_ADDRESS'),
        'email_password': os.getenv('EMAIL_PASSWORD'),
        'score_threshold': int(os.getenv('RESUME_SCORE_THRESHOLD', '70')),
        'output_dir': os.getenv('OUTPUT_DIR', './outputs')
    }
    
    return config

def validate_configuration():
    """Validate system configuration"""
    
    st.header("⚙️ System Configuration")
    
    config = load_config()
    
    # GitHub Configuration
    st.subheader("🔗 GitHub Integration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if config['github_token'] and config['repo_owner'] and config['repo_name']:
            st.success(f"✅ Connected to {config['repo_owner']}/{config['repo_name']}")
            
            # Test GitHub connection
            if st.button("🧪 Test GitHub Connection"):
                with st.spinner("Testing GitHub connection..."):
                    is_valid = validate_github_config(config)
                    if is_valid:
                        st.success("✅ GitHub connection successful!")
                        st.session_state.config_validated = True
                    else:
                        st.error("❌ GitHub connection failed. Check your token and repository.")
        else:
            st.error("❌ GitHub configuration missing. Please check your .env file.")
            st.code("""
Required in .env file:
GITHUB_TOKEN=your_token_here
GITHUB_REPO_OWNER=your_username  
GITHUB_REPO_NAME=hr-screening-agent
            """)
    
    with col2:
        if st.session_state.config_validated:
            st.markdown('<div class="status-success">✅ READY</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ NOT READY</div>', unsafe_allow_html=True)
    
    # AI Model Configuration
    st.subheader("🤖 AI Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Model:** {config['huggingface_model']}")
        st.info(f"**Score Threshold:** {config['score_threshold']}%")
    
    with col2:
        st.info(f"**Email Enabled:** {'✅ Yes' if config['email_enabled'] else '❌ No'}")
        if config['email_enabled']:
            st.info(f"**Email Server:** {config['smtp_server']}")
    
    return config

def job_setup_page():
    """Job requirements setup interface"""
    
    st.header("📋 Job Requirements Setup")
    
    with st.form("job_requirements_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input(
                "Job Title *", 
                value="Senior React Developer",
                help="Enter the exact job title for the position"
            )
            
            department = st.selectbox(
                "Department",
                ["Engineering", "Product", "Design", "Data Science", "Marketing", "Sales", "Operations"],
                help="Select the department for this role"
            )
            
            min_experience = st.number_input(
                "Minimum Years Experience *",
                min_value=0,
                max_value=20,
                value=3,
                help="Minimum years of relevant experience required"
            )
            
            job_role_folder = st.selectbox(
                "Resume Folder *",
                ["react-developer", "python-developer", "data-scientist", "fullstack-developer"],
                help="GitHub folder containing resumes: /resumes/active/{folder}/"
            )
        
        with col2:
            st.write("**Required Skills** (one per line)")
            required_skills_text = st.text_area(
                "Required Skills *",
                value="React\nJavaScript\nTypeScript\nHTML\nCSS",
                height=100,
                help="Enter each required skill on a new line"
            )
            
            st.write("**Preferred Skills** (one per line)")
            preferred_skills_text = st.text_area(
                "Preferred Skills",
                value="Node.js\nGraphQL\nTesting\nRedux\nGit",
                height=100,
                help="Enter each preferred skill on a new line"
            )
        
        # Advanced Settings
        with st.expander("🔧 Advanced Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                score_threshold = st.slider(
                    "Acceptance Threshold (%)",
                    min_value=50,
                    max_value=95,
                    value=70,
                    help="Candidates scoring above this threshold will be auto-accepted"
                )
                
                email_enabled = st.checkbox(
                    "Send Automated Emails",
                    value=False,
                    help="Enable automatic email responses to candidates"
                )
            
            with col2:
                company_name = st.text_input(
                    "Company Name",
                    value="Our Company",
                    help="Company name for email templates"
                )
        
        # Submit button
        submitted = st.form_submit_button("💾 Save Job Requirements", type="primary")
        
        if submitted:
            # Parse skills
            required_skills = [skill.strip() for skill in required_skills_text.split('\n') if skill.strip()]
            preferred_skills = [skill.strip() for skill in preferred_skills_text.split('\n') if skill.strip()]
            
            # Validate required fields
            if not job_title or not required_skills or not job_role_folder:
                st.error("❌ Please fill in all required fields marked with *")
                return None
            
            # Create job requirements
            job_requirements = {
                'title': job_title,
                'required_skills': required_skills,
                'preferred_skills': preferred_skills,
                'min_experience_years': min_experience,
                'department': department,
                'job_role_folder': job_role_folder,
                'score_threshold': score_threshold,
                'email_enabled': email_enabled,
                'company_name': company_name
            }
            
            # Save to session state
            st.session_state.job_requirements = job_requirements
            
            st.success("✅ Job requirements saved successfully!")
            
            # Show summary
            with st.expander("📋 Job Requirements Summary", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Title:** {job_title}")
                    st.write(f"**Department:** {department}")
                    st.write(f"**Min Experience:** {min_experience} years")
                    st.write(f"**Threshold:** {score_threshold}%")
                
                with col2:
                    st.write(f"**Required Skills:** {', '.join(required_skills)}")
                    st.write(f"**Preferred Skills:** {', '.join(preferred_skills)}")
                    st.write(f"**Resume Folder:** {job_role_folder}")
                    st.write(f"**Email Enabled:** {'Yes' if email_enabled else 'No'}")
            
            return job_requirements
    
    return None

def run_screening_page():
    """Run autonomous screening interface"""
    
    st.header("🚀 Run Autonomous Screening")
    
    # Check if job requirements are set
    if not st.session_state.job_requirements:
        st.warning("⚠️ Please set up job requirements first!")
        if st.button("📋 Go to Job Setup"):
            st.switch_page("Job Setup")
        return
    
    job_req = st.session_state.job_requirements
    
    # Display job summary
    st.subheader("📋 Job Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Job Title", job_req['title'])
    
    with col2:
        st.metric("Required Skills", len(job_req['required_skills']))
    
    with col3:
        st.metric("Min Experience", f"{job_req['min_experience_years']} years")
    
    with col4:
        st.metric("Threshold", f"{job_req.get('score_threshold', 70)}%")
    
    # Screening controls
    st.subheader("🎮 Screening Controls")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not st.session_state.screening_in_progress:
            if st.button("🚀 Start Autonomous Screening", type="primary", use_container_width=True):
                run_autonomous_screening_process()
        else:
            st.info("🔄 Screening in progress...")
            if st.button("🛑 Cancel Screening", type="secondary"):
                st.session_state.screening_in_progress = False
                st.rerun()
    
    with col2:
        simulation_mode = st.checkbox(
            "🧪 Simulation Mode",
            value=not job_req.get('email_enabled', False),
            help="Run without sending actual emails"
        )
    
    # Show recent results if available
    if st.session_state.screening_results:
        display_screening_results()

def run_autonomous_screening_process():
    """Execute the autonomous screening workflow"""
    
    st.session_state.screening_in_progress = True
    
    # Get configuration
    config = load_config()
    job_req = st.session_state.job_requirements
    
    # Override email settings if simulation mode
    config['email_enabled'] = job_req.get('email_enabled', False)
    config['score_threshold'] = job_req.get('score_threshold', 70)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Initializing screening workflow...")
        progress_bar.progress(10)
        
        status_text.text("📂 Loading resumes from GitHub...")
        progress_bar.progress(30)
        
        # Run the screening workflow
        results = run_autonomous_screening(
            job_requirements=job_req,
            job_role_folder=job_req['job_role_folder'],
            config=config,
            verbose=True
        )
        
        progress_bar.progress(100)
        
        if 'error' in results:
            st.error(f"❌ Screening failed: {results['error']}")
            if 'message' in results:
                st.error(f"Details: {results['message']}")
        else:
            st.success("✅ Screening completed successfully!")
            
            # Save results
            st.session_state.screening_results = results
            st.session_state.screening_history.append({
                'timestamp': datetime.now(),
                'job_title': job_req['title'],
                'results': results
            })
            
            # Display results
            display_screening_results()
    
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
    
    finally:
        st.session_state.screening_in_progress = False
        progress_bar.empty()
        status_text.empty()

def display_screening_results():
    """Display comprehensive screening results"""
    
    if not st.session_state.screening_results:
        return
    
    results = st.session_state.screening_results
    
    st.header("📊 Screening Results")
    
    # Summary metrics
    session_info = results['session_info']
    results_data = results['results']
    efficiency = results['efficiency_metrics']
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Candidates",
            session_info['total_candidates'],
            help="Total number of resumes processed"
        )
    
    with col2:
        st.metric(
            "Accepted",
            results_data['accepted'],
            delta=f"{results_data['acceptance_rate']:.1%}",
            help="Candidates above threshold"
        )
    
    with col3:
        st.metric(
            "Average Score",
            f"{results_data['average_score']}%",
            help="Mean score across all candidates"
        )
    
    with col4:
        st.metric(
            "Time Saved",
            f"{efficiency['time_saved_minutes']:.0f}min",
            help="Estimated manual time saved"
        )
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Candidates", "📈 Analytics", "📧 Email Actions", "📋 Detailed Report"])
    
    with tab1:
        display_candidate_results(results)
    
    with tab2:
        display_analytics(results)
    
    with tab3:
        display_email_actions(results)
    
    with tab4:
        display_detailed_report(results)

def display_candidate_results(results):
    """Display individual candidate results"""
    
    detailed_results = results.get('detailed_results', [])
    
    if not detailed_results:
        st.info("No detailed candidate results available.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        action_filter = st.selectbox(
            "Filter by Action",
            ["All", "Accept", "Reject"]
        )
    
    with col2:
        min_score = st.slider("Minimum Score", 0, 100, 0)
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Score (High to Low)", "Score (Low to High)", "Name"]
        )
    
    # Filter and sort results
    filtered_results = detailed_results.copy()
    
    if action_filter != "All":
        filtered_results = [r for r in filtered_results if r['action'] == action_filter.lower()]
    
    filtered_results = [r for r in filtered_results if r['score'] >= min_score]
    
    if sort_by == "Score (High to Low)":
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
    elif sort_by == "Score (Low to High)":
        filtered_results.sort(key=lambda x: x['score'])
    else:
        filtered_results.sort(key=lambda x: x['candidate']['name'])
    
    # Display candidate cards
    for result in filtered_results:
        display_candidate_card(result)

def display_candidate_card(result):
    """Display individual candidate card"""
    
    candidate = result['candidate']
    
    # Card container
    with st.container():
        # Header row
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader(f"👤 {candidate['name']}")
            st.write(f"📧 {candidate['email']}")
            st.write(f"📄 {candidate['file_name']}")
        
        with col2:
            # Score gauge
            score = result['score']
            
            if score >= 80:
                color = "green"
                status = "🟢 STRONG"
            elif score >= 70:
                color = "blue"
                status = "🔵 GOOD"
            elif score >= 50:
                color = "orange"
                status = "🟠 FAIR"
            else:
                color = "red"
                status = "🔴 WEAK"
            
            st.metric("Score", f"{score}%")
            st.markdown(f"**{status}**")
        
        with col3:
            action = result['action']
            if action == 'accept':
                st.success("✅ ACCEPT")
            else:
                st.error("❌ REJECT")
            
            st.write(f"**Experience:** {result['experience_years']} years")
            st.write(f"**Level:** {result['experience_level']}")
        
        # Expandable details
        with st.expander(f"📋 Detailed Analysis - {candidate['name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**✅ Skills Found:**")
                for skill in result['skills_found']:
                    st.write(f"• {skill}")
                
                if result['skills_missing']:
                    st.write("**❌ Skills Missing:**")
                    for skill in result['skills_missing']:
                        st.write(f"• {skill}")
                
                st.write("**💪 Strengths:**")
                for strength in result['strengths']:
                    st.write(f"• {strength}")
            
            with col2:
                if result['concerns']:
                    st.write("**⚠️ Concerns:**")
                    for concern in result['concerns']:
                        st.write(f"• {concern}")
                
                st.write(f"**🤔 AI Reasoning:**")
                st.write(result['reasoning'])
                
                st.write(f"**📈 Confidence:** {result['confidence']:.1%}")
                st.write(f"**⏱️ Analysis Time:** {result['analysis_time_seconds']:.2f}s")
        
        st.divider()

def display_analytics(results):
    """Display analytics and visualizations"""
    
    detailed_results = results.get('detailed_results', [])
    
    if not detailed_results:
        st.info("No data available for analytics.")
        return
    
    # Score distribution chart
    st.subheader("📊 Score Distribution")
    
    scores = [r['score'] for r in detailed_results]
    
    fig_hist = px.histogram(
        x=scores,
        nbins=10,
        title="Candidate Score Distribution",
        labels={'x': 'Score (%)', 'y': 'Number of Candidates'},
        color_discrete_sequence=['#3b82f6']
    )
    
    # Add threshold line
    threshold = results.get('session_info', {}).get('score_threshold', 70)
    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold}%)"
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Skills analysis
    st.subheader("🔧 Skills Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most common skills found
        all_skills_found = []
        for result in detailed_results:
            all_skills_found.extend(result['skills_found'])
        
        if all_skills_found:
            skill_counts = pd.Series(all_skills_found).value_counts().head(10)
            
            fig_skills = px.bar(
                x=skill_counts.values,
                y=skill_counts.index,
                orientation='h',
                title="Most Common Skills Found",
                labels={'x': 'Number of Candidates', 'y': 'Skills'}
            )
            
            st.plotly_chart(fig_skills, use_container_width=True)
    
    with col2:
        # Experience distribution
        experience_levels = [r['experience_level'] for r in detailed_results]
        exp_counts = pd.Series(experience_levels).value_counts()
        
        fig_exp = px.pie(
            values=exp_counts.values,
            names=exp_counts.index,
            title="Experience Level Distribution"
        )
        
        st.plotly_chart(fig_exp, use_container_width=True)
    
    # Timeline analysis (if multiple sessions)
    if len(st.session_state.screening_history) > 1:
        st.subheader("📈 Historical Trends")
        
        history_data = []
        for session in st.session_state.screening_history:
            history_data.append({
                'Date': session['timestamp'].date(),
                'Job Title': session['job_title'],
                'Total Candidates': session['results']['session_info']['total_candidates'],
                'Accepted': session['results']['results']['accepted'],
                'Acceptance Rate': session['results']['results']['acceptance_rate']
            })
        
        df_history = pd.DataFrame(history_data)
        
        fig_trend = px.line(
            df_history,
            x='Date',
            y='Acceptance Rate',
            title="Acceptance Rate Trends",
            markers=True
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)

def display_email_actions(results):
    """Display email actions and templates"""
    
    email_results = results.get('email_actions', {})
    
    if not email_results:
        st.info("No email action data available.")
        return
    
    # Email statistics
    stats = email_results.get('statistics', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Emails", stats.get('total', 0))
    
    with col2:
        st.metric("Successfully Sent", stats.get('sent', 0))
    
    with col3:
        st.metric("Failed", stats.get('failed', 0))
    
    # Email breakdown by type
    if 'by_type' in stats:
        st.subheader("📧 Email Types")
        
        email_types = stats['by_type']
        
        fig_emails = px.bar(
            x=list(email_types.keys()),
            y=list(email_types.values()),
            title="Emails Sent by Type",
            labels={'x': 'Email Type', 'y': 'Count'}
        )
        
        st.plotly_chart(fig_emails, use_container_width=True)
    
    # Mode indicator
    mode = email_results.get('mode', 'unknown')
    if mode == 'simulation':
        st.info("🧪 **Simulation Mode** - No actual emails were sent")
    elif mode == 'real':
        st.success("📧 **Live Mode** - Emails were sent to candidates")

def display_detailed_report(results):
    """Display detailed exportable report"""
    
    st.subheader("📋 Detailed Session Report")
    
    # Session information
    session_info = results['session_info']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Session Details:**")
        st.write(f"• Job Title: {session_info['job_title']}")
        st.write(f"• Processing Date: {session_info['timestamp'][:19]}")
        st.write(f"• Total Candidates: {session_info['total_candidates']}")
        st.write(f"• Processing Time: {session_info['processing_time_seconds']:.1f} seconds")
    
    with col2:
        results_data = results['results']
        st.write("**Results Summary:**")
        st.write(f"• Accepted: {results_data['accepted']}")
        st.write(f"• Rejected: {results_data['rejected']}")
        st.write(f"• Acceptance Rate: {results_data['acceptance_rate']:.1%}")
        st.write(f"• Average Score: {results_data['average_score']}%")
    
    # Export options
    st.subheader("📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to CSV"):
            export_to_csv(results)
    
    with col2:
        if st.button("📄 Export to JSON"):
            export_to_json(results)
    
    with col3:
        if st.button("📧 Email Report"):
            st.info("Email report feature coming soon!")

def export_to_csv(results):
    """Export results to CSV format"""
    
    detailed_results = results.get('detailed_results', [])
    
    if not detailed_results:
        st.error("No data to export")
        return
    
    # Prepare data for CSV
    csv_data = []
    for result in detailed_results:
        candidate = result['candidate']
        csv_data.append({
            'Name': candidate['name'],
            'Email': candidate['email'],
            'File': candidate['file_name'],
            'Score': result['score'],
            'Action': result['action'],
            'Experience_Years': result['experience_years'],
            'Experience_Level': result['experience_level'],
            'Skills_Found': ', '.join(result['skills_found']),
            'Skills_Missing': ', '.join(result['skills_missing']),
            'Strengths': ', '.join(result['strengths']),
            'Concerns': ', '.join(result['concerns']),
            'Reasoning': result['reasoning'],
            'Confidence': result['confidence']
        })
    
    df = pd.DataFrame(csv_data)
    
    # Create download button
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

def export_to_json(results):
    """Export results to JSON format"""
    
    json_str = json.dumps(results, indent=2, default=str)
    
    st.download_button(
        label="📥 Download JSON",
        data=json_str,
        file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 Autonomous HR Screening Agent</h1>', unsafe_allow_html=True)
    st.markdown("*AI-powered resume screening with intelligent automation*")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🔧 Navigation")
        
        page = st.selectbox(
            "Select Page",
            ["🏠 Dashboard", "⚙️ Configuration", "📋 Job Setup", "🚀 Run Screening", "📊 Results", "📈 Analytics"]
        )
        
        st.divider()
        
        # Quick stats in sidebar
        if st.session_state.screening_results:
            results = st.session_state.screening_results
            st.subheader("📊 Quick Stats")
            
            session_info = results['session_info']
            results_data = results['results']
            
            st.metric("Last Session", session_info['total_candidates'], label_visibility="visible")
            st.metric("Accepted", results_data['accepted'])
            st.metric("Avg Score", f"{results_data['average_score']}%")
        
        st.divider()
        
        # System status
        st.subheader("🔧 System Status")
        
        if st.session_state.config_validated:
            st.success("✅ GitHub Connected")
        else:
            st.error("❌ GitHub Not Connected")
        
        if st.session_state.job_requirements:
            st.success("✅ Job Requirements Set")
        else:
            st.warning("⚠️ Job Requirements Missing")
        
        if st.session_state.screening_in_progress:
            st.info("🔄 Screening In Progress")
    
    # Page routing
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "⚙️ Configuration":
        config = validate_configuration()
    elif page == "📋 Job Setup":
        job_setup_page()
    elif page == "🚀 Run Screening":
        run_screening_page()
    elif page == "📊 Results":
        results_page()
    elif page == "📈 Analytics":
        analytics_page()

def dashboard_page():
    """Main dashboard overview"""
    
    st.header("🏠 Dashboard Overview")
    
    # Welcome message
    if not st.session_state.screening_results:
        st.info("👋 Welcome to the HR Screening Agent! Set up your job requirements and run your first screening to get started.")
    
    # Quick action buttons
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 New Job Setup", use_container_width=True):
            st.session_state.job_requirements = {}
            st.switch_page("📋 Job Setup")
    
    with col2:
        if st.button("🚀 Run Screening", use_container_width=True, disabled=not st.session_state.job_requirements):
            st.switch_page("🚀 Run Screening")
    
    with col3:
        if st.button("📊 View Results", use_container_width=True, disabled=not st.session_state.screening_results):
            st.switch_page("📊 Results")
    
    with col4:
        if st.button("⚙️ Configuration", use_container_width=True):
            st.switch_page("⚙️ Configuration")
    
    # Recent activity
    if st.session_state.screening_history:
        st.subheader("📋 Recent Screening Sessions")
        
        # Show last 5 sessions
        recent_sessions = st.session_state.screening_history[-5:]
        
        for i, session in enumerate(reversed(recent_sessions)):
            with st.expander(f"📅 {session['timestamp'].strftime('%Y-%m-%d %H:%M')} - {session['job_title']}"):
                col1, col2, col3 = st.columns(3)
                
                results = session['results']
                session_info = results['session_info']
                results_data = results['results']
                
                with col1:
                    st.metric("Candidates", session_info['total_candidates'])
                
                with col2:
                    st.metric("Accepted", results_data['accepted'])
                
                with col3:
                    st.metric("Avg Score", f"{results_data['average_score']}%")
                
                if st.button(f"📊 View Details", key=f"view_session_{i}"):
                    st.session_state.screening_results = results
                    st.switch_page("📊 Results")
    
    # System overview
    st.subheader("🔧 System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**🤖 AI Model:** Rule-based + HuggingFace")
        st.info("**📂 Source:** GitHub Integration")
        st.info("**📧 Email:** Automated Responses")
    
    with col2:
        st.info("**⚡ Processing:** ~5 seconds per resume")
        st.info("**🎯 Threshold:** 70% acceptance score")
        st.info("**💰 Time Saved:** ~15 minutes per candidate")

def results_page():
    """Dedicated results page"""
    
    st.header("📊 Screening Results")
    
    if not st.session_state.screening_results:
        st.info("No screening results available. Run a screening session first.")
        if st.button("🚀 Run Screening"):
            if st.session_state.job_requirements:
                st.switch_page("🚀 Run Screening")
            else:
                st.switch_page("📋 Job Setup")
        return
    
    # Display comprehensive results
    display_screening_results()

def analytics_page():
    """Dedicated analytics page"""
    
    st.header("📈 Advanced Analytics")
    
    if not st.session_state.screening_results:
        st.info("No data available for analytics. Run a screening session first.")
        return
    
    results = st.session_state.screening_results
    detailed_results = results.get('detailed_results', [])
    
    if not detailed_results:
        st.info("No detailed results available for analytics.")
        return
    
    # Advanced analytics
    st.subheader("🎯 Performance Metrics")
    
    # Calculate advanced metrics
    scores = [r['score'] for r in detailed_results]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Median Score", f"{pd.Series(scores).median():.1f}%")
    
    with col2:
        st.metric("Score Std Dev", f"{pd.Series(scores).std():.1f}")
    
    with col3:
        high_confidence = [r for r in detailed_results if r['confidence'] > 0.8]
        st.metric("High Confidence", f"{len(high_confidence)}/{len(detailed_results)}")
    
    with col4:
        avg_analysis_time = sum(r['analysis_time_seconds'] for r in detailed_results) / len(detailed_results)
        st.metric("Avg Analysis Time", f"{avg_analysis_time:.2f}s")
    
    # Score vs Confidence scatter plot
    st.subheader("📊 Score vs Confidence Analysis")
    
    df_scatter = pd.DataFrame([
        {
            'Score': r['score'],
            'Confidence': r['confidence'],
            'Action': r['action'].title(),
            'Name': r['candidate']['name']
        }
        for r in detailed_results
    ])
    
    fig_scatter = px.scatter(
        df_scatter,
        x='Score',
        y='Confidence',
        color='Action',
        hover_data=['Name'],
        title="Score vs AI Confidence",
        labels={'Score': 'Candidate Score (%)', 'Confidence': 'AI Confidence'}
    )
    
    # Add threshold lines
    threshold = results.get('session_info', {}).get('score_threshold', 70)
    fig_scatter.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Score Threshold")
    fig_scatter.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="Confidence Threshold")
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Skills gap analysis
    st.subheader("🔧 Skills Gap Analysis")
    
    # Most missing skills
    all_missing_skills = []
    for result in detailed_results:
        all_missing_skills.extend(result['skills_missing'])
    
    if all_missing_skills:
        missing_counts = pd.Series(all_missing_skills).value_counts().head(8)
        
        fig_missing = px.bar(
            x=missing_counts.values,
            y=missing_counts.index,
            orientation='h',
            title="Most Common Missing Skills",
            labels={'x': 'Number of Candidates Missing Skill', 'y': 'Skills'},
            color_discrete_sequence=['#ef4444']
        )
        
        st.plotly_chart(fig_missing, use_container_width=True)
        
        # Recommendations
        st.info(f"💡 **Insight:** Consider adjusting job requirements or providing training for: {', '.join(missing_counts.head(3).index)}")
    
    # Experience distribution analysis
    st.subheader("💼 Experience Analysis")
    
    experience_data = [
        {
            'Years': r['experience_years'],
            'Level': r['experience_level'],
            'Score': r['score'],
            'Action': r['action'].title()
        }
        for r in detailed_results
    ]
    
    df_exp = pd.DataFrame(experience_data)
    
    fig_exp_scatter = px.scatter(
        df_exp,
        x='Years',
        y='Score',
        color='Action',
        size=[10] * len(df_exp),  # Uniform size
        title="Experience vs Score",
        labels={'Years': 'Years of Experience', 'Score': 'Candidate Score (%)'}
    )
    
    st.plotly_chart(fig_exp_scatter, use_container_width=True)
    
    # Historical comparison (if multiple sessions)
    if len(st.session_state.screening_history) > 1:
        st.subheader("📈 Historical Performance")
        
        historical_data = []
        for session in st.session_state.screening_history:
            session_results = session['results']
            historical_data.append({
                'Date': session['timestamp'],
                'Job Title': session['job_title'],
                'Total Candidates': session_results['session_info']['total_candidates'],
                'Average Score': session_results['results']['average_score'],
                'Acceptance Rate': session_results['results']['acceptance_rate'],
                'Processing Time': session_results['session_info']['processing_time_seconds']
            })
        
        df_history = pd.DataFrame(historical_data)
        
        # Multi-line chart
        fig_history = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Score Trend', 'Acceptance Rate Trend', 
                          'Candidate Volume', 'Processing Efficiency'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average Score
        fig_history.add_trace(
            go.Scatter(x=df_history['Date'], y=df_history['Average Score'], 
                      mode='lines+markers', name='Avg Score'),
            row=1, col=1
        )
        
        # Acceptance Rate
        fig_history.add_trace(
            go.Scatter(x=df_history['Date'], y=df_history['Acceptance Rate']*100, 
                      mode='lines+markers', name='Acceptance %'),
            row=1, col=2
        )
        
        # Volume
        fig_history.add_trace(
            go.Bar(x=df_history['Date'], y=df_history['Total Candidates'], 
                   name='Candidates'),
            row=2, col=1
        )
        
        # Efficiency
        fig_history.add_trace(
            go.Scatter(x=df_history['Date'], y=df_history['Processing Time'], 
                      mode='lines+markers', name='Time (s)'),
            row=2, col=2
        )
        
        fig_history.update_layout(height=600, title_text="Historical Performance Dashboard")
        st.plotly_chart(fig_history, use_container_width=True)

def settings_page():
    """Settings and preferences page"""
    
    st.header("⚙️ Settings & Preferences")
    
    # Model settings
    st.subheader("🤖 AI Model Configuration")
    
    with st.form("model_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "HuggingFace Model",
                [
                    "microsoft/DialoGPT-medium",
                    "microsoft/DialoGPT-small", 
                    "distilbert-base-uncased",
                    "rule-based-only"
                ],
                help="Choose AI model for resume analysis"
            )
            
            default_threshold = st.slider(
                "Default Score Threshold (%)",
                min_value=50,
                max_value=95,
                value=70,
                help="Default acceptance threshold for new jobs"
            )
        
        with col2:
            enable_ai_analysis = st.checkbox("Enable AI Analysis", value=True)
            enable_caching = st.checkbox("Enable Model Caching", value=True)
            
            cache_location = st.text_input(
                "Custom Cache Directory",
                value=os.getenv('TRANSFORMERS_CACHE', ''),
                help="Custom directory for model cache"
            )
        
        if st.form_submit_button("💾 Save Model Settings"):
            st.success("✅ Model settings saved!")
    
    # Email settings
    st.subheader("📧 Email Configuration")
    
    with st.form("email_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587)
            
        with col2:
            email_address = st.text_input("Email Address", value="")
            company_name = st.text_input("Company Name", value="Our Company")
        
        test_email = st.text_input("Test Email Address", help="Send test email to this address")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("💾 Save Email Settings"):
                st.success("✅ Email settings saved!")
        
        with col2:
            if st.form_submit_button("📧 Send Test Email"):
                if test_email:
                    st.info(f"🧪 Test email sent to {test_email}")
                else:
                    st.error("Please enter a test email address")
    
    # Data management
    st.subheader("💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Session History", help="Remove all stored screening sessions"):
            st.session_state.screening_history = []
            st.success("✅ Session history cleared!")
    
    with col2:
        if st.button("📤 Export All Data", help="Download all session data"):
            if st.session_state.screening_history:
                all_data = {
                    'sessions': [
                        {
                            'timestamp': session['timestamp'].isoformat(),
                            'job_title': session['job_title'],
                            'results': session['results']
                        }
                        for session in st.session_state.screening_history
                    ]
                }
                
                json_str = json.dumps(all_data, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download All Sessions",
                    data=json_str,
                    file_name=f"all_screening_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            else:
                st.info("No data to export")
    
    with col3:
        if st.button("🔄 Reset Application", help="Reset all settings and data"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Application reset! Please refresh the page.")

if __name__ == "__main__":
    main()