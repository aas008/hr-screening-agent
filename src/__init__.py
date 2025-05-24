"""
Autonomous HR Screening Agent
AI-powered resume screening and candidate management system
"""

__version__ = "1.0.0"
__author__ = "HR Screening Agent Team" 
__email__ = "support@hr-screening-agent.com"

# Package metadata
DESCRIPTION = "Autonomous AI agent for screening job applications"
LONG_DESCRIPTION = """
An intelligent HR screening system that automatically processes resumes, 
analyzes candidates against job requirements, and sends personalized 
email responses using AI and workflow automation.
"""

# Core components
from .agents.github_loader import GitHubResumeLoader, CandidateInfo
from .agents.resume_analyzer import HuggingFaceResumeAnalyzer, AnalysisResult  
from .agents.email_sender import EmailSender, EmailResult

__all__ = [
    'GitHubResumeLoader',
    'CandidateInfo', 
    'HuggingFaceResumeAnalyzer',
    'AnalysisResult',
    'EmailSender',
    'EmailResult'
]