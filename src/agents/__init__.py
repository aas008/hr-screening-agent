"""
HR Screening Agents
Core agent components for resume processing, analysis, and communication
"""

from .github_loader import GitHubResumeLoader, CandidateInfo, validate_github_config
from .resume_analyzer import HuggingFaceResumeAnalyzer, AnalysisResult
from .email_sender import EmailSender, EmailResult

__all__ = [
    'GitHubResumeLoader',
    'CandidateInfo',
    'validate_github_config',
    'HuggingFaceResumeAnalyzer', 
    'AnalysisResult',
    'EmailSender',
    'EmailResult'
]