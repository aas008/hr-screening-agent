"""
Email Sender for HR Screening Agent
Handles automated email responses to candidates
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from .github_loader import CandidateInfo
from .resume_analyzer import AnalysisResult

@dataclass
class EmailResult:
    """Result of email sending attempt"""
    candidate_name: str
    candidate_email: str
    email_type: str
    success: bool
    error_message: Optional[str] = None
    sent_timestamp: Optional[str] = None

class EmailSender:
    """Send automated emails to candidates based on screening results"""
    
    def __init__(self, config: Dict):
        """Initialize email sender with SMTP configuration"""
        
        self.enabled = config.get('email_enabled', False)
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.email_address = config.get('email_address')
        self.email_password = config.get('email_password')
        
        # Load email templates
        self.templates = self._load_email_templates()
        
        # Validate configuration
        if self.enabled:
            if not all([self.email_address, self.email_password]):
                print("⚠️  Warning: Email enabled but missing credentials")
                self.enabled = False
            else:
                print(f"📧 Email sender initialized: {self.email_address}")
        else:
            print("📧 Email sender initialized in simulation mode")
    
    def _load_email_templates(self) -> Dict[str, str]:
        """Load email templates from files"""
        
        templates = {}
        template_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'templates')
        
        template_files = {
            'acceptance': 'acceptance_email.txt',
            'rejection': 'rejection_email.txt',
            'info_request': 'info_request_email.txt'
        }
        
        for template_name, filename in template_files.items():
            file_path = os.path.join(template_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    templates[template_name] = f.read().strip()
                print(f"   ✅ Loaded template: {template_name}")
            except FileNotFoundError:
                # Use default template if file not found
                templates[template_name] = self._get_default_template(template_name)
                print(f"   ⚠️  Using default template for: {template_name}")
            except Exception as e:
                print(f"   ❌ Error loading template {template_name}: {e}")
                templates[template_name] = self._get_default_template(template_name)
        
        return templates
    
    def _get_default_template(self, template_type: str) -> str:
        """Get default email template if file is missing"""
        
        defaults = {
            'acceptance': """Dear {candidate_name},

Thank you for your application for the {job_title} position. We're impressed with your background and would like to move forward with the next step in our hiring process.

We will be in touch soon to schedule a phone screening to discuss your experience and the role in more detail.

We're excited about the possibility of you joining our team!

Best regards,
{company_name} HR Team""",
            
            'rejection': """Dear {candidate_name},

Thank you for your interest in the {job_title} position and for taking the time to submit your application.

After careful review of your qualifications, we have decided to move forward with other candidates whose experience more closely aligns with our current requirements.

We appreciate your interest in {company_name} and encourage you to apply for future opportunities that may be a better match for your background and career goals.

We wish you the best of luck in your job search.

Best regards,
{company_name} HR Team""",
            
            'info_request': """Dear {candidate_name},

Thank you for your application for the {job_title} position. We're interested in learning more about your background and experience.

Could you please provide additional information about:
{info_requests}

Please reply to this email with the requested details, and we'll continue reviewing your application.

Thank you for your time and interest in {company_name}.

Best regards,
{company_name} HR Team"""
        }
        
        return defaults.get(template_type, "Template not available")
    
    def send_screening_emails(self, results: List[AnalysisResult], job_title: str, 
                            company_name: str = "Our Company") -> Dict[str, any]:
        """Send emails to all candidates based on screening results"""
        
        print(f"\n📧 SENDING SCREENING EMAILS")
        print("-" * 40)
        
        email_results = []
        stats = {
            'total': len(results),
            'sent': 0,
            'failed': 0,
            'simulated': 0,
            'by_type': {'acceptance': 0, 'rejection': 0, 'info_request': 0}
        }
        
        for result in results:
            try:
                # Determine email type based on analysis result
                if result.action == 'accept':
                    email_type = 'acceptance'
                elif result.action == 'reject':
                    email_type = 'rejection'
                elif result.action == 'request_info':
                    email_type = 'info_request'
                else:
                    # Skip manual review cases
                    continue
                
                # Send email
                email_result = self._send_single_email(
                    result.candidate, 
                    email_type, 
                    job_title, 
                    company_name,
                    result
                )
                
                email_results.append(email_result)
                
                # Update statistics
                if email_result.success:
                    if self.enabled:
                        stats['sent'] += 1
                    else:
                        stats['simulated'] += 1
                    stats['by_type'][email_type] += 1
                else:
                    stats['failed'] += 1
                
            except Exception as e:
                print(f"   ❌ Unexpected error for {result.candidate.name}: {e}")
                stats['failed'] += 1
        
        # Print summary
        print(f"\n📊 EMAIL SUMMARY:")
        if self.enabled:
            print(f"   ✅ Sent: {stats['sent']}")
        else:
            print(f"   🧪 Simulated: {stats['simulated']}")
        print(f"   ❌ Failed: {stats['failed']}")
        print(f"   📈 By Type: {stats['by_type']}")
        
        return {
            'results': email_results,
            'statistics': stats,
            'mode': 'real' if self.enabled else 'simulation'
        }
    
    def _send_single_email(self, candidate: CandidateInfo, email_type: str, 
                          job_title: str, company_name: str, 
                          analysis_result: AnalysisResult) -> EmailResult:
        """Send a single email to a candidate"""
        
        try:
            # Prepare email content
            subject = self._create_subject(email_type, job_title, candidate.name)
            message = self._create_message(email_type, candidate, job_title, company_name, analysis_result)
            
            if self.enabled:
                # Send real email
                success = self._send_smtp_email(candidate.email, subject, message)
                status = "✅ SENT" if success else "❌ FAILED"
            else:
                # Simulation mode
                success = True
                status = "🧪 SIMULATED"
            
            print(f"   {status}: {email_type} → {candidate.name} ({candidate.email})")
            
            return EmailResult(
                candidate_name=candidate.name,
                candidate_email=candidate.email,
                email_type=email_type,
                success=success,
                sent_timestamp=datetime.now().isoformat() if success else None
            )
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ ERROR: {email_type} → {candidate.name}: {error_msg}")
            
            return EmailResult(
                candidate_name=candidate.name,
                candidate_email=candidate.email,
                email_type=email_type,
                success=False,
                error_message=error_msg
            )
    
    def _create_subject(self, email_type: str, job_title: str, candidate_name: str) -> str:
        """Create email subject line"""
        
        subjects = {
            'acceptance': f"Next Steps - {job_title} Position",
            'rejection': f"Application Update - {job_title} Position", 
            'info_request': f"Additional Information Needed - {job_title} Position"
        }
        
        return subjects.get(email_type, f"Regarding Your Application - {job_title}")
    
    def _create_message(self, email_type: str, candidate: CandidateInfo, 
                       job_title: str, company_name: str, 
                       analysis_result: AnalysisResult) -> str:
        """Create personalized email message"""
        
        template = self.templates.get(email_type, "Template not available")
        
        # Prepare template variables
        template_vars = {
            'candidate_name': candidate.name,
            'job_title': job_title,
            'company_name': company_name,
            'candidate_email': candidate.email
        }
        
        # Add specific variables for info request emails
        if email_type == 'info_request':
            info_requests = self._generate_info_requests(analysis_result)
            template_vars['info_requests'] = info_requests
        
        # Replace template variables
        try:
            message = template.format(**template_vars)
        except KeyError as e:
            print(f"   ⚠️  Template variable missing: {e}")
            message = template  # Use template as-is if formatting fails
        
        return message
    
    def _generate_info_requests(self, analysis_result: AnalysisResult) -> str:
        """Generate specific information requests based on analysis"""
        
        requests = []
        
        # Request clarification on missing skills
        if analysis_result.skills_missing:
            missing_skills = ', '.join(analysis_result.skills_missing[:3])
            requests.append(f"• Your experience with: {missing_skills}")
        
        # Request more experience details if below threshold
        if analysis_result.experience_years < 2:
            requests.append("• More details about your professional experience and projects")
        
        # Request portfolio/examples
        if any(skill in ['react', 'javascript', 'frontend'] for skill in [s.lower() for s in analysis_result.skills_found]):
            requests.append("• Links to your portfolio, GitHub profile, or relevant project examples")
        
        # Default request if no specific items identified
        if not requests:
            requests.append("• Additional details about your relevant experience and qualifications")
        
        return '\n'.join(requests)
    
    def _send_smtp_email(self, to_email: str, subject: str, message: str) -> bool:
        """Send email via SMTP"""
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_address
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to server and send
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()  # Enable encryption
            server.login(self.email_address, self.email_password)
            
            # Send email
            text = msg.as_string()
            server.sendmail(self.email_address, to_email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"   ❌ SMTP Error: {e}")
            return False
    
    def send_summary_email(self, admin_email: str, session_summary: Dict) -> bool:
        """Send session summary to HR admin"""
        
        if not self.enabled:
            print("📧 Summary email simulation mode - would send to admin")
            return True
        
        try:
            subject = f"HR Screening Session Complete - {session_summary['session_info']['job_title']}"
            
            message = self._create_summary_message(session_summary)
            
            success = self._send_smtp_email(admin_email, subject, message)
            
            if success:
                print(f"📧 Summary email sent to admin: {admin_email}")
            else:
                print(f"❌ Failed to send summary email to admin")
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending summary email: {e}")
            return False
    
    def _create_summary_message(self, session_summary: Dict) -> str:
        """Create summary email message for admin"""
        
        session_info = session_summary['session_info']
        results = session_summary['results']
        efficiency = session_summary['efficiency_metrics']
        
        message = f"""HR Screening Session Summary

Job: {session_info['job_title']}
Date: {session_info['timestamp'][:19]}
Processing Time: {session_info['processing_time_seconds']:.1f} seconds

RESULTS:
• Total Candidates: {session_info['total_candidates']}
• Accepted: {results['accepted']}
• Rejected: {results['rejected']}
• Acceptance Rate: {results['acceptance_rate']:.1%}
• Average Score: {results['average_score']}%

EFFICIENCY:
• Time Saved: {efficiency['time_saved_minutes']:.0f} minutes
• Automation Rate: {efficiency['automation_rate']:.1%}

The detailed results have been saved to your outputs folder.

Automated HR Screening Agent
"""
        return message

def test_email_sending():
    """Test function for email sending (development use)"""
    
    # Sample test data
    from github_loader import CandidateInfo
    from resume_analyzer import AnalysisResult
    
    candidate = CandidateInfo(
        name="John Smith",
        email="john.smith@example.com",
        phone="(555) 123-4567",
        resume_text="Sample resume text...",
        file_name="john_smith.pdf",
        application_date=datetime.now().isoformat()
    )
    
    analysis_result = AnalysisResult(
        candidate=candidate,
        score=85.0,
        skills_found=['React', 'JavaScript', 'TypeScript'],
        skills_missing=['Node.js'],
        experience_years=4,
        experience_level="Mid-level",
        strengths=['Strong technical skills', 'Good experience'],
        concerns=[],
        action='accept',
        reasoning="Strong candidate with excellent skills match",
        confidence=0.9,
        analysis_time_seconds=2.5
    )
    
    # Test email sender in simulation mode
    config = {
        'email_enabled': False,  # Simulation mode
        'email_address': 'hr@company.com'
    }
    
    email_sender = EmailSender(config)
    
    # Test sending emails
    results = [analysis_result]
    email_results = email_sender.send_screening_emails(
        results, 
        "Senior React Developer", 
        "TechCorp Inc"
    )
    
    print(f"\n📊 EMAIL TEST RESULTS:")
    print(f"Mode: {email_results['mode']}")
    print(f"Statistics: {email_results['statistics']}")
    
    for result in email_results['results']:
        print(f"• {result.candidate_name}: {result.email_type} - {'Success' if result.success else 'Failed'}")

if __name__ == "__main__":
    test_email_sending()