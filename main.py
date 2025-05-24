#!/usr/bin/env python3
"""
Autonomous HR Screening Agent - Main Entry Point
Automatically screens job applications using AI and GitHub integration
"""

import os
import sys
import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from workflows.langgraph_workflow import run_autonomous_screening
from agents.github_loader import validate_github_config

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    
    # Validate required environment variables
    required_vars = [
        'GITHUB_TOKEN',
        'GITHUB_REPO_OWNER', 
        'GITHUB_REPO_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("💡 Please copy .env.example to .env and fill in your values")
        sys.exit(1)
    
    return {
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

def create_job_requirements(args):
    """Create job requirements from command line arguments or interactive input"""
    
    if args.interactive:
        print("\n📋 JOB REQUIREMENTS SETUP")
        print("-" * 30)
        
        job_title = input("Job Title: ").strip()
        
        print("Required Skills (comma-separated):")
        required_skills = [skill.strip() for skill in input().split(',') if skill.strip()]
        
        print("Preferred Skills (comma-separated, optional):")
        preferred_input = input().strip()
        preferred_skills = [skill.strip() for skill in preferred_input.split(',') if skill.strip()] if preferred_input else []
        
        min_experience = int(input("Minimum Years of Experience: ").strip() or "0")
        department = input("Department (optional): ").strip() or "Unknown"
        
    else:
        # Use command line arguments with defaults
        job_title = args.job_title or f"Senior {args.job_role.replace('-', ' ').title()} Developer"
        
        # Default skills based on job role
        skill_defaults = {
            'react-developer': ['React', 'JavaScript', 'TypeScript', 'HTML', 'CSS'],
            'python-developer': ['Python', 'Django', 'Flask', 'SQL', 'API'],
            'data-scientist': ['Python', 'SQL', 'Machine Learning', 'Statistics', 'Pandas'],
            'fullstack-developer': ['JavaScript', 'Python', 'React', 'Node.js', 'SQL']
        }
        
        required_skills = args.required_skills or skill_defaults.get(args.job_role, ['Programming'])
        preferred_skills = args.preferred_skills or []
        min_experience = args.min_experience
        department = args.department
    
    return {
        'title': job_title,
        'required_skills': required_skills,
        'preferred_skills': preferred_skills,
        'min_experience_years': min_experience,
        'department': department
    }

def save_results(results, config):
    """Save screening results to file"""
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(config['output_dir'], 'screening_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_title_clean = results['session_info']['job_title'].lower().replace(' ', '_')
    filename = f"{job_title_clean}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {filepath}")
    return filepath

def print_summary(results):
    """Print session summary to console"""
    
    print(f"\n📊 SCREENING SESSION SUMMARY")
    print("=" * 50)
    
    session_info = results['session_info']
    results_data = results['results']
    efficiency = results['efficiency_metrics']
    
    print(f"🎯 Job: {session_info['job_title']}")
    print(f"📅 Date: {session_info['timestamp'][:19]}")
    print(f"⏱️  Processing Time: {session_info['processing_time_seconds']:.1f} seconds")
    
    print(f"\n📈 RESULTS:")
    print(f"   Total Candidates: {session_info['total_candidates']}")
    print(f"   ✅ Accepted: {results_data['accepted']}")
    print(f"   ❌ Rejected: {results_data['rejected']}")
    print(f"   📊 Acceptance Rate: {results_data['acceptance_rate']:.1%}")
    print(f"   🎯 Average Score: {results_data['average_score']}%")
    
    print(f"\n⚡ EFFICIENCY:")
    print(f"   Time Saved: {efficiency['time_saved_minutes']:.0f} minutes")
    print(f"   Automation Rate: {efficiency['automation_rate']:.1%}")
    
    if results.get('email_actions'):
        email_info = results['email_actions']
        print(f"   📧 Emails Sent: {email_info['sent']}/{email_info['total']}")
    
    # Show score distribution
    if results_data.get('score_distribution'):
        print(f"\n📊 SCORE DISTRIBUTION:")
        for range_name, count in results_data['score_distribution'].items():
            if count > 0:
                print(f"   {range_name}: {count} candidates")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Autonomous HR Screening Agent')
    
    # Required arguments
    parser.add_argument('--job-role', required=True, 
                       help='Job role folder name (e.g., react-developer)')
    
    # Optional job configuration
    parser.add_argument('--job-title', 
                       help='Job title (default: auto-generated from job-role)')
    parser.add_argument('--required-skills', nargs='+',
                       help='Required skills list')
    parser.add_argument('--preferred-skills', nargs='+', default=[],
                       help='Preferred skills list')
    parser.add_argument('--min-experience', type=int, default=2,
                       help='Minimum years of experience (default: 2)')
    parser.add_argument('--department', default='Engineering',
                       help='Department name (default: Engineering)')
    
    # Execution options
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode for job requirements setup')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without sending emails (simulation mode)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        print("🚀 AUTONOMOUS HR SCREENING AGENT")
        print("=" * 40)
        
        # Load configuration
        print("⚙️  Loading configuration...")
        config = load_environment()
        
        # Validate GitHub access
        print("🔍 Validating GitHub access...")
        if not validate_github_config(config):
            print("❌ GitHub configuration validation failed")
            sys.exit(1)
        
        # Create job requirements
        print("📋 Setting up job requirements...")
        job_requirements = create_job_requirements(args)
        
        # Override email settings for dry run
        if args.dry_run:
            config['email_enabled'] = False
            print("🧪 Running in dry-run mode (no emails will be sent)")
        
        print(f"\n🎯 Job: {job_requirements['title']}")
        print(f"📂 Folder: resumes/active/{args.job_role}/")
        print(f"📧 Email sending: {'Enabled' if config['email_enabled'] else 'Disabled'}")
        
        # Run the autonomous screening workflow
        print(f"\n🤖 Starting autonomous screening...")
        results = run_autonomous_screening(
            job_requirements=job_requirements,
            job_role_folder=args.job_role,
            config=config,
            verbose=args.verbose
        )
        
        # Handle errors
        if 'error' in results:
            print(f"\n❌ ERROR: {results['error']}")
            if 'message' in results:
                print(f"💡 {results['message']}")
            sys.exit(1)
        
        # Print summary
        print_summary(results)
        
        # Save results
        saved_path = save_results(results, config)
        
        # Success message
        print(f"\n🎉 Screening completed successfully!")
        print(f"📄 Full report: {saved_path}")
        
        # Show next steps
        accepted_count = results['results']['accepted']
        if accepted_count > 0:
            print(f"\n📅 NEXT STEPS:")
            print(f"   • Schedule interviews for {accepted_count} accepted candidates")
            print(f"   • Review detailed candidate analysis in the report")
            if config['email_enabled']:
                print(f"   • Follow up on sent emails")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Screening interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()