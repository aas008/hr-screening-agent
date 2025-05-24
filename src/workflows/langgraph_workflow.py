"""
LangGraph Workflow for Autonomous HR Screening
Orchestrates the complete screening process using LangGraph state management
"""

import sys
import os
from typing import Dict, List, Any, TypedDict
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langgraph.graph import StateGraph, END
from agents.github_loader import GitHubResumeLoader, CandidateInfo
from agents.resume_analyzer import HuggingFaceResumeAnalyzer, AnalysisResult
from agents.email_sender import EmailSender

# LangGraph State Definition
class HRScreeningState(TypedDict):
    # Configuration
    config: Dict[str, Any]
    job_requirements: Dict[str, Any]
    job_role_folder: str
    
    # Workflow Data
    candidates: List[Dict[str, Any]]
    analysis_results: List[Dict[str, Any]]
    email_results: Dict[str, Any]
    
    # Error Handling
    errors: List[str]
    human_interventions: List[Dict[str, Any]]
    
    # Session Tracking
    session_start_time: str
    current_step: str
    step_results: Dict[str, Any]
    
    # Final Output
    session_summary: Dict[str, Any]

class HRScreeningWorkflow:
    """LangGraph workflow for autonomous HR screening"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.github_loader = GitHubResumeLoader(
            config['github_token'],
            config['repo_owner'],
            config['repo_name']
        )
        
        self.resume_analyzer = HuggingFaceResumeAnalyzer(
            model_name=config.get('huggingface_model', 'microsoft/DialoGPT-medium'),
            score_threshold=config.get('score_threshold', 70)
        )
        
        self.email_sender = EmailSender(config)
        
        # Create workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        workflow = StateGraph(HRScreeningState)
        
        # Add workflow nodes
        workflow.add_node("initialize_session", self._initialize_session_node)
        workflow.add_node("load_resumes", self._load_resumes_node)
        workflow.add_node("analyze_candidates", self._analyze_candidates_node)
        workflow.add_node("send_emails", self._send_emails_node)
        workflow.add_node("generate_summary", self._generate_summary_node)
        workflow.add_node("handle_errors", self._handle_errors_node)
        
        # Define workflow edges
        workflow.add_edge("initialize_session", "load_resumes")
        workflow.add_edge("load_resumes", "analyze_candidates")
        workflow.add_edge("analyze_candidates", "send_emails")
        workflow.add_edge("send_emails", "generate_summary")
        workflow.add_edge("generate_summary", END)
        workflow.add_edge("handle_errors", END)
        
        # Set entry point
        workflow.set_entry_point("initialize_session")
        
        return workflow.compile()
    
    def run_screening(self, job_requirements: Dict, job_role_folder: str, verbose: bool = False) -> Dict:
        """Run the complete screening workflow"""
        
        # Initialize state
        initial_state = HRScreeningState(
            config=self.config,
            job_requirements=job_requirements,
            job_role_folder=job_role_folder,
            candidates=[],
            analysis_results=[],
            email_results={},
            errors=[],
            human_interventions=[],
            session_start_time=datetime.now().isoformat(),
            current_step="initializing",
            step_results={},
            session_summary={}
        )
        
        try:
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Return results or error
            if final_state.get('errors'):
                return {
                    'error': 'Workflow completed with errors',
                    'errors': final_state['errors'],
                    'partial_results': final_state.get('session_summary', {})
                }
            else:
                return final_state['session_summary']
                
        except Exception as e:
            return {
                'error': 'Workflow execution failed',
                'message': str(e),
                'step': initial_state.get('current_step', 'unknown')
            }
    
    def _initialize_session_node(self, state: HRScreeningState) -> HRScreeningState:
        """Initialize the screening session"""
        
        print("🚀 INITIALIZING AUTONOMOUS HR SCREENING SESSION")
        print("=" * 60)
        
        state["current_step"] = "initialization"
        state["errors"] = []
        state["human_interventions"] = []
        
        # Log session details
        job_req = state["job_requirements"]
        print(f"🎯 Job: {job_req['title']}")
        print(f"📂 Folder: resumes/active/{state['job_role_folder']}/")
        print(f"🔧 Required Skills: {', '.join(job_req['required_skills'])}")
        print(f"📅 Min Experience: {job_req['min_experience_years']} years")
        print(f"⚡ Score Threshold: {self.config.get('score_threshold', 70)}%")
        
        state["step_results"] = {
            "status": "success",
            "message": "Session initialized successfully"
        }
        
        return state
    
    def _load_resumes_node(self, state: HRScreeningState) -> HRScreeningState:
        """Load resumes from GitHub repository"""
        
        print(f"\n📂 LOADING RESUMES FROM GITHUB")
        print("-" * 40)
        
        state["current_step"] = "loading_resumes"
        
        try:
            # Load candidates from GitHub
            candidates, errors = self.github_loader.load_resumes_from_job_role(
                state["job_role_folder"]
            )
            
            if not candidates and not errors:
                error_msg = f"No resumes found in /resumes/active/{state['job_role_folder']}/"
                state["errors"].append(error_msg)
                state["step_results"] = {
                    "status": "error",
                    "message": error_msg
                }
                return state
            
            # Convert candidates to dict format for JSON serialization
            candidates_dict = []
            for candidate in candidates:
                candidates_dict.append({
                    "name": candidate.name,
                    "email": candidate.email,
                    "phone": candidate.phone,
                    "resume_text": candidate.resume_text,
                    "file_name": candidate.file_name,
                    "application_date": candidate.application_date,
                    "raw_file_size": getattr(candidate, 'raw_file_size', 0)
                })
            
            state["candidates"] = candidates_dict
            
            # Handle any loading errors
            if errors:
                state["errors"].extend(errors)
                print(f"⚠️  {len(errors)} files had processing errors")
            
            state["step_results"] = {
                "status": "success",
                "candidates_loaded": len(candidates),
                "errors_encountered": len(errors)
            }
            
            print(f"✅ Successfully loaded {len(candidates)} candidates")
            
        except Exception as e:
            error_msg = f"Failed to load resumes: {str(e)}"
            state["errors"].append(error_msg)
            state["step_results"] = {
                "status": "error", 
                "message": error_msg
            }
            print(f"❌ {error_msg}")
        
        return state
    
    def _analyze_candidates_node(self, state: HRScreeningState) -> HRScreeningState:
        """Analyze all candidates using AI"""
        
        print(f"\n🤖 ANALYZING CANDIDATES WITH AI")
        print("-" * 40)
        
        state["current_step"] = "analyzing_candidates"
        
        try:
            candidates_data = state["candidates"]
            job_requirements = state["job_requirements"]
            
            if not candidates_data:
                error_msg = "No candidates to analyze"
                state["errors"].append(error_msg)
                return state
            
            analysis_results = []
            accepted_count = 0
            rejected_count = 0
            
            for i, candidate_data in enumerate(candidates_data, 1):
                print(f"   🔍 Analyzing {i}/{len(candidates_data)}: {candidate_data['name']}")
                
                try:
                    # Convert dict back to CandidateInfo object
                    candidate = CandidateInfo(
                        name=candidate_data["name"],
                        email=candidate_data["email"],
                        phone=candidate_data["phone"],
                        resume_text=candidate_data["resume_text"],
                        file_name=candidate_data["file_name"],
                        application_date=candidate_data["application_date"]
                    )
                    
                    # Analyze candidate
                    analysis_result = self.resume_analyzer.analyze_resume(candidate, job_requirements)
                    
                    # Convert to dict for JSON serialization
                    analysis_dict = {
                        "candidate": candidate_data,
                        "score": analysis_result.score,
                        "skills_found": analysis_result.skills_found,
                        "skills_missing": analysis_result.skills_missing,
                        "experience_years": analysis_result.experience_years,
                        "experience_level": analysis_result.experience_level,
                        "strengths": analysis_result.strengths,
                        "concerns": analysis_result.concerns,
                        "action": analysis_result.action,
                        "reasoning": analysis_result.reasoning,
                        "confidence": analysis_result.confidence,
                        "analysis_time_seconds": analysis_result.analysis_time_seconds
                    }
                    
                    analysis_results.append(analysis_dict)
                    
                    # Count actions
                    if analysis_result.action == "accept":
                        accepted_count += 1
                    else:
                        rejected_count += 1
                    
                except Exception as e:
                    error_msg = f"Analysis failed for {candidate_data['name']}: {str(e)}"
                    state["errors"].append(error_msg)
                    
                    # Flag for human intervention
                    state["human_interventions"].append({
                        "candidate_name": candidate_data['name'],
                        "issue": "Analysis failed",
                        "error": str(e),
                        "action_needed": "Manual review required"
                    })
                    
                    print(f"   ❌ {error_msg}")
            
            state["analysis_results"] = analysis_results
            
            state["step_results"] = {
                "status": "success",
                "total_analyzed": len(analysis_results),
                "accepted": accepted_count,
                "rejected": rejected_count,
                "errors": len(state["errors"])
            }
            
            print(f"✅ Analysis complete: {accepted_count} accept, {rejected_count} reject")
            
        except Exception as e:
            error_msg = f"Candidate analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            state["step_results"] = {
                "status": "error",
                "message": error_msg
            }
            print(f"❌ {error_msg}")
        
        return state
    
    def _send_emails_node(self, state: HRScreeningState) -> HRScreeningState:
        """Send automated emails to candidates"""
        
        print(f"\n📧 SENDING AUTOMATED EMAILS")
        print("-" * 40)
        
        state["current_step"] = "sending_emails"
        
        try:
            analysis_results = state["analysis_results"]
            job_title = state["job_requirements"]["title"]
            
            if not analysis_results:
                print("⚠️  No analysis results to process for emails")
                state["email_results"] = {"sent": 0, "total": 0, "mode": "none"}
                return state
            
            # Convert analysis results back to AnalysisResult objects
            analysis_objects = []
            for result_data in analysis_results:
                # Convert candidate data back to CandidateInfo
                candidate_data = result_data["candidate"]
                candidate = CandidateInfo(
                    name=candidate_data["name"],
                    email=candidate_data["email"],
                    phone=candidate_data["phone"],
                    resume_text=candidate_data["resume_text"],
                    file_name=candidate_data["file_name"],
                    application_date=candidate_data["application_date"]
                )
                
                # Create AnalysisResult object
                analysis_result = AnalysisResult(
                    candidate=candidate,
                    score=result_data["score"],
                    skills_found=result_data["skills_found"],
                    skills_missing=result_data["skills_missing"],
                    experience_years=result_data["experience_years"],
                    experience_level=result_data["experience_level"],
                    strengths=result_data["strengths"],
                    concerns=result_data["concerns"],
                    action=result_data["action"],
                    reasoning=result_data["reasoning"],
                    confidence=result_data["confidence"],
                    analysis_time_seconds=result_data["analysis_time_seconds"]
                )
                
                analysis_objects.append(analysis_result)
            
            # Send emails
            email_results = self.email_sender.send_screening_emails(
                analysis_objects,
                job_title,
                "Our Company"  # Default company name
            )
            
            state["email_results"] = email_results
            
            state["step_results"] = {
                "status": "success",
                "emails_processed": email_results["statistics"]["total"],
                "emails_sent": email_results["statistics"]["sent"],
                "email_mode": email_results["mode"]
            }
            
        except Exception as e:
            error_msg = f"Email sending failed: {str(e)}"
            state["errors"].append(error_msg)
            state["step_results"] = {
                "status": "error",
                "message": error_msg
            }
            print(f"❌ {error_msg}")
        
        return state
    
    def _generate_summary_node(self, state: HRScreeningState) -> HRScreeningState:
        """Generate final session summary"""
        
        print(f"\n📊 GENERATING SESSION SUMMARY")
        print("-" * 40)
        
        state["current_step"] = "generating_summary"
        
        try:
            # Calculate session timing
            start_time = datetime.fromisoformat(state["session_start_time"])
            end_time = datetime.now()
            session_duration = (end_time - start_time).total_seconds()
            
            # Gather statistics
            candidates_count = len(state["candidates"])
            analysis_results = state["analysis_results"]
            email_results = state["email_results"]
            
            # Calculate results breakdown
            accepted = sum(1 for r in analysis_results if r["action"] == "accept")
            rejected = sum(1 for r in analysis_results if r["action"] == "reject")
            
            # Calculate average score
            if analysis_results:
                avg_score = sum(r["score"] for r in analysis_results) / len(analysis_results)
                score_distribution = self._calculate_score_distribution(analysis_results)
            else:
                avg_score = 0
                score_distribution = {}
            
            # Create comprehensive summary
            session_summary = {
                "session_info": {
                    "job_title": state["job_requirements"]["title"],
                    "job_role_folder": state["job_role_folder"],
                    "timestamp": end_time.isoformat(),
                    "processing_time_seconds": session_duration,
                    "total_candidates": candidates_count
                },
                "results": {
                    "accepted": accepted,
                    "rejected": rejected,
                    "acceptance_rate": accepted / candidates_count if candidates_count > 0 else 0,
                    "average_score": round(avg_score, 1),
                    "score_distribution": score_distribution
                },
                "email_actions": email_results,
                "efficiency_metrics": {
                    "estimated_manual_time_minutes": candidates_count * 15,
                    "actual_processing_time_minutes": session_duration / 60,
                    "time_saved_minutes": (candidates_count * 15) - (session_duration / 60), 
                    "automation_rate": 1.0 if not state["human_interventions"] else 0.8
                },
                "error_summary": {
                    "total_errors": len(state["errors"]),
                    "errors": state["errors"],
                    "human_interventions_needed": len(state["human_interventions"]),
                    "interventions": state["human_interventions"]
                },
                "detailed_results": analysis_results
            }
            
            state["session_summary"] = session_summary
            
            # Print summary
            self._print_final_summary(session_summary)
            
            state["step_results"] = {
                "status": "success",
                "summary_generated": True
            }
            
        except Exception as e:
            error_msg = f"Summary generation failed: {str(e)}"
            state["errors"].append(error_msg)
            state["step_results"] = {
                "status": "error",
                "message": error_msg
            }
            print(f"❌ {error_msg}")
        
        return state
    
    def _handle_errors_node(self, state: HRScreeningState) -> HRScreeningState:
        """Handle workflow errors and provide guidance"""
        
        print(f"\n⚠️  HANDLING WORKFLOW ERRORS")
        print("-" * 40)
        
        errors = state["errors"]
        interventions = state["human_interventions"]
        
        print(f"Total Errors: {len(errors)}")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")
        
        if interventions:
            print(f"\nHuman Interventions Needed: {len(interventions)}")
            for intervention in interventions:
                print(f"   • {intervention['candidate_name']}: {intervention['action_needed']}")
        
        return state
    
    def _calculate_score_distribution(self, analysis_results: List[Dict]) -> Dict[str, int]:
        """Calculate score distribution for summary"""
        
        distribution = {
            "90-100%": 0,
            "80-89%": 0, 
            "70-79%": 0,
            "60-69%": 0,
            "50-59%": 0,
            "Below 50%": 0
        }
        
        for result in analysis_results:
            score = result["score"]
            if score >= 90:
                distribution["90-100%"] += 1
            elif score >= 80:
                distribution["80-89%"] += 1
            elif score >= 70:
                distribution["70-79%"] += 1
            elif score >= 60:
                distribution["60-69%"] += 1
            elif score >= 50:
                distribution["50-59%"] += 1
            else:
                distribution["Below 50%"] += 1
        
        return distribution
    
    def _print_final_summary(self, summary: Dict):
        """Print formatted final summary"""
        
        session_info = summary["session_info"]
        results = summary["results"]
        efficiency = summary["efficiency_metrics"]
        
        print(f"✅ SESSION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"🎯 Job: {session_info['job_title']}")
        print(f"⏱️  Total Time: {session_info['processing_time_seconds']:.1f} seconds")
        print(f"📊 Candidates: {session_info['total_candidates']}")
        print(f"✅ Accepted: {results['accepted']}")
        print(f"❌ Rejected: {results['rejected']}")
        print(f"📈 Acceptance Rate: {results['acceptance_rate']:.1%}")
        print(f"🎯 Average Score: {results['average_score']}%")
        print(f"💰 Time Saved: {efficiency['time_saved_minutes']:.0f} minutes")

# Main function to run the workflow
def run_autonomous_screening(job_requirements: Dict, job_role_folder: str, 
                           config: Dict, verbose: bool = False) -> Dict:
    """
    Main function to run autonomous HR screening workflow
    
    Args:
        job_requirements: Job requirements dictionary
        job_role_folder: Folder name in /resumes/active/
        config: Configuration dictionary with API keys, etc.
        verbose: Enable verbose logging
        
    Returns:
        Dictionary with session results or error information
    """
    
    try:
        # Create and run workflow
        workflow = HRScreeningWorkflow(config)
        results = workflow.run_screening(job_requirements, job_role_folder, verbose)
        
        return results
        
    except Exception as e:
        return {
            'error': 'Workflow initialization failed',
            'message': str(e)
        }

# Test function
def test_workflow():
    """Test the workflow with sample data"""
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Sample configuration
    config = {
        'github_token': os.getenv('GITHUB_TOKEN'),
        'repo_owner': os.getenv('GITHUB_REPO_OWNER'),
        'repo_name': os.getenv('GITHUB_REPO_NAME'),
        'huggingface_model': os.getenv('HUGGINGFACE_MODEL', 'microsoft/DialoGPT-medium'),
        'email_enabled': False,  # Test in simulation mode
        'score_threshold': 70
    }
    
    # Sample job requirements
    job_requirements = {
        'title': 'Senior React Developer',
        'required_skills': ['React', 'JavaScript', 'TypeScript'],
        'preferred_skills': ['Node.js', 'Testing'],
        'min_experience_years': 3,
        'department': 'Engineering'
    }
    
    if not all([config['github_token'], config['repo_owner'], config['repo_name']]):
        print("❌ Missing GitHub configuration for testing")
        return
    
    # Run workflow
    results = run_autonomous_screening(
        job_requirements=job_requirements,
        job_role_folder='react-developer',
        config=config,
        verbose=True
    )
    
    if 'error' in results:
        print(f"❌ Workflow failed: {results['error']}")
    else:
        print(f"✅ Workflow completed successfully!")
        print(f"📊 Results: {results.get('results', {})}")

if __name__ == "__main__":
    test_workflow()