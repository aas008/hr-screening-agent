"""
Resume Analyzer using HuggingFace Models
Analyzes resumes against job requirements using free AI models
"""

import re
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from .github_loader import CandidateInfo

@dataclass
class AnalysisResult:
    """Data class for resume analysis results"""
    candidate: CandidateInfo
    score: float
    skills_found: List[str]
    skills_missing: List[str]
    experience_years: int
    experience_level: str
    strengths: List[str]
    concerns: List[str]
    action: str  # "accept" or "reject"
    reasoning: str
    confidence: float
    analysis_time_seconds: float

class HuggingFaceResumeAnalyzer:
    """Analyze resumes using HuggingFace models"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", score_threshold: int = 70):
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.analyzer = None
        self.tokenizer = None
        
        print(f"🤖 Initializing HuggingFace analyzer with model: {model_name}")
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the HuggingFace model"""
        
        try:
            # Check if CUDA is available
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                print("🚀 Using GPU acceleration")
            else:
                print("💻 Using CPU (consider GPU for faster processing)")
            
            # Try to load the model for text generation
            self.analyzer = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
                max_length=512,
                truncation=True,
                do_sample=True,
                temperature=0.7,
                return_full_text=False
            )
            
            print("✅ HuggingFace model loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load HuggingFace model: {e}")
            print("🔄 Falling back to rule-based analysis")
            self.analyzer = None
    
    def analyze_resume(self, candidate: CandidateInfo, job_requirements: Dict) -> AnalysisResult:
        """
        Analyze a resume against job requirements
        Returns analysis result with score and recommendation
        """
        
        start_time = datetime.now()
        print(f"🔍 Analyzing: {candidate.name}")
        
        try:
            # Use AI analysis if model is available, otherwise use rule-based
            if self.analyzer:
                result = self._ai_enhanced_analysis(candidate, job_requirements)
            else:
                result = self._rule_based_analysis(candidate, job_requirements)
            
            # Calculate analysis time
            end_time = datetime.now()
            analysis_time = (end_time - start_time).total_seconds()
            result.analysis_time_seconds = analysis_time
            
            print(f"   📊 Score: {result.score}% - {result.action.upper()}")
            print(f"   ⏱️  Analysis time: {analysis_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            # Return a default "manual review" result
            return self._create_error_result(candidate, str(e))
    
    def _ai_enhanced_analysis(self, candidate: CandidateInfo, job_req: Dict) -> AnalysisResult:
        """Use AI model for enhanced resume analysis"""
        
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(candidate, job_req)
            
            # Get AI analysis
            ai_response = self.analyzer(prompt, max_new_tokens=200, num_return_sequences=1)
            ai_text = ai_response[0]['generated_text'] if ai_response else ""
            
            # Parse AI response and combine with rule-based analysis
            rule_result = self._rule_based_analysis(candidate, job_req)
            
            # Enhance with AI insights
            ai_insights = self._parse_ai_response(ai_text)
            
            # Combine results
            return self._combine_analyses(rule_result, ai_insights)
            
        except Exception as e:
            print(f"   ⚠️  AI analysis failed, using rule-based: {e}")
            return self._rule_based_analysis(candidate, job_req)
    
    def _create_analysis_prompt(self, candidate: CandidateInfo, job_req: Dict) -> str:
        """Create prompt for AI analysis"""
        
        # Truncate resume for token limits
        resume_excerpt = candidate.resume_text[:800]
        
        prompt = f"""
Analyze this resume for the {job_req['title']} position.

Required Skills: {', '.join(job_req['required_skills'])}
Minimum Experience: {job_req['min_experience_years']} years

Resume excerpt:
{resume_excerpt}

Provide brief analysis focusing on:
1. Skills match
2. Experience level
3. Key strengths
4. Main concerns
"""
        
        return prompt
    
    def _parse_ai_response(self, ai_text: str) -> Dict:
        """Parse AI response for insights"""
        
        # This is a simplified parser - in production you'd want more sophisticated parsing
        insights = {
            'ai_strengths': [],
            'ai_concerns': [],
            'ai_confidence': 0.7  # Default confidence
        }
        
        # Look for key phrases in AI response
        text_lower = ai_text.lower()
        
        # Positive indicators
        if any(word in text_lower for word in ['strong', 'excellent', 'experienced', 'qualified']):
            insights['ai_confidence'] += 0.1
        
        # Negative indicators
        if any(word in text_lower for word in ['lacking', 'weak', 'insufficient', 'missing']):
            insights['ai_confidence'] -= 0.1
        
        # Keep confidence in reasonable bounds
        insights['ai_confidence'] = max(0.3, min(0.9, insights['ai_confidence']))
        
        return insights
    
    def _combine_analyses(self, rule_result: AnalysisResult, ai_insights: Dict) -> AnalysisResult:
        """Combine rule-based and AI analyses"""
        
        # Adjust confidence based on AI insights
        rule_result.confidence = ai_insights.get('ai_confidence', rule_result.confidence)
        
        # Add AI insights to strengths/concerns if available
        if ai_insights.get('ai_strengths'):
            rule_result.strengths.extend(ai_insights['ai_strengths'])
        
        if ai_insights.get('ai_concerns'):
            rule_result.concerns.extend(ai_insights['ai_concerns'])
        
        return rule_result
    
    def _rule_based_analysis(self, candidate: CandidateInfo, job_req: Dict) -> AnalysisResult:
        """Comprehensive rule-based resume analysis"""
        
        resume_text = candidate.resume_text.lower()
        
        # 1. Skills Analysis
        skills_analysis = self._analyze_skills(resume_text, job_req)
        
        # 2. Experience Analysis
        experience_analysis = self._analyze_experience(resume_text, job_req)
        
        # 3. Overall Quality Analysis
        quality_analysis = self._analyze_resume_quality(resume_text)
        
        # 4. Calculate Overall Score
        overall_score = self._calculate_overall_score(
            skills_analysis, experience_analysis, quality_analysis
        )
        
        # 5. Determine Action (accept/reject based on threshold)
        action = "accept" if overall_score >= self.score_threshold else "reject"
        
        # 6. Generate Reasoning
        reasoning = self._generate_reasoning(
            overall_score, skills_analysis, experience_analysis, job_req
        )
        
        # 7. Identify Strengths and Concerns
        strengths = self._identify_strengths(skills_analysis, experience_analysis, quality_analysis)
        concerns = self._identify_concerns(skills_analysis, experience_analysis, quality_analysis, job_req)
        
        return AnalysisResult(
            candidate=candidate,
            score=round(overall_score, 1),
            skills_found=skills_analysis['found'],
            skills_missing=skills_analysis['missing'],
            experience_years=experience_analysis['years'],
            experience_level=experience_analysis['level'],
            strengths=strengths,
            concerns=concerns,
            action=action,
            reasoning=reasoning,
            confidence=self._calculate_confidence(overall_score, skills_analysis, experience_analysis),
            analysis_time_seconds=0.0  # Will be set later
        )
    
    def _analyze_skills(self, resume_text: str, job_req: Dict) -> Dict:
        """Analyze skills match between resume and job requirements"""
        
        required_skills = [skill.lower() for skill in job_req['required_skills']]
        preferred_skills = [skill.lower() for skill in job_req.get('preferred_skills', [])]
        
        found_required = []
        found_preferred = []
        
        # Check for required skills
        for skill in required_skills:
            if self._skill_mentioned(skill, resume_text):
                found_required.append(skill)
        
        # Check for preferred skills  
        for skill in preferred_skills:
            if self._skill_mentioned(skill, resume_text):
                found_preferred.append(skill)
        
        missing_required = [skill for skill in required_skills if skill not in found_required]
        
        # Calculate skills score
        required_score = (len(found_required) / len(required_skills)) * 100 if required_skills else 100
        preferred_score = (len(found_preferred) / len(preferred_skills)) * 100 if preferred_skills else 0
        
        # Weight: 80% required, 20% preferred
        skills_score = (required_score * 0.8) + (preferred_score * 0.2)
        
        return {
            'score': skills_score,
            'found': found_required + found_preferred,
            'missing': missing_required,
            'required_found': len(found_required),
            'required_total': len(required_skills),
            'preferred_found': len(found_preferred)
        }
    
    def _skill_mentioned(self, skill: str, resume_text: str) -> bool:
        """Check if a skill is mentioned in the resume"""
        
        # Create flexible skill patterns
        skill_patterns = [
            skill,  # Exact match
            skill.replace(' ', ''),  # No spaces (e.g., "node.js" -> "nodejs")
            skill.replace('.', ''),  # No dots (e.g., "node.js" -> "nodejs")
        ]
        
        # Add common variations
        skill_variations = {
            'javascript': ['js', 'ecmascript', 'es6', 'es2015'],
            'typescript': ['ts'],
            'react': ['reactjs', 'react.js'],
            'node.js': ['nodejs', 'node'],
            'css': ['css3', 'cascading style sheets'],
            'html': ['html5', 'hypertext markup'],
            'python': ['py'],
            'machine learning': ['ml', 'artificial intelligence', 'ai'],
            'sql': ['mysql', 'postgresql', 'sqlite', 'database']
        }
        
        if skill in skill_variations:
            skill_patterns.extend(skill_variations[skill])
        
        # Check for any pattern match
        for pattern in skill_patterns:
            if pattern in resume_text:
                return True
        
        return False
    
    def _analyze_experience(self, resume_text: str, job_req: Dict) -> Dict:
        """Analyze experience level from resume"""
        
        # Method 1: Look for explicit experience mentions
        experience_years = self._extract_explicit_experience(resume_text)
        
        # Method 2: Calculate from employment dates
        if experience_years == 0:
            experience_years = self._calculate_experience_from_dates(resume_text)
        
        # Method 3: Estimate from seniority indicators
        if experience_years == 0:
            experience_years = self._estimate_from_seniority_indicators(resume_text)
        
        # Determine experience level
        if experience_years >= 7:
            level = "Senior"
        elif experience_years >= 3:
            level = "Mid-level"
        elif experience_years >= 1:
            level = "Junior"
        else:
            level = "Entry-level"
        
        # Calculate experience score vs requirement
        min_required = job_req['min_experience_years']
        experience_score = min((experience_years / min_required) * 100, 100) if min_required > 0 else 100
        
        return {
            'years': experience_years,
            'level': level,
            'score': experience_score,
            'meets_requirement': experience_years >= min_required
        }
    
    def _extract_explicit_experience(self, resume_text: str) -> int:
        """Extract explicitly mentioned years of experience"""
        
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in\s*\w+',
            r'(\d+)\+?\s*yrs?\s*experience',
            r'experience\s*[:]\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*(?:professional\s*)?(?:work\s*)?experience'
        ]
        
        years_found = []
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            years_found.extend([int(match) for match in matches if match.isdigit()])
        
        return max(years_found) if years_found else 0
    
    def _calculate_experience_from_dates(self, resume_text: str) -> int:
        """Calculate experience from employment date ranges"""
        
        # Look for date patterns like "2020-2023", "Jan 2020 - Present"
        date_patterns = [
            r'(20\d{2})\s*[-–]\s*(20\d{2}|present|current)',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(20\d{2})\s*[-–]\s*(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(20\d{2})|present|current)',
            r'(\d{1,2})/(\d{4})\s*[-–]\s*(?:(\d{1,2})/(\d{4})|present|current)'
        ]
        
        current_year = datetime.now().year
        employment_periods = []
        
        for pattern in date_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            
            for match in matches:
                try:
                    if len(match) == 2:  # Simple year range
                        start_year = int(match[0])
                        end_year = current_year if match[1].lower() in ['present', 'current'] else int(match[1])
                        employment_periods.append((start_year, end_year))
                except (ValueError, IndexError):
                    continue
        
        # Calculate total unique employment time
        if not employment_periods:
            return 0
        
        # Sort periods and merge overlapping ones
        employment_periods.sort()
        merged_periods = []
        
        for start, end in employment_periods:
            if not merged_periods or start > merged_periods[-1][1]:
                merged_periods.append((start, end))
            else:
                # Merge overlapping periods
                merged_periods[-1] = (merged_periods[-1][0], max(merged_periods[-1][1], end))
        
        # Calculate total years
        total_years = sum(end - start for start, end in merged_periods)
        return max(0, total_years)
    
    def _estimate_from_seniority_indicators(self, resume_text: str) -> int:
        """Estimate experience from seniority indicators"""
        
        seniority_indicators = {
            'senior': 5,
            'lead': 6,
            'principal': 8,
            'staff': 7,
            'architect': 8,
            'manager': 6,
            'director': 10,
            'team lead': 5,
            'tech lead': 6,
            'junior': 1,
            'intern': 0,
            'entry': 0,
            'graduate': 0
        }
        
        max_experience = 0
        
        for indicator, years in seniority_indicators.items():
            if indicator in resume_text:
                max_experience = max(max_experience, years)
        
        return max_experience
    
    def _analyze_resume_quality(self, resume_text: str) -> Dict:
        """Analyze overall resume quality indicators"""
        
        quality_score = 0
        indicators = []
        
        # Length check (good resumes are typically 200-2000 words)
        word_count = len(resume_text.split())
        if 200 <= word_count <= 2000:
            quality_score += 20
            indicators.append("Appropriate length")
        elif word_count < 200:
            indicators.append("Too brief")
        else:
            indicators.append("Too lengthy")
        
        # Structure indicators
        structure_keywords = [
            'experience', 'education', 'skills', 'projects', 
            'achievements', 'responsibilities', 'summary'
        ]
        
        sections_found = sum(1 for keyword in structure_keywords if keyword in resume_text)
        quality_score += min(sections_found * 10, 30)
        
        if sections_found >= 4:
            indicators.append("Well-structured")
        
        # Professional language indicators
        professional_terms = [
            'developed', 'implemented', 'managed', 'led', 'created',
            'designed', 'optimized', 'improved', 'collaborated'
        ]
        
        professional_count = sum(1 for term in professional_terms if term in resume_text)
        quality_score += min(professional_count * 5, 25)
        
        if professional_count >= 3:
            indicators.append("Professional language")
        
        # Technical depth indicators
        technical_indicators = [
            'github', 'portfolio', 'project', 'framework', 'library',
            'database', 'api', 'testing', 'deployment'
        ]
        
        technical_count = sum(1 for indicator in technical_indicators if indicator in resume_text)
        quality_score += min(technical_count * 3, 25)
        
        if technical_count >= 4:
            indicators.append("Technical depth")
        
        return {
            'score': min(quality_score, 100),
            'indicators': indicators,
            'word_count': word_count
        }
    
    def _calculate_overall_score(self, skills_analysis: Dict, experience_analysis: Dict, quality_analysis: Dict) -> float:
        """Calculate overall candidate score"""
        
        # Weighted scoring:
        # - Skills: 60%
        # - Experience: 30% 
        # - Quality: 10%
        
        skills_score = skills_analysis['score']
        experience_score = experience_analysis['score']
        quality_score = quality_analysis['score']
        
        overall_score = (
            skills_score * 0.6 +
            experience_score * 0.3 +
            quality_score * 0.1
        )
        
        return min(overall_score, 100.0)
    
    def _generate_reasoning(self, score: float, skills_analysis: Dict, experience_analysis: Dict, job_req: Dict) -> str:
        """Generate human-readable reasoning for the decision"""
        
        reasoning_parts = []
        
        # Score summary
        if score >= 80:
            reasoning_parts.append(f"Excellent candidate with {score:.1f}% match")
        elif score >= 70:
            reasoning_parts.append(f"Strong candidate with {score:.1f}% match")
        elif score >= 50:
            reasoning_parts.append(f"Potential candidate with {score:.1f}% match")
        else:
            reasoning_parts.append(f"Below requirements with {score:.1f}% match")
        
        # Skills reasoning
        skills_found = skills_analysis['required_found']
        skills_total = skills_analysis['required_total']
        
        if skills_found == skills_total:
            reasoning_parts.append(f"Has all {skills_total} required skills")
        elif skills_found > 0:
            reasoning_parts.append(f"Has {skills_found}/{skills_total} required skills")
        else:
            reasoning_parts.append("Missing most required skills")
        
        # Experience reasoning
        exp_years = experience_analysis['years']
        min_required = job_req['min_experience_years']
        
        if exp_years >= min_required + 2:
            reasoning_parts.append(f"Exceeds experience requirement ({exp_years} vs {min_required} years)")
        elif exp_years >= min_required:
            reasoning_parts.append(f"Meets experience requirement ({exp_years} years)")
        else:
            reasoning_parts.append(f"Below experience requirement ({exp_years} vs {min_required} years)")
        
        return ". ".join(reasoning_parts) + "."
    
    def _identify_strengths(self, skills_analysis: Dict, experience_analysis: Dict, quality_analysis: Dict) -> List[str]:
        """Identify candidate strengths"""
        
        strengths = []
        
        # Skills strengths
        if skills_analysis['required_found'] >= skills_analysis['required_total'] * 0.8:
            strengths.append("Strong technical skills match")
        
        if skills_analysis['preferred_found'] > 0:
            strengths.append("Has preferred skills")
        
        # Experience strengths
        exp_level = experience_analysis['level']
        if exp_level in ['Senior', 'Mid-level']:
            strengths.append(f"{exp_level} experience level")
        
        # Quality strengths
        quality_indicators = quality_analysis['indicators']
        if 'Well-structured' in quality_indicators:
            strengths.append("Well-organized resume")
        
        if 'Professional language' in quality_indicators:
            strengths.append("Professional presentation")
        
        if 'Technical depth' in quality_indicators:
            strengths.append("Demonstrates technical depth")
        
        return strengths[:5]  # Limit to top 5 strengths
    
    def _identify_concerns(self, skills_analysis: Dict, experience_analysis: Dict, quality_analysis: Dict, job_req: Dict) -> List[str]:
        """Identify potential concerns about the candidate"""
        
        concerns = []
        
        # Skills concerns
        missing_skills = skills_analysis['missing']
        if missing_skills:
            if len(missing_skills) == 1:
                concerns.append(f"Missing {missing_skills[0]} skill")
            else:
                concerns.append(f"Missing {len(missing_skills)} required skills")
        
        # Experience concerns
        if not experience_analysis['meets_requirement']:
            exp_years = experience_analysis['years']
            min_required = job_req['min_experience_years']
            concerns.append(f"Only {exp_years} years experience (need {min_required})")
        
        if experience_analysis['level'] == 'Entry-level' and job_req['min_experience_years'] > 1:
            concerns.append("Entry-level for mid/senior role")
        
        # Quality concerns
        word_count = quality_analysis['word_count']
        if word_count < 200:
            concerns.append("Resume lacks detail")
        elif word_count > 2000:
            concerns.append("Resume too lengthy")
        
        quality_score = quality_analysis['score']
        if quality_score < 50:
            concerns.append("Poor resume quality")
        
        return concerns[:3]  # Limit to top 3 concerns
    
    def _calculate_confidence(self, score: float, skills_analysis: Dict, experience_analysis: Dict) -> float:
        """Calculate confidence in the analysis"""
        
        confidence = 0.7  # Base confidence
        
        # Higher confidence for clear accept/reject cases
        if score >= 85 or score <= 30:
            confidence += 0.2
        elif score >= 75 or score <= 40:
            confidence += 0.1
        
        # Adjust based on data quality
        if skills_analysis['required_total'] > 0:
            confidence += 0.05
        
        if experience_analysis['years'] > 0:
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    def _create_error_result(self, candidate: CandidateInfo, error_msg: str) -> AnalysisResult:
        """Create result for cases where analysis failed"""
        
        return AnalysisResult(
            candidate=candidate,
            score=0.0,
            skills_found=[],
            skills_missing=[],
            experience_years=0,
            experience_level="Unknown",
            strengths=[],
            concerns=[f"Analysis error: {error_msg}"],
            action="manual_review",
            reasoning=f"Could not analyze resume automatically: {error_msg}",
            confidence=0.0,
            analysis_time_seconds=0.0
        )

def test_resume_analysis():
    """Test function for resume analysis (development use)"""
    
    # Sample test data
    sample_resume = """
    John Smith
    Senior Software Engineer
    john.smith@email.com
    (555) 123-4567
    
    Experience:
    Senior React Developer - TechCorp (2020-Present)
    - Developed scalable web applications using React, TypeScript, and Node.js
    - Led team of 4 developers on multiple projects
    - Implemented automated testing with Jest and Cypress
    
    Frontend Developer - StartupXYZ (2018-2020)  
    - Built responsive UIs with React and CSS
    - Collaborated with design team on user experience
    - Optimized application performance
    
    Skills: React, JavaScript, TypeScript, HTML, CSS, Node.js, Git, Testing
    Education: BS Computer Science, University of Technology (2018)
    """
    
    job_requirements = {
        'title': 'Senior React Developer',
        'required_skills': ['React', 'JavaScript', 'TypeScript', 'HTML', 'CSS'],
        'preferred_skills': ['Node.js', 'Testing', 'Git'],
        'min_experience_years': 3,
        'department': 'Engineering'
    }
    
    # Create candidate info
    from github_loader import CandidateInfo
    candidate = CandidateInfo(
        name="John Smith",
        email="john.smith@email.com",
        phone="(555) 123-4567",
        resume_text=sample_resume,
        file_name="john_smith.pdf",
        application_date=datetime.now().isoformat()
    )
    
    # Run analysis
    analyzer = HuggingFaceResumeAnalyzer(score_threshold=70)
    result = analyzer.analyze_resume(candidate, job_requirements)
    
    # Print results
    print(f"\n📊 ANALYSIS RESULTS FOR {result.candidate.name}")
    print("=" * 50)
    print(f"Overall Score: {result.score}%")
    print(f"Action: {result.action.upper()}")
    print(f"Experience: {result.experience_years} years ({result.experience_level})")
    print(f"Skills Found: {', '.join(result.skills_found)}")
    if result.skills_missing:
        print(f"Skills Missing: {', '.join(result.skills_missing)}")
    print(f"Strengths: {', '.join(result.strengths)}")
    if result.concerns:
        print(f"Concerns: {', '.join(result.concerns)}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Analysis Time: {result.analysis_time_seconds:.2f}s")

if __name__ == "__main__":
    test_resume_analysis()