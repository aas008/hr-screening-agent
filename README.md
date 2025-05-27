# HR Screening Agent

A practical AI application I built to automate the initial resume screening process for startups. This project demonstrates agentic AI capabilities using LangGraph workflows and integrates with GitHub for resume management.

## What This Does

As someone who's been through hiring processes, I know how time-consuming initial resume screening can be. This agent:

- Pulls resumes directly from a GitHub repository structure I designed
- Analyzes them against job requirements using HuggingFace models
- Makes accept/reject decisions based on a scoring system I implemented
- Sends automated emails to candidates
- Provides a simple Streamlit interface for monitoring

The core idea: let AI handle the obvious "yes" and "no" candidates, so humans can focus on the borderline cases.

## My Technical Approach

I chose this stack after experimenting with different options:

**LangGraph** - For the agent workflow. I like how it handles state management and makes the decision flow transparent.

**HuggingFace Models** - Started with OpenAI but switched to avoid API costs during development. Currently using a model that works well for resume analysis.

**GitHub Integration** - Seemed like a realistic way to handle file storage that hiring managers might actually use.

**Streamlit** - Quick to build, gets the job done for a demo interface.

## Project Structure

```
hr-screening-agent/
├── src/
│   ├── agents/           # My agent implementations
│   ├── workflows/        # LangGraph workflow logic
│   └── ui/              # Streamlit dashboard
├── resumes/active/      # Where I organize resumes by role
├── templates/           # Email templates I wrote
└── outputs/            # Generated reports
```

## Running This

1. **Setup:**
   ```bash
   git clone [your-repo]
   pip install -r requirements.txt
   cp .env.example .env  # Add your tokens here
   ```

2. **Add some resumes:**
   ```
   resumes/active/react-developer/
   ├── candidate1.pdf
   └── candidate2.pdf
   ```

3. **Run the screening:**
   ```bash
   python main.py --job-role react-developer --job-title "React Developer"
   ```

   Or use the web interface:
   ```bash
   streamlit run src/ui/streamlit_app.py
   ```

## How I Built the Decision Logic

The agent follows this workflow I designed:

1. **Load resumes** from the specified GitHub folder
2. **Extract text** using PyPDF2/python-docx (had to handle formatting issues)
3. **Analyze with AI** - I prompt the model to score against specific job requirements
4. **Make decisions** - 70% threshold seemed reasonable from my testing
5. **Send emails** - Using templates I wrote for different scenarios

The scoring considers skills match, experience level, and overall fit. I spent time tuning the prompts to get consistent results.

## What I Learned Building This

**LangGraph** was new to me - took some time to understand the state management, but it's powerful for complex workflows.

**Resume parsing** is messier than expected. PDFs especially can be tricky to extract clean text from.

**Email automation** required careful template design to avoid sounding too robotic.

**Threshold tuning** needed several test runs with sample resumes to get right.

## Current Limitations

- Only handles PDF and DOCX files
- Email templates are basic (could be more personalized)
- No bias detection yet (would be a good addition)
- Limited to English resumes
- Requires manual job requirement input

## Future Improvements I'm Considering

- Add interview scheduling integration
- Build better analytics dashboard
- Implement bias detection in scoring
- Add support for more file formats
- Create job description generator

## Testing

I've been testing with sample resumes for different roles. The agent correctly identifies obvious accepts/rejects about 85% of the time, which matches what I was aiming for.

```bash
python main.py --job-role react-developer --test-mode
```

## Why This Matters

For a startup hiring their first few developers, this could save 2-3 hours per open position just on initial screening. That's meaningful time that could go toward actually interviewing the promising candidates.

---

Built as part of my learning journey with agentic AI systems. The code isn't perfect, but it works and solves a real problem I've experienced.
