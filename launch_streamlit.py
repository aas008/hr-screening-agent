#!/usr/bin/env python3
"""
Launch script for HR Screening Agent Streamlit interface
Handles setup and launches the web application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'python-dotenv',
        'langgraph',
        'transformers',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_environment():
    """Check if environment is properly configured"""
    
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found")
        print("💡 Copy .env.example to .env and configure your settings")
        return False
    
    # Load and check critical environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['GITHUB_TOKEN', 'GITHUB_REPO_OWNER', 'GITHUB_REPO_NAME']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   • {var}")
        print("\n💡 Please configure these in your .env file")
        return False
    
    print("✅ Environment configuration is valid")
    return True

def launch_streamlit():
    """Launch the Streamlit application"""
    
    print("🚀 Launching HR Screening Agent Web Interface...")
    print("🌐 The application will open in your default web browser")
    print("📍 Default URL: http://localhost:8501")
    print("\n⚠️  Press Ctrl+C to stop the application")
    print("="*50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, 
            '-m', 'streamlit', 'run', 
            'src/ui/streamlit_app.py',
            '--server.address', 'localhost',
            '--server.port', '8501',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        print("💡 Try running manually: streamlit run src/ui/streamlit_app.py")

def main():
    """Main launcher function"""
    
    print("🎯 HR SCREENING AGENT - WEB INTERFACE LAUNCHER")
    print("="*50)
    
    # Check current directory
    if not Path('main.py').exists():
        print("❌ Please run this script from the project root directory")
        print("💡 Expected files: main.py, src/, requirements.txt")
        return
    
    print("📂 Project directory: ✅")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check environment
    if not check_environment():
        return
    
    print("\n🎉 All checks passed! Starting web interface...")
    print("💡 Tip: Bookmark http://localhost:8501 for quick access")
    
    # Launch the application
    launch_streamlit()

if __name__ == "__main__":
    main()