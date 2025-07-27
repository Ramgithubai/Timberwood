@echo off
echo 🤖 AI-Powered CSV Data Analyzer
echo ================================
echo.
echo Checking for Python and dependencies...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

echo ✅ Python found
echo.
echo 💡 Tip: Create .env file with API keys for enhanced AI features
echo    See .env.example for reference
echo.
echo 🚀 Starting AI CSV Analyzer...
echo.
echo If you see errors, run: pip install -r requirements.txt
echo.
echo 📄 Upload any CSV file and start exploring!
echo 🤖 Chat with your data using AI
echo 📊 Create custom X vs Y visualizations
echo 🔍 Filter data with simple Primary/Secondary filters
echo.
echo Press Ctrl+C to stop the server when done
echo.

streamlit run ai_csv_analyzer.py

if errorlevel 1 (
    echo.
    echo ❌ Error occurred. Please install dependencies:
    echo pip install -r requirements.txt
    echo.
    echo Then try again: streamlit run ai_csv_analyzer.py
)

pause
