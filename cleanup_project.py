#!/usr/bin/env python3
"""
Project Cleanup Script
Removes unnecessary files and keeps only essential components
"""

import os
import shutil
import glob

def cleanup_project():
    """Clean up the project directory"""
    
    # Current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🧹 Starting Project Cleanup...")
    print(f"📁 Working directory: {base_dir}")
    
    # Files to remove
    files_to_remove = [
        # Test files
        "test_hs_code_fix.py",
        "test_plot_functionality.py", 
        "test_refactor.py",
        "test_setup.py",
        "test_syntax.py",
        
        # Old application files
        "app.py",
        "enhanced_dashboard.py",
        "quick_start.py",
        "standalone_analysis.py",
        "launch_ai_analyzer.py",
        "fix_colorscales.py",
        
        # Multiple requirements files (keep only requirements_generic_ai.txt)
        "requirements.txt",
        "requirements_ai.txt", 
        "requirements_ai_enhanced.txt",
        "requirements_flexible.txt",
        
        # Multiple batch files (keep only run_ai_analyzer.bat)
        "run_dashboard.bat",
        "run_enhanced_ai_analyzer.bat",
        
        # Documentation files (many are outdated)
        "AI_ANALYZER_README.md",
        "CONFIG_GUIDE.md", 
        "DASHBOARD_OVERVIEW.md",
        "FILE_SUMMARY.md",
        "FINAL_GUIDE.md", 
        "HS_CODE_FIX_GUIDE.md",
        "PLOT_GENERATION_GUIDE.md",
        "QUICK_START.md",
        "README_AI.md",
        "UPDATES.md",
        
        # Duplicate/temporary files
        ".env.fixed",
        "5.15.0",
        
        # Extra CSV files (keep sample_trade_data.csv)
        "timber_businesses.csv",
        
        # Backup file created during testing
        "test_hs_code_fix.py.backup"
    ]
    
    # Directories to remove
    dirs_to_remove = [
        "__pycache__",
        "modules/__pycache__"
    ]
    
    removed_files = []
    removed_dirs = []
    
    # Remove files
    print("\n📄 Removing unnecessary files...")
    for filename in files_to_remove:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                removed_files.append(filename)
                print(f"   ✅ Removed: {filename}")
            except Exception as e:
                print(f"   ❌ Error removing {filename}: {e}")
        else:
            print(f"   ⚠️  Not found: {filename}")
    
    # Remove directories
    print("\n📁 Removing unnecessary directories...")
    for dirname in dirs_to_remove:
        dirpath = os.path.join(base_dir, dirname)
        if os.path.exists(dirpath):
            try:
                shutil.rmtree(dirpath)
                removed_dirs.append(dirname)
                print(f"   ✅ Removed directory: {dirname}")
            except Exception as e:
                print(f"   ❌ Error removing directory {dirname}: {e}")
        else:
            print(f"   ⚠️  Directory not found: {dirname}")
    
    # Rename requirements file
    print("\n📋 Updating requirements file...")
    old_req = os.path.join(base_dir, "requirements_generic_ai.txt")
    new_req = os.path.join(base_dir, "requirements.txt")
    
    if os.path.exists(old_req):
        try:
            if os.path.exists(new_req):
                os.remove(new_req)  # Remove old requirements.txt first
            os.rename(old_req, new_req)
            print(f"   ✅ Renamed requirements_generic_ai.txt to requirements.txt")
        except Exception as e:
            print(f"   ❌ Error renaming requirements file: {e}")
    
    # Create new clean README
    print("\n📝 Creating updated README...")
    readme_content = '''# 🤖 AI-Powered CSV Data Analyzer

A powerful, intelligent CSV data analysis tool with AI chat capabilities, smart visualizations, and business research features.

## ✨ Features

- **🤖 AI Chat**: Chat with your data using multiple AI providers (OpenAI, Groq, Claude, Local Analysis)
- **🔍 Smart Data Explorer**: Simple filtering with Primary/Secondary filters and text search
- **📈 Quick Visualizations**: Individual column analysis + X vs Y custom plotting
- **🔑 Intelligent Detection**: Automatically detects identifier columns (HS codes, product IDs, etc.)
- **🌐 Business Research**: AI-powered web scraping for business contact information
- **📊 Multiple Chart Types**: Scatter plots, bar charts, box plots, heatmaps, correlation analysis

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run ai_csv_analyzer.py
   ```
   Or use the batch file:
   ```bash
   run_ai_analyzer.bat
   ```

3. **Upload your CSV** and start exploring!

## 🔧 Setup (Optional)

For AI chat features, create a `.env` file with your API keys:

```bash
# Copy the example file
cp .env.example .env

# Edit with your API keys
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here  
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## 📁 Project Structure

```
Dashboard/
├── ai_csv_analyzer.py          # Main application
├── data_explorer.py            # Data explorer module
├── modules/
│   └── web_scraping_module.py  # Business research functionality
├── requirements.txt            # Dependencies
├── run_ai_analyzer.bat        # Launch script
├── .env.example               # Environment template
└── sample_trade_data.csv      # Sample data
```

## 💡 Example Use Cases

- **Trade Data Analysis**: Analyze HS codes, shipment patterns, business relationships
- **Sales Data**: Explore customer patterns, product performance, regional trends  
- **Inventory Management**: Track stock levels, identify trends, forecast demand
- **Business Intelligence**: Quick insights, correlation analysis, data quality checks
- **Market Research**: Identify top performers, analyze competition, find business contacts

## 🎯 Smart Features

- **Auto-detects identifiers**: HS codes, product codes, and IDs are handled correctly
- **Intelligent filtering**: No more complex interfaces - just Primary + Secondary filters
- **X vs Y plotting**: Select any two columns for custom visualizations
- **AI-powered insights**: Ask natural language questions about your data
- **Business research**: Find contact info for companies in your data

## 🆘 Support

The tool works with any CSV file and automatically adapts to your data structure. No configuration needed!

For business research features, you'll need OpenAI and Tavily API keys in your `.env` file.
'''
    
    try:
        with open(os.path.join(base_dir, "README.md"), 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"   ✅ Created updated README.md")
    except Exception as e:
        print(f"   ❌ Error creating README: {e}")
    
    # Summary
    print(f"\n🎉 Cleanup Complete!")
    print(f"📄 Removed {len(removed_files)} files")
    print(f"📁 Removed {len(removed_dirs)} directories")
    
    print(f"\n✅ **Essential files remaining:**")
    essential_files = [
        "ai_csv_analyzer.py",
        "data_explorer.py", 
        "modules/web_scraping_module.py",
        "requirements.txt",
        "run_ai_analyzer.bat",
        ".env",
        ".env.example",
        "sample_trade_data.csv",
        "README.md"
    ]
    
    for filename in essential_files:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            print(f"   ✅ {filename}")
        else:
            print(f"   ⚠️  {filename} (missing)")
    
    print(f"\n🚀 **Ready to use! Run:**")
    print(f"   streamlit run ai_csv_analyzer.py")
    print(f"   or")
    print(f"   run_ai_analyzer.bat")

if __name__ == "__main__":
    cleanup_project()
