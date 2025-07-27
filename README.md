# 🤖 AI-Powered CSV Data Analyzer

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
