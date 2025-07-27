import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import requests
import json
import warnings
import os
import re
import sys
import asyncio
import tempfile

# Handle environment variables for both local and Streamlit Cloud
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load from .env file if running locally
except ImportError:
    # dotenv not available (e.g., on Streamlit Cloud)
    pass

# Import web scraping module
from modules.web_scraping_module import perform_web_scraping

# Import simplified data explorer
from data_explorer import create_data_explorer

warnings.filterwarnings('ignore')

# Helper function to get environment variables from either .env or Streamlit secrets
def get_env_var(key, default=None):
    """Get environment variable from .env file (local) or Streamlit secrets (cloud)"""
    # First try regular environment variables (from .env or system)
    value = os.getenv(key)
    if value:
        return value
    
    # Then try Streamlit secrets (for Streamlit Cloud deployment)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    return default

# Page configuration
st.set_page_config(
    page_title="AI-Powered CSV Data Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better dark/light mode support
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Better contrast for both light and dark modes */
    .user-message {
        background: rgba(33, 150, 243, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
        color: var(--text-color);
    }
    
    .ai-message {
        background: rgba(76, 175, 80, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4caf50;
        color: var(--text-color);
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .user-message {
            background: rgba(33, 150, 243, 0.2);
            color: #e0e0e0;
        }
        
        .ai-message {
            background: rgba(76, 175, 80, 0.2);
            color: #e0e0e0;
        }
    }
    
    /* Streamlit dark theme support */
    .stApp[data-theme="dark"] .user-message {
        background: rgba(33, 150, 243, 0.2);
        color: #ffffff;
    }
    
    .stApp[data-theme="dark"] .ai-message {
        background: rgba(76, 175, 80, 0.2);
        color: #ffffff;
    }
    
    /* Light theme support */
    .stApp[data-theme="light"] .user-message {
        background: rgba(33, 150, 243, 0.1);
        color: #000000;
    }
    
    .stApp[data-theme="light"] .ai-message {
        background: rgba(76, 175, 80, 0.1);
        color: #000000;
    }
    
    .data-insight {
        background: rgba(255, 152, 0, 0.1);
        padding: 10px;
        border-radius: 8px;
        border-left: 3px solid #ff9800;
        margin: 10px 0;
    }
    
    .identifier-warning {
        background: rgba(255, 193, 7, 0.1);
        padding: 10px;
        border-radius: 8px;
        border-left: 3px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def detect_identifier_columns(df):
    """
    Detect columns that should be treated as identifiers (like HS codes) rather than numeric values
    """
    identifier_columns = []
    identifier_patterns = [
        # HS codes and trade-related identifiers
        r'.*hs.*code.*', r'.*harmonized.*', r'.*tariff.*', r'.*commodity.*code.*',
        # Product/item identifiers  
        r'.*product.*code.*', r'.*item.*code.*', r'.*sku.*', r'.*barcode.*', r'.*upc.*',
        # General ID patterns
        r'.*\bid\b.*', r'.*identifier.*', r'.*ref.*', r'.*code.*', r'.*key.*',
        # Postal/geographic codes
        r'.*zip.*', r'.*postal.*', r'.*country.*code.*', r'.*region.*code.*',
        # Other common identifiers
        r'.*serial.*', r'.*batch.*', r'.*lot.*'
    ]
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check if column name matches identifier patterns
        is_identifier_by_name = any(re.match(pattern, col_lower) for pattern in identifier_patterns)
        
        # Check data characteristics for likely identifiers
        if df[col].dtype in ['int64', 'float64'] or df[col].dtype == 'object':
            sample_values = df[col].dropna().astype(str).head(100)
            
            if len(sample_values) > 0:
                # Check for patterns that suggest identifiers
                has_leading_zeros = any(val.startswith('0') and len(val) > 1 for val in sample_values if val.isdigit())
                has_fixed_length = len(set(len(str(val)) for val in sample_values)) <= 3  # Most values have similar length
                mostly_unique = df[col].nunique() / len(df) > 0.8  # High uniqueness ratio
                contains_non_numeric = any(not str(val).replace('.', '').isdigit() for val in sample_values)
                
                # HS codes are typically 4-10 digits
                looks_like_hs_code = all(
                    len(str(val).replace('.', '')) >= 4 and 
                    len(str(val).replace('.', '')) <= 10 
                    for val in sample_values[:10] if str(val).replace('.', '').isdigit()
                )
                
                is_identifier_by_data = (
                    has_leading_zeros or 
                    (has_fixed_length and mostly_unique) or
                    (looks_like_hs_code and col_lower in ['hs_code', 'hs', 'code', 'tariff_code']) or
                    (contains_non_numeric and not df[col].dtype in ['datetime64[ns]'])
                )
                
                if is_identifier_by_name or is_identifier_by_data:
                    identifier_columns.append(col)
    
    return identifier_columns

def smart_csv_loader(uploaded_file):
    """
    Intelligently load CSV with proper data type detection for identifiers
    """
    # First pass: read with default types to analyze structure
    try:
        df_preview = pd.read_csv(uploaded_file, nrows=1000)
        uploaded_file.seek(0)  # Reset file pointer
        
        # Detect identifier columns
        identifier_cols = detect_identifier_columns(df_preview)
        
        # Create dtype dictionary to force string types for identifiers
        dtype_dict = {}
        for col in identifier_cols:
            dtype_dict[col] = str
        
        # Read the full file with proper dtypes
        df = pd.read_csv(uploaded_file, dtype=dtype_dict)
        
        # Additional processing for identifier columns
        for col in identifier_cols:
            # Ensure identifiers are treated as strings
            df[col] = df[col].astype(str)
            # Clean up common issues
            df[col] = df[col].str.strip()  # Remove whitespace
            df[col] = df[col].replace('nan', np.nan)  # Handle 'nan' strings
            
        # Basic column name cleaning
        df.columns = df.columns.str.strip()
        
        # Try to parse date columns automatically
        for col in df.columns:
            if col not in identifier_cols and any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        return df, identifier_cols
        
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None, []



class GenericDataAI:
    """Generic AI assistant for CSV data analysis"""
    
    def __init__(self):
        self.data_summary = {}
        self.column_insights = {}
        self.identifier_columns = []
        # Load API keys from environment (works for both local .env and Streamlit secrets)
        self.groq_api_key = get_env_var('GROQ_API_KEY')
        self.openai_api_key = get_env_var('OPENAI_API_KEY')
        self.anthropic_api_key = get_env_var('ANTHROPIC_API_KEY')
    
    def analyze_dataset(self, df, identifier_cols=None):
        """Dynamically analyze any dataset to understand its structure and content"""
        if identifier_cols:
            self.identifier_columns = identifier_cols
        
        analysis = {
            "basic_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            },
            "column_types": {},
            "data_insights": {},
            "sample_data": df.head(3).to_dict('records'),
            "missing_data": df.isnull().sum().to_dict(),
            "numeric_summary": {},
            "categorical_summary": {},
            "identifier_summary": {}
        }
        
        # Analyze each column
        for col in df.columns:
            dtype = str(df[col].dtype)
            analysis["column_types"][col] = dtype
            
            # Handle identifier columns specially
            if col in self.identifier_columns:
                value_counts = df[col].value_counts().head(10)
                analysis["identifier_summary"][col] = {
                    "unique_count": int(df[col].nunique()),
                    "top_values": value_counts.to_dict(),
                    "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else "N/A",
                    "sample_values": df[col].dropna().head(5).tolist()
                }
                
            # Numeric columns (excluding identifiers)
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32'] and col not in self.identifier_columns:
                analysis["numeric_summary"][col] = {
                    "min": float(df[col].min()) if not df[col].empty else 0,
                    "max": float(df[col].max()) if not df[col].empty else 0,
                    "mean": float(df[col].mean()) if not df[col].empty else 0,
                    "std": float(df[col].std()) if not df[col].empty else 0,
                    "unique_count": int(df[col].nunique())
                }
            
            # Categorical/text columns (excluding identifiers)
            elif (df[col].dtype == 'object' or df[col].dtype.name == 'category') and col not in self.identifier_columns:
                value_counts = df[col].value_counts().head(10)
                analysis["categorical_summary"][col] = {
                    "unique_count": int(df[col].nunique()),
                    "top_values": value_counts.to_dict(),
                    "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else "N/A"
                }
            
            # Date columns (try to detect)
            if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']) or dtype == 'datetime64[ns]':
                if df[col].dtype == 'object':
                    # Try to parse dates
                    try:
                        parsed_dates = pd.to_datetime(df[col], errors='coerce')
                        if not parsed_dates.isna().all():
                            analysis["data_insights"][f"{col}_date_range"] = {
                                "earliest": str(parsed_dates.min()),
                                "latest": str(parsed_dates.max()),
                                "span_days": (parsed_dates.max() - parsed_dates.min()).days if parsed_dates.notna().any() else 0
                            }
                    except:
                        pass
        
        self.data_summary = analysis
        return analysis
    
    def generate_data_context(self, df, question=""):
        """Generate context about the data for AI"""
        if not self.data_summary:
            self.analyze_dataset(df, self.identifier_columns)
        
        context = f"""
DATASET OVERVIEW:
- Dataset has {self.data_summary['basic_info']['rows']:,} rows and {self.data_summary['basic_info']['columns']} columns
- Memory usage: {self.data_summary['basic_info']['memory_usage']}
- Columns: {', '.join(self.data_summary['basic_info']['column_names'])}

COLUMN TYPES:"""
        
        for col, dtype in self.data_summary['column_types'].items():
            col_type = "identifier" if col in self.identifier_columns else dtype
            context += f"\n- {col}: {col_type}"
        
        # Identifier columns summary
        if self.data_summary['identifier_summary']:
            context += "\n\nIDENTIFIER COLUMNS (HS Codes, Product Codes, etc.):"
            for col, stats in self.data_summary['identifier_summary'].items():
                context += f"\n- {col}: {stats['unique_count']} unique values, most common: '{stats['most_common']}'"
                context += f"\n  Sample values: {', '.join(map(str, stats['sample_values'][:3]))}"
        
        context += "\n\nNUMERIC COLUMNS SUMMARY:"
        for col, stats in self.data_summary['numeric_summary'].items():
            context += f"\n- {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, unique={stats['unique_count']}"
        
        context += "\n\nCATEGORICAL COLUMNS SUMMARY:"
        for col, stats in self.data_summary['categorical_summary'].items():
            top_value = list(stats['top_values'].keys())[0] if stats['top_values'] else 'N/A'
            context += f"\n- {col}: {stats['unique_count']} unique values, most common: '{top_value}'"
        
        context += "\n\nSAMPLE DATA (first 3 rows):"
        for i, row in enumerate(self.data_summary['sample_data']):
            context += f"\nRow {i+1}: {row}"
        
        if any(keyword in question.lower() for keyword in ['trend', 'time', 'date', 'change']):
            context += "\n\nDATE/TIME INSIGHTS:"
            for key, info in self.data_summary['data_insights'].items():
                if 'date_range' in key:
                    context += f"\n- {key}: {info['earliest']} to {info['latest']} ({info['span_days']} days)"
        
        return context
    
    def get_ai_response(self, question, df, provider="Local Analysis"):
        """Get AI response for the question"""
        # Get text response
        if provider == "Local Analysis":
            text_response = self.get_local_response(question, df)
        elif provider == "Claude":
            text_response = self.get_claude_response(question, df)
        elif provider == "Groq":
            text_response = self.get_groq_response(question, df)
        elif provider == "OpenAI":
            text_response = self.get_openai_response(question, df)
        else:
            text_response = self.get_local_response(question, df)
        
        return text_response
    
    def get_claude_response(self, question, df):
        """Get response using built-in Claude API"""
        try:
            context = self.generate_data_context(df, question)
            
            prompt = f"""You are a data analyst AI. Analyze this dataset and answer the user's question.

{context}

USER QUESTION: {question}

Instructions:
- Provide specific insights based on the actual data structure and content shown above
- Note that identifier columns (like HS codes) are treated as categorical data, not numeric
- If the question asks for analysis, use the statistics provided
- If asking about trends, mention what columns could be used for time-series analysis
- If asking about relationships, suggest relevant column combinations
- Be practical and actionable in your response
- If you need more specific data to answer fully, suggest what analysis would help
- Focus on text-based analysis and insights rather than visualizations

Answer:"""
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                return "I'm having trouble connecting right now. Please try again."
                
        except Exception as e:
            return f"Error: {str(e)[:100]}..."
    
    def get_groq_response(self, question, df):
        """Get response using Groq API"""
        if not self.groq_api_key:
            return "Groq API key not found in environment variables. Please add GROQ_API_KEY to your .env file."
        
        try:
            context = self.generate_data_context(df, question)
            
            prompt = f"""You are a data analyst. Answer questions about this dataset:

{context}

Question: {question}

Provide specific insights based on the data shown above. Note that identifier columns (like HS codes) are categorical, not numeric. Focus on text-based analysis and insights."""
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.2
                }
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"GenAI API error: {response.status_code}. Please check your API key in .env file."
                
        except Exception as e:
            return f"GenAI error: {str(e)[:100]}..."
    
    def get_openai_response(self, question, df):
        """Get response using OpenAI API"""
        if not self.openai_api_key:
            return "OpenAI API key not found in environment variables. Please add OPENAI_API_KEY to your .env file."
        
        try:
            context = self.generate_data_context(df, question)
            
            prompt = f"""Analyze this dataset and answer the question:

{context}

Question: {question}

Provide insights based on the actual data structure and statistics shown. Note that identifier columns (like HS codes) are categorical, not numeric. Focus on text-based analysis and insights."""
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.2
                }
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"GenAI API error: {response.status_code}. Please check your API key in .env file."
                
        except Exception as e:
            return f"GenAI error: {str(e)[:100]}..."
    
    def get_local_response(self, question, df):
        """Get response using local analysis"""
        if not self.data_summary:
            self.analyze_dataset(df, self.identifier_columns)
        
        question_lower = question.lower()
        response = "**Data Analysis:**\n\n"
        
        # Dataset overview
        if any(word in question_lower for word in ['overview', 'summary', 'describe', 'what']):
            response += f"📊 **Dataset Overview:**\n"
            response += f"- {self.data_summary['basic_info']['rows']:,} rows, {self.data_summary['basic_info']['columns']} columns\n"
            response += f"- Columns: {', '.join(self.data_summary['basic_info']['column_names'][:5])}{'...' if len(self.data_summary['basic_info']['column_names']) > 5 else ''}\n\n"
            
            # Show identifier columns if present
            if self.identifier_columns:
                response += f"🔑 **Identifier Columns:** {', '.join(self.identifier_columns)}\n"
                response += "   *(These are treated as categorical data, not numeric)*\n\n"
        
        # Identifier columns analysis
        if self.data_summary['identifier_summary'] and any(word in question_lower for word in ['hs', 'code', 'identifier', 'id']):
            response += "🔑 **Identifier Columns (HS Codes, Product Codes, etc.):**\n"
            for col, stats in list(self.data_summary['identifier_summary'].items())[:5]:
                response += f"- **{col}:** {stats['unique_count']} unique values, most common: '{stats['most_common']}'\n"
                response += f"  Sample: {', '.join(map(str, stats['sample_values'][:3]))}\n"
            response += "\n"
        
        # Numeric columns analysis
        if any(word in question_lower for word in ['numbers', 'numeric', 'statistics', 'stats']):
            if self.data_summary['numeric_summary']:
                response += "📈 **Numeric Columns:**\n"
                for col, stats in list(self.data_summary['numeric_summary'].items())[:5]:
                    response += f"- **{col}:** {stats['min']:.1f} to {stats['max']:.1f} (avg: {stats['mean']:.1f})\n"
                response += "\n"
        
        # Categorical analysis
        if any(word in question_lower for word in ['categories', 'categorical', 'text', 'unique']):
            if self.data_summary['categorical_summary']:
                response += "📝 **Categorical Columns:**\n"
                for col, stats in list(self.data_summary['categorical_summary'].items())[:5]:
                    response += f"- **{col}:** {stats['unique_count']} unique values, most common: '{stats['most_common']}'\n"
                response += "\n"
        
        # Trends/time analysis
        if any(word in question_lower for word in ['trend', 'time', 'date', 'change']):
            date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'created', 'updated'])]
            if date_cols:
                response += f"⏰ **Time-based Analysis:**\n"
                response += f"- Potential date columns: {', '.join(date_cols)}\n"
                for key, info in self.data_summary['data_insights'].items():
                    if 'date_range' in key:
                        response += f"- {key.replace('_date_range', '')}: {info['span_days']} days of data\n"
                response += "\n"
        
        # Missing data
        missing_cols = [col for col, missing in self.data_summary['missing_data'].items() if missing > 0]
        if missing_cols and any(word in question_lower for word in ['missing', 'null', 'empty', 'quality']):
            response += "⚠️ **Data Quality:**\n"
            for col in missing_cols[:5]:
                missing_count = self.data_summary['missing_data'][col]
                missing_pct = (missing_count / self.data_summary['basic_info']['rows']) * 100
                response += f"- **{col}:** {missing_count} missing values ({missing_pct:.1f}%)\n"
            response += "\n"
        
        # Visualization-related responses
        if any(word in question_lower for word in ['plot', 'chart', 'graph', 'visualize', 'show']):
            response += "📊 **Visualization Suggestion:**\n"
            response += "For visualizations, you can use the 'Quick Viz' tab which provides ready-made charts based on your data structure.\n"
            if self.identifier_columns:
                response += "Note: Identifier columns (like HS codes) are best visualized as bar charts showing frequency distributions.\n\n"
        
        # Suggest analyses
        response += "💡 **Suggested Questions:**\n"
        response += "- 'Analyze trends over time'\n"
        if self.identifier_columns:
            response += f"- 'What are the top values in {self.identifier_columns[0]}?'\n"
        if self.data_summary['numeric_summary']:
            numeric_col = list(self.data_summary['numeric_summary'].keys())[0]
            response += f"- 'Analyze the distribution of {numeric_col}'\n"
        response += "- 'What correlations exist between numeric columns?'\n"
        
        return response

def create_data_overview(df, ai_assistant, identifier_cols):
    """Create an overview of the loaded dataset"""
    st.subheader("📊 Dataset Overview")
    
    # Show identifier detection results
    if identifier_cols:
        st.markdown(f"""
        <div class="identifier-warning">
            <strong>🔑 Identifier Columns Detected:</strong><br>
            The following columns are being treated as identifiers (not numeric): <strong>{', '.join(identifier_cols)}</strong><br>
            <em>This includes HS codes, product codes, and other ID fields.</em>
        </div>
        """, unsafe_allow_html=True)
    
    # Get analysis
    analysis = ai_assistant.analyze_dataset(df, identifier_cols)
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{analysis['basic_info']['rows']:,}")
    
    with col2:
        st.metric("Total Columns", analysis['basic_info']['columns'])
    
    with col3:
        numeric_cols = len(analysis['numeric_summary'])
        st.metric("Numeric Columns", numeric_cols)
    
    with col4:
        categorical_cols = len(analysis['categorical_summary'])
        identifier_cols_count = len(analysis['identifier_summary'])
        st.metric("Categorical + ID Columns", categorical_cols + identifier_cols_count)
    
    # Data insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🔑 Identifier Columns:**")
        if analysis['identifier_summary']:
            for col, stats in list(analysis['identifier_summary'].items())[:5]:
                with st.expander(f"{col} (Identifier)"):
                    st.write(f"Unique values: {stats['unique_count']}")
                    st.write(f"Most common: {stats['most_common']}")
                    st.write(f"Sample values: {', '.join(map(str, stats['sample_values'][:3]))}")
        else:
            st.write("No identifier columns detected.")
    
    with col2:
        st.write("**📈 Numeric Columns:**")
        if analysis['numeric_summary']:
            for col, stats in list(analysis['numeric_summary'].items())[:5]:
                with st.expander(f"{col}"):
                    st.write(f"Min: {stats['min']:.2f}")
                    st.write(f"Max: {stats['max']:.2f}")
                    st.write(f"Mean: {stats['mean']:.2f}")
                    st.write(f"Unique values: {stats['unique_count']}")
        else:
            st.write("No purely numeric columns found.")
    
    with col3:
        st.write("**📝 Categorical Columns:**")
        if analysis['categorical_summary']:
            for col, stats in list(analysis['categorical_summary'].items())[:5]:
                with st.expander(f"{col}"):
                    st.write(f"Unique values: {stats['unique_count']}")
                    st.write(f"Most common: {stats['most_common']}")
                    if stats['top_values']:
                        st.write("Top values:")
                        for value, count in list(stats['top_values'].items())[:3]:
                            st.write(f"  • {value}: {count}")
        else:
            st.write("No text categorical columns found.")
    
    # Missing data analysis
    missing_data = {col: missing for col, missing in analysis['missing_data'].items() if missing > 0}
    if missing_data:
        st.write("**⚠️ Missing Data:**")
        missing_df = pd.DataFrame([
            {"Column": col, "Missing Count": missing, "Missing %": f"{(missing/len(df)*100):.1f}%"}
            for col, missing in missing_data.items()
        ])
        st.dataframe(missing_df, use_container_width=True)



def create_ai_chat_section(df, identifier_cols):
    """Create the AI chat interface for data analysis"""
    st.subheader("🤖 Chat with Your Data")
    
    # Initialize AI assistant
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = GenericDataAI()
    
    ai_assistant = st.session_state.ai_assistant
    ai_assistant.identifier_columns = identifier_cols
    
    # Auto-select AI provider based on available API keys
    if 'ai_provider' not in st.session_state:
        if ai_assistant.groq_api_key:
            st.session_state.ai_provider = "Groq"
        elif ai_assistant.openai_api_key:
            st.session_state.ai_provider = "OpenAI"
        elif ai_assistant.anthropic_api_key:
            st.session_state.ai_provider = "Claude"
        else:
            st.session_state.ai_provider = "Local Analysis"
    
    # Show current AI provider status
    ai_provider = st.session_state.ai_provider
    
    with st.expander("🔧 AI Provider Status", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Available Providers:**")
            if ai_assistant.groq_api_key:
                st.write("✅ Groq (configured)")
            else:
                st.write("❌ Groq (add GROQ_API_KEY to .env)")
                
            if ai_assistant.openai_api_key:
                st.write("✅ OpenAI (configured)")
            else:
                st.write("❌ OpenAI (add OPENAI_API_KEY to .env)")
        
        with col2:
            if ai_assistant.anthropic_api_key:
                st.write("✅ Claude (configured)")
            else:
                st.write("❌ Claude (add ANTHROPIC_API_KEY to .env)")
            
            st.write("✅ Local Analysis (always available)")
        
        # Provider selection
        available_providers = ["Local Analysis"]
        if ai_assistant.groq_api_key:
            available_providers.append("Groq")
        if ai_assistant.openai_api_key:
            available_providers.append("OpenAI")
        if ai_assistant.anthropic_api_key:
            available_providers.append("Claude")
        
        selected_provider = st.selectbox("Select AI Provider:", available_providers, 
                                       index=available_providers.index(ai_provider) if ai_provider in available_providers else 0)
        st.session_state.ai_provider = selected_provider
    
    # Dynamic example questions based on data
    if df is not None:
        columns = df.columns.tolist()
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() if col not in identifier_cols]
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        st.write("**💡 Suggested Questions for Your Data:**")
        
        example_questions = [
            "What's the overview of this dataset?",
            f"Analyze the distribution of {numeric_cols[0] if numeric_cols else 'numeric columns'}",
            f"What are the top values in {identifier_cols[0] if identifier_cols else categorical_cols[0] if categorical_cols else 'categorical columns'}?",
            "What correlations exist between numeric columns?",
            "Analyze the relationship between columns",
            f"What are the most common HS codes?" if any('hs' in col.lower() for col in identifier_cols) else "What are the most common identifier values?"
        ]
        
        cols = st.columns(3)
        for i, question in enumerate(example_questions):
            with cols[i % 3]:
                if st.button(f"📋 {question[:25]}...", key=f"example_{i}", help=question):
                    st.session_state.current_question = question
    
    # Chat interface
    st.markdown("---")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Question input
    user_question = st.text_area(
        "🎯 Ask anything about your data:",
        value=st.session_state.get('current_question', ''),
        height=80,
        placeholder="e.g., What are the top HS codes? Analyze correlations, Describe the relationship between X and Y"
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ask_button = st.button("🤖 Analyze with GenAI", type="primary")
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.current_question = ""
            st.rerun()
    
    with col3:
        if st.button("📊 Quick Stats"):
            st.session_state.current_question = "Give me a comprehensive overview and key insights about this dataset"
    
    # Process question
    if ask_button and user_question.strip():
        with st.spinner(f"🤔 GenAI is analyzing your data..."):
            try:
                # Get AI response
                response = ai_assistant.get_ai_response(
                    user_question, df, st.session_state.ai_provider
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": response,
                    "provider": st.session_state.ai_provider,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                st.session_state.current_question = ""
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💬 Analysis History")
        
        for chat in reversed(st.session_state.chat_history[-3:]):  # Show last 3
            # Process the answer to replace newlines with HTML breaks
            processed_answer = chat['answer'].replace('\n', '<br>')
            
            st.markdown(f"""
            <div class="user-message">
                <strong>🙋 You ({chat['timestamp']}):</strong><br>
                {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="ai-message">
                <strong>🤖 {chat['provider']}:</strong><br>
                {processed_answer}
            </div>
            """, unsafe_allow_html=True)

def perform_web_scraping(filtered_df):
    """Perform web scraping of business contact information from filtered data"""
    
    # Check if DataFrame is empty
    if len(filtered_df) == 0:
        st.error("❌ No data to scrape. Please adjust your filters.")
        return
    
    # Find suitable columns for business names
    potential_name_columns = []
    for col in filtered_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['consignee', 'name', 'company', 'business', 'shipper', 'supplier']):
            potential_name_columns.append(col)
    
    if not potential_name_columns:
        st.error("❌ No suitable business name columns found. Need columns like 'Consignee Name', 'Company Name', etc.")
        return
    
    # User selects which column to use for business names
    st.write("🏷️ **Select Business Name Column:**")
    selected_column = st.selectbox(
        "Choose the column containing business names:",
        potential_name_columns,
        help="Select the column that contains the business names you want to research",
        key="business_name_column_selector"  # Added key
    )
    
    # Check unique business count
    unique_businesses = filtered_df[selected_column].dropna().nunique()
    if unique_businesses == 0:
        st.error(f"❌ No business names found in column '{selected_column}'")
        return
    
    st.info(f"📊 Found {unique_businesses} unique businesses to research in '{selected_column}'")
    
    # Research limit selection - FIXED: Added key and session state management
    if 'max_businesses_slider' not in st.session_state:
        st.session_state.max_businesses_slider = min(5, unique_businesses)
    
    max_businesses = st.slider(
        "🎯 Maximum businesses to research:",
        min_value=1,
        max_value=min(20, unique_businesses),  # Limit to 20 max to avoid API costs
        value=st.session_state.max_businesses_slider,
        help="Limit research to avoid high API costs. Each business costs ~$0.02-0.05",
        key="max_businesses_research_slider"  # Added unique key
    )
    
    # Update session state
    st.session_state.max_businesses_slider = max_businesses
    
    # Cost estimation
    estimated_cost = max_businesses * 0.03  # Rough estimate
    st.warning(f"💰 **Estimated API Cost:** ~${estimated_cost:.2f} (approx $0.03 per business)")
    
    # API Configuration check - IMPROVED: Better key detection and validation
    st.write("🔧 **API Configuration:**")
    
    # Force reload environment variables
    import importlib
    import dotenv
    import os  # Ensure os is imported
    importlib.reload(dotenv)
    load_dotenv(override=True)  # Force reload with override
    
    openai_key = os.getenv('OPENAI_API_KEY')
    tavily_key = os.getenv('TAVILY_API_KEY')
    
    # IMPROVED: More robust key validation
    def is_valid_key(key, key_type):
        if not key or key.strip() == '':
            return False, "Key is empty or missing"
        if key.strip() in ['your_openai_key_here', 'your_tavily_key_here', 'sk-...', 'tvly-...']:
            return False, "Key is a placeholder value"
        if key_type == 'openai' and not key.startswith('sk-'):
            return False, "OpenAI key should start with 'sk-'"
        if key_type == 'tavily' and not key.startswith('tvly-'):
            return False, "Tavily key should start with 'tvly-'"
        return True, "Key format is valid"
    
    openai_valid, openai_reason = is_valid_key(openai_key, 'openai')
    tavily_valid, tavily_reason = is_valid_key(tavily_key, 'tavily')
    
    # Display status with detailed feedback
    col_api1, col_api2 = st.columns(2)
    
    with col_api1:
        if openai_valid:
            st.success("✅ OpenAI API Key: Configured")
            # Show partial key for verification
            masked_key = f"{openai_key[:10]}...{openai_key[-4:]}" if len(openai_key) > 14 else f"{openai_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ OpenAI API Key: {openai_reason}")
            st.caption("Add OPENAI_API_KEY to .env file")
    
    with col_api2:
        if tavily_valid:
            st.success("✅ Tavily API Key: Configured")
            # Show partial key for verification
            masked_key = f"{tavily_key[:10]}...{tavily_key[-4:]}" if len(tavily_key) > 14 else f"{tavily_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ Tavily API Key: {tavily_reason}")
            st.caption("Add TAVILY_API_KEY to .env file")
    
    # Show detailed error messages if keys are invalid
    if not openai_valid or not tavily_valid:
        st.warning("⚠️ **Setup Required**: Please configure both API keys before starting research.")
        
        with st.expander("📝 Setup Instructions", expanded=False):
            st.markdown("""
            **To set up API keys:**
            
            1. **Edit your .env file** in the app directory
            2. **Add your API keys:**
               ```
               OPENAI_API_KEY=sk-your_actual_openai_key_here
               TAVILY_API_KEY=tvly-your_actual_tavily_key_here
               ```
            3. **Restart the app**
            4. **Get API keys from:**
               - [OpenAI API Keys](https://platform.openai.com/api-keys)
               - [Tavily API](https://tavily.com)
            
            **Current .env file status:**
            """)
            
            # Show current .env content with better masking
            try:
                with open('.env', 'r') as f:
                    env_content = f.read()
                    lines = env_content.split('\n')
                    display_lines = []
                    for line in lines:
                        if '=' in line and ('API_KEY' in line):
                            key_part, value_part = line.split('=', 1)
                            if value_part and len(value_part) > 10:
                                masked_value = f"{value_part[:6]}...{value_part[-4:]}"
                                display_lines.append(f"{key_part}={masked_value}")
                            else:
                                display_lines.append(line)
                        else:
                            display_lines.append(line)
                    st.code('\n'.join(display_lines))
            except FileNotFoundError:
                st.error(".env file not found in current directory")
    
    # Test API connectivity if both keys are valid
    both_apis_configured = openai_valid and tavily_valid
    api_test_results = None
    
    # Add business researcher path to sys.path early
    business_researcher_path = r"C:\01_Projects\Teakwood_Business\Web_Scraping\business_contact_finder"
    if business_researcher_path not in sys.path:
        sys.path.append(business_researcher_path)
    
    if both_apis_configured:
        st.info("🟢 **Both API keys configured!** You can proceed with web scraping.")
        
        # Optionally test APIs before research
        if st.button("🧪 Test API Connection", help="Test if APIs are working correctly", key="test_api_button"):
            with st.spinner("Testing API connections..."):
                try:
                    # Test import first
                    from streamlit_business_researcher import StreamlitBusinessResearcher
                    
                    # Test API connectivity
                    test_researcher = StreamlitBusinessResearcher()
                    api_ok, api_message = test_researcher.test_apis()
                    
                    if api_ok:
                        st.success(f"✅ API Test Successful: {api_message}")
                        api_test_results = True
                    else:
                        st.error(f"❌ API Test Failed: {api_message}")
                        api_test_results = False
                        
                except Exception as e:
                    st.error(f"❌ API Test Error: {str(e)}")
                    if "OPENAI_API_KEY not found" in str(e):
                        st.error("🔑 OpenAI API key issue detected")
                    elif "TAVILY_API_KEY not found" in str(e):
                        st.error("🔑 Tavily API key issue detected")
                    api_test_results = False
    
    # Confirmation and start button
    st.markdown("---")
    
    # Show research button with proper validation
    button_disabled = not both_apis_configured
    button_help = f"Research {max_businesses} businesses using AI web scraping" if both_apis_configured else "Configure both API keys first"
    
    if st.button(
        f"🚀 Start Research ({max_businesses} businesses)",
        type="primary",
        disabled=button_disabled,
        help=button_help,
        key="start_research_button"  # Added key
    ):
        
        # Double-check API keys before starting
        if not both_apis_configured:
            st.error("❌ Cannot start research: API keys not properly configured")
            return
        
        # Show starting message
        st.info("🔄 Starting business research...")
        
        try:
            # Import the streamlit business researcher
            from streamlit_business_researcher import research_businesses_from_dataframe
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.info("🚀 Initializing research system...")
            progress_bar.progress(10)
            
            with st.spinner("Researching businesses using AI web scraping..."):
                
                # Run the async research function
                async def run_research():
                    return await research_businesses_from_dataframe(
                        df=filtered_df,
                        consignee_column=selected_column,
                        max_businesses=max_businesses
                    )
                
                status_text.info("🔍 Starting business research process...")
                progress_bar.progress(20)
                
                # Execute the async function with better error handling
                try:
                    # Handle async in Streamlit
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    status_text.info("🌐 Connecting to research APIs...")
                    progress_bar.progress(30)
                    
                    results_df, summary, csv_filename = loop.run_until_complete(run_research())
                    loop.close()
                    
                    progress_bar.progress(90)
                    status_text.info("✅ Research completed successfully!")
                    
                except RuntimeError as e:
                    if "asyncio" in str(e).lower():
                        # Fallback for asyncio issues
                        try:
                            status_text.info("🔄 Trying alternative async method...")
                            results_df, summary, csv_filename = asyncio.run(run_research())
                            progress_bar.progress(90)
                            status_text.success("✅ Research completed with fallback method!")
                        except Exception as e2:
                            st.error(f"❌ Error with async execution: {str(e2)}")
                            st.error("Please restart the app and try again.")
                            return
                    else:
                        raise e
                
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Research Error: {error_msg}")
                    
                    # Provide specific guidance based on error type
                    if "API" in error_msg or "key" in error_msg.lower():
                        st.error("🔑 This appears to be an API key issue. Please check your .env file.")
                    elif "billing" in error_msg.lower() or "quota" in error_msg.lower():
                        st.error("💳 This appears to be an API billing/quota issue. Please check your API account.")
                    elif "import" in error_msg.lower():
                        st.error("📁 Module import error. Please ensure all required files are present.")
                    else:
                        st.error("⚠️ Please check your internet connection and API configuration.")
                    
                    # Show full error in expander for debugging
                    with st.expander("🔍 Debug Information", expanded=False):
                        st.code(f"Full error: {error_msg}")
                        st.code(f"Business researcher path: {business_researcher_path}")
                    
                    return
            
            progress_bar.progress(100)
            status_text.success("✅ Research completed!")
            
            # Check if we got valid results
            if results_df is not None and not results_df.empty:
                # Display summary
                st.success(f"🎉 **Research Summary:**")
                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                
                with col_sum1:
                    st.metric("Total Processed", summary['total_processed'])
                with col_sum2:
                    st.metric("Successful", summary['successful'])
                with col_sum3:
                    st.metric("Manual Required", summary['manual_required'])
                with col_sum4:
                    st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
                
                # Display results table
                st.subheader("📈 Research Results")
                st.dataframe(results_df, use_container_width=True, height=400)
                
                # Download results
                st.subheader("📥 Download Research Results")
                
                # Convert results to CSV for download
                csv_data = results_df.to_csv(index=False)
                
                col_down1, col_down2 = st.columns(2)
                with col_down1:
                    st.download_button(
                        label="📄 Download Research Results CSV",
                        data=csv_data,
                        file_name=f"business_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col_down2:
                    # Create combined dataset (original + research results)
                    if 'business_name' in results_df.columns:
                        # Try to merge with original data
                        try:
                            # Create a mapping from research results
                            research_mapping = results_df.set_index('business_name')[['phone', 'email', 'website', 'address']].to_dict('index')
                            
                            # Add research results to original dataframe
                            enhanced_df = filtered_df.copy()
                            enhanced_df['research_phone'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('phone', ''))
                            enhanced_df['research_email'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('email', ''))
                            enhanced_df['research_website'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('website', ''))
                            enhanced_df['research_address'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('address', ''))
                            
                            enhanced_csv = enhanced_df.to_csv(index=False)
                            
                            st.download_button(
                                label="🔗 Download Enhanced Dataset",
                                data=enhanced_csv,
                                file_name=f"enhanced_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help="Original data + research results combined"
                            )
                        except Exception as e:
                            st.warning(f"Could not create enhanced dataset: {e}")
                
                # Success message
                st.balloons()
                st.success(f"🎉 Successfully researched {summary['successful']} businesses!")
                
                if summary['manual_required'] > 0:
                    st.info(f"🔍 {summary['manual_required']} businesses require manual research (check the results for details)")
                
                if summary['billing_errors'] > 0:
                    st.error(f"💳 {summary['billing_errors']} businesses failed due to API billing issues")
            
            elif results_df is not None:
                st.warning("⚠️ Research completed but no results were found. This might be due to:")
                st.info("- API rate limits or quota issues\n- No search results found for the business names\n- Connection problems")
            
            else:
                st.error("❌ No research results obtained. Please check:")
                st.info("1. 🔑 API keys are valid and have sufficient quota\n2. 🌐 Internet connection is stable\n3. 📁 All required files are present")
        
        except ImportError as e:
            st.error(f"❌ Could not import business researcher: {str(e)}")
            st.error("📁 Please ensure the business_contact_finder directory is accessible.")
            st.info(f"Expected path: {business_researcher_path}")
            
            # Check if file exists
            import os
            expected_file = os.path.join(business_researcher_path, "streamlit_business_researcher.py")
            if os.path.exists(expected_file):
                st.info("✅ File exists - might be a Python environment issue")
            else:
                st.error("❌ File not found - check the path")
        
        except Exception as e:
            st.error(f"❌ Unexpected error during research: {str(e)}")
            st.error("🔄 Please restart the app and try again. If the problem persists, check your configuration.")
            
            # Show debug info
            with st.expander("🔍 Debug Information", expanded=False):
                st.code(f"Error details: {str(e)}")
                st.code(f"Error type: {type(e).__name__}")
                import traceback
                st.code(f"Traceback: {traceback.format_exc()}")


def create_enhanced_data_explorer(df, identifier_cols):
    """Create an advanced data explorer with comprehensive business-focused filters"""
    st.subheader("📊 Data Explorer - Advanced Business Intelligence")
    
    # Create comprehensive filter layout
    st.write("**🔍 Filter Your Data:**")
    
    # Dynamically detect available columns
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() if col not in identifier_cols]
    categorical_cols = [col for col in df.select_dtypes(include=['object']).columns.tolist() if col not in identifier_cols]
    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    
    # Row 1: Primary Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Primary categorical filter (non-identifier categories only)
        selected_primary = 'All'
        primary_col = 'None'
        if categorical_cols:
            primary_col = st.selectbox("🏷️ Primary Category Filter", ['None'] + categorical_cols)
            if primary_col != 'None':
                categories = ['All'] + sorted(df[primary_col].dropna().unique().tolist())
                selected_primary = st.selectbox(f"Filter by {primary_col}", categories)
    
    with col2:
        # Identifier filter (dedicated section for HS codes, product codes, etc.)
        selected_identifier = 'All'
        identifier_col = 'None'
        if identifier_cols:
            identifier_col = st.selectbox("🔑 Identifier Filter (HS Codes, Product IDs, etc.)", ['None'] + identifier_cols)
            if identifier_col != 'None':
                identifiers = ['All'] + sorted(df[identifier_col].dropna().unique().tolist())
                selected_identifier = st.selectbox(f"Filter by {identifier_col}", identifiers)
    
    with col3:
        # Date filter
        selected_date_range = None
        date_col = 'None'
        if date_cols:
            date_col = st.selectbox("📅 Date Column", ['None'] + date_cols)
            if date_col != 'None':
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                selected_date_range = st.date_input(
                    f"Date Range for {date_col}",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
    
    # Row 2: Numeric and Search Filters
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # Numeric range filter with safety check
        selected_numeric_range = None
        numeric_col = 'None'
        if numeric_cols:
            numeric_col = st.selectbox("📊 Numeric Column Filter", ['None'] + numeric_cols)
            if numeric_col != 'None':
                # Safety check to prevent slider error
                col_data = df[numeric_col].dropna()
                if len(col_data) > 0:
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    
                    # Check if min and max are the same (which would cause slider error)
                    if min_val == max_val:
                        st.info(f"All values in {numeric_col} are the same: {min_val}")
                        selected_numeric_range = (min_val, max_val)  # No slider needed
                    else:
                        selected_numeric_range = st.slider(
                            f"{numeric_col} Range",
                            min_val, max_val, (min_val, max_val),
                            format="%.2f"
                        )
                else:
                    st.warning(f"No valid data in {numeric_col}")
                    selected_numeric_range = None
    
    with col5:
        # Text search filter
        search_col = st.selectbox("🔍 Search in Column", ['All Columns'] + list(df.columns))
        search_term = st.text_input("Search Term", placeholder="Enter search term...")
    
    with col6:
        # Secondary categorical filter (remaining non-identifier categories)
        selected_secondary = 'All'
        secondary_col = 'None'
        remaining_categorical = [col for col in categorical_cols if col != primary_col] if primary_col != 'None' else categorical_cols
        if remaining_categorical:
            secondary_col = st.selectbox("🎯 Secondary Category Filter", ['None'] + remaining_categorical)
            if secondary_col != 'None':
                categories = ['All'] + sorted(df[secondary_col].dropna().unique().tolist())
                selected_secondary = st.selectbox(f"Filter by {secondary_col}", categories)
    
    # Apply all filters
    filtered_df = df.copy()
    
    # Apply primary categorical filter
    if primary_col != 'None' and selected_primary != 'All':
        filtered_df = filtered_df[filtered_df[primary_col] == selected_primary]
    
    # Apply secondary categorical filter
    if secondary_col != 'None' and selected_secondary != 'All':
        filtered_df = filtered_df[filtered_df[secondary_col] == selected_secondary]
    
    # Apply date filter
    if date_col != 'None' and selected_date_range:
        if len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
            filtered_df = filtered_df[
                (filtered_df[date_col].dt.date >= start_date) & 
                (filtered_df[date_col].dt.date <= end_date)
            ]
    
    # Apply numeric filter with safety check
    if numeric_col != 'None' and selected_numeric_range:
        min_range, max_range = selected_numeric_range
        # Only apply filter if we have a valid range and it's not the same value
        if min_range != max_range:
            filtered_df = filtered_df[
                (filtered_df[numeric_col] >= min_range) & 
                (filtered_df[numeric_col] <= max_range)
            ]
        # If min_range == max_range, we keep all rows (no filtering needed)
    
    # Apply identifier filter
    if identifier_col != 'None' and selected_identifier != 'All':
        filtered_df = filtered_df[filtered_df[identifier_col] == selected_identifier]
    
    # Apply text search
    if search_term:
        if search_col == 'All Columns':
            # Search across all string columns
            mask = filtered_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            filtered_df = filtered_df[mask]
        else:
            # Search in specific column
            filtered_df = filtered_df[
                filtered_df[search_col].astype(str).str.contains(search_term, case=False, na=False)
            ]
    
    # Show filter results summary
    st.markdown("---")
    col_summary1, col_summary2, col_summary3, col_summary4, col_summary5 = st.columns(5)
    
    with col_summary1:
        st.metric("📊 Filtered Records", f"{len(filtered_df):,}")
    
    with col_summary2:
        # Calculate total for first numeric column
        if numeric_cols:
            total_value = filtered_df[numeric_cols[0]].sum() if numeric_cols[0] in filtered_df.columns else 0
            st.metric(f"💰 Total {numeric_cols[0]}", f"{total_value:,.2f}")
        else:
            st.metric("🔢 Unique Values", filtered_df.nunique().sum())
    
    with col_summary3:
        # Count unique values in primary categorical column
        if primary_col != 'None':
            unique_primary = filtered_df[primary_col].nunique()
            st.metric(f"🏷️ Unique {primary_col}", unique_primary)
        else:
            st.metric("📝 Text Columns", len(categorical_cols))
    
    with col_summary4:
        # Count unique identifiers
        if identifier_cols:
            unique_identifiers = filtered_df[identifier_cols[0]].nunique()
            st.metric(f"🔑 Unique {identifier_cols[0]}", unique_identifiers)
        else:
            st.metric("🔢 Numeric Columns", len(numeric_cols))
    
    with col_summary5:
        # Data completeness
        completeness = (1 - filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns))) * 100 if len(filtered_df) > 0 else 0
        st.metric("✅ Data Completeness", f"{completeness:.1f}%")
    
    # Intelligent column reordering based on detected column types
    important_columns = []
    
    # Add date columns first
    important_columns.extend([col for col in date_cols if col in filtered_df.columns])
    
    # Add primary categorical columns
    if primary_col != 'None':
        important_columns.append(primary_col)
    
    # Add identifier columns
    important_columns.extend([col for col in identifier_cols if col in filtered_df.columns and col not in important_columns])
    
    # Add other categorical columns
    important_columns.extend([col for col in categorical_cols if col in filtered_df.columns and col not in important_columns])
    
    # Add numeric columns
    important_columns.extend([col for col in numeric_cols if col in filtered_df.columns and col not in important_columns])
    
    # Add any remaining columns
    other_columns = [col for col in filtered_df.columns if col not in important_columns]
    column_order = important_columns + other_columns
    
    st.subheader(f"📈 Filtered Dataset ({len(filtered_df):,} records)")
    
    # Display controls
    col_display1, col_display2 = st.columns(2)
    with col_display1:
        display_rows = st.slider("Rows to display:", 10, min(500, len(filtered_df)), min(100, len(filtered_df)))
    with col_display2:
        st.write(f"Showing {min(display_rows, len(filtered_df))} of {len(filtered_df)} total rows")
    
    # Display data
    st.dataframe(
        filtered_df[column_order].head(display_rows),
        use_container_width=True,
        height=500
    )
    
    # Advanced Analytics Summary
    if len(filtered_df) > 0:
        st.markdown("---")
        st.subheader("📈 Advanced Analytics Summary")
        
        col_analytics1, col_analytics2 = st.columns(2)
        
        with col_analytics1:
            # Top values analysis
            if primary_col != 'None':
                st.write(f"**🏆 Top 5 {primary_col} by Frequency:**")
                top_values = filtered_df[primary_col].value_counts().head(5)
                for value, count in top_values.items():
                    percentage = (count / len(filtered_df)) * 100
                    st.write(f"• **{value}**: {count} records ({percentage:.1f}%)")
            
            # Identifier analysis
            if identifier_cols:
                st.write(f"**🔑 {identifier_cols[0]} Analysis:**")
                id_stats = filtered_df[identifier_cols[0]].value_counts().head(5)
                for id_val, count in id_stats.items():
                    st.write(f"• **{id_val}**: {count} occurrences")
        
        with col_analytics2:
            # Numeric analysis
            if numeric_cols:
                st.write(f"**📊 {numeric_cols[0]} Statistics:**")
                col_data = filtered_df[numeric_cols[0]].dropna()
                if len(col_data) > 0:
                    st.write(f"• Mean: {col_data.mean():.2f}")
                    st.write(f"• Median: {col_data.median():.2f}")
                    st.write(f"• Std Dev: {col_data.std():.2f}")
                    st.write(f"• Min: {col_data.min():.2f}")
                    st.write(f"• Max: {col_data.max():.2f}")
            
            # Data quality insights
            st.write("**🔍 Data Quality Insights:**")
            missing_data = filtered_df.isnull().sum()
            cols_with_missing = missing_data[missing_data > 0].head(3)
            if len(cols_with_missing) > 0:
                st.write("Columns with missing data:")
                for col, missing in cols_with_missing.items():
                    pct = (missing / len(filtered_df)) * 100
                    st.write(f"• {col}: {missing} ({pct:.1f}%)")
            else:
                st.write("✅ No missing data in top columns")
    
    # Download options
    st.markdown("---")
    st.subheader("📥 Download & Research Options")
    col_download1, col_download2, col_download3 = st.columns(3)
    
    with col_download1:
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_download2:
        # Download analytics summary
        if len(filtered_df) > 0:
            summary_text = f"📊 DATA ANALYTICS SUMMARY\n"
            summary_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            summary_text += "="*50 + "\n\n"
            
            summary_text += f"OVERVIEW:\n"
            summary_text += f"Total Records: {len(filtered_df):,}\n"
            summary_text += f"Data Completeness: {completeness:.1f}%\n"
            
            if numeric_cols and len(filtered_df[numeric_cols[0]].dropna()) > 0:
                col_data = filtered_df[numeric_cols[0]].dropna()
                summary_text += f"\n{numeric_cols[0]} STATISTICS:\n"
                summary_text += f"Mean: {col_data.mean():.2f}\n"
                summary_text += f"Median: {col_data.median():.2f}\n"
                summary_text += f"Std Dev: {col_data.std():.2f}\n"
            
            if primary_col != 'None':
                top_values = filtered_df[primary_col].value_counts().head(5)
                summary_text += f"\nTOP {primary_col.upper()} VALUES:\n"
                for value, count in top_values.items():
                    percentage = (count / len(filtered_df)) * 100
                    summary_text += f"  {value}: {count} ({percentage:.1f}%)\n"
            
            st.download_button(
                label="📄 Download Analytics Summary",
                data=summary_text,
                file_name=f"analytics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col_download3:
        # Web Scraping button
        st.write("**🔍 Business Research:**")
        if st.button("🌐 Web Scrape Filtered Data", help="Research business contact information using AI"):
            perform_web_scraping(filtered_df)

def create_quick_viz(df, identifier_cols):
    """Create quick visualizations for any dataset"""
    st.subheader("📈 Quick Visualizations")
    
    # Separate true numeric columns from identifiers
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() if col not in identifier_cols]
    categorical_cols = [col for col in df.select_dtypes(include=['object']).columns.tolist() if col not in identifier_cols]
    all_categorical = categorical_cols + identifier_cols
    all_columns = list(df.columns)  # All columns for X vs Y plotting
    
    if not numeric_cols and not all_categorical:
        st.warning("No suitable columns found for visualization.")
        return
    
    # Create three sections: Individual Columns, X vs Y Plotting, and Correlation
    st.markdown("### 📊 Individual Column Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if all_categorical:
            st.write("**Top Values Distribution**")
            selected_cat_col = st.selectbox("Select categorical/identifier column:", all_categorical, key="cat_viz")
            
            if selected_cat_col:
                value_counts = df[selected_cat_col].value_counts().head(15)
                fig = px.bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    title=f"Top 15 values in {selected_cat_col}",
                    labels={'x': 'Count', 'y': selected_cat_col}
                )
                if selected_cat_col in identifier_cols:
                    fig.update_layout(title=f"Top 15 {selected_cat_col} (Identifier Column)")
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if numeric_cols:
            st.write("**Numeric Distribution**")
            selected_num_col = st.selectbox("Select numeric column:", numeric_cols, key="num_viz")
            
            if selected_num_col:
                fig = px.histogram(
                    df,
                    x=selected_num_col,
                    title=f"Distribution of {selected_num_col}",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("**📊 Available Columns**")
            st.write(f"Total columns: {len(df.columns)}")
            st.write(f"Categorical columns: {len(all_categorical)}")
            st.write(f"Identifier columns: {len(identifier_cols)}")
            if all_categorical:
                st.write("Use the X vs Y plotting section below to create custom visualizations.")
    
    # NEW FEATURE: X vs Y Plotting
    st.markdown("---")
    st.markdown("### 🎯 X vs Y Custom Plotting")
    st.write("💡 **Select any two columns to create custom visualizations**")
    
    col_x, col_y, col_chart = st.columns([1, 1, 1])
    
    with col_x:
        x_column = st.selectbox("Select X-axis column:", all_columns, key="x_axis")
    
    with col_y:
        # Filter out the X column from Y options to avoid same column selection
        y_options = [col for col in all_columns if col != x_column]
        y_column = st.selectbox("Select Y-axis column:", y_options, key="y_axis")
    
    with col_chart:
        # Auto-detect best chart type but allow user override
        x_is_numeric = x_column in numeric_cols
        y_is_numeric = y_column in numeric_cols
        
        if x_is_numeric and y_is_numeric:
            default_chart = "Scatter Plot"
            chart_options = ["Scatter Plot", "Line Chart", "Bar Chart"]
        elif x_is_numeric or y_is_numeric:
            default_chart = "Bar Chart"
            chart_options = ["Bar Chart", "Box Plot", "Violin Plot"]
        else:
            default_chart = "Bar Chart"
            chart_options = ["Bar Chart", "Stacked Bar", "Heatmap"]
        
        chart_type = st.selectbox(
            "Chart type:", 
            chart_options, 
            index=chart_options.index(default_chart),
            key="chart_type"
        )
    
    # Generate the X vs Y plot
    if x_column and y_column:
        try:
            st.markdown(f"#### {chart_type}: {x_column} vs {y_column}")
            
            # Create the appropriate plot based on selection
            if chart_type == "Scatter Plot":
                fig = px.scatter(
                    df, 
                    x=x_column, 
                    y=y_column,
                    title=f"{x_column} vs {y_column}",
                    hover_data=[x_column, y_column]
                )
            
            elif chart_type == "Line Chart":
                # Sort by x-axis for better line visualization
                df_sorted = df.sort_values(x_column)
                fig = px.line(
                    df_sorted, 
                    x=x_column, 
                    y=y_column,
                    title=f"{x_column} vs {y_column} (Line)"
                )
            
            elif chart_type == "Bar Chart":
                if x_is_numeric and y_is_numeric:
                    # For numeric vs numeric, create aggregated bar chart
                    df_agg = df.groupby(x_column)[y_column].mean().reset_index()
                    fig = px.bar(
                        df_agg.head(20),  # Limit to top 20 for readability
                        x=x_column, 
                        y=y_column,
                        title=f"Average {y_column} by {x_column} (Top 20)"
                    )
                else:
                    # For categorical data
                    if not x_is_numeric:
                        # Group by X (categorical) and aggregate Y
                        if y_is_numeric:
                            df_agg = df.groupby(x_column)[y_column].sum().reset_index()
                            df_agg = df_agg.sort_values(y_column, ascending=False).head(15)
                        else:
                            df_agg = df[x_column].value_counts().head(15).reset_index()
                            df_agg.columns = [x_column, 'count']
                            y_column = 'count'
                    else:
                        # X is numeric, Y is categorical
                        df_agg = df.groupby(y_column)[x_column].sum().reset_index()
                        df_agg = df_agg.sort_values(x_column, ascending=False).head(15)
                        # Swap for proper visualization
                        x_column, y_column = y_column, x_column
                    
                    fig = px.bar(
                        df_agg,
                        x=x_column, 
                        y=y_column,
                        title=f"{y_column} by {x_column}"
                    )
            
            elif chart_type == "Box Plot":
                if x_is_numeric and not y_is_numeric:
                    fig = px.box(
                        df, 
                        y=x_column, 
                        x=y_column,
                        title=f"{x_column} distribution by {y_column}"
                    )
                else:
                    fig = px.box(
                        df, 
                        x=x_column, 
                        y=y_column,
                        title=f"{y_column} distribution by {x_column}"
                    )
            
            elif chart_type == "Violin Plot":
                if x_is_numeric and not y_is_numeric:
                    fig = px.violin(
                        df, 
                        y=x_column, 
                        x=y_column,
                        title=f"{x_column} distribution by {y_column}"
                    )
                else:
                    fig = px.violin(
                        df, 
                        x=x_column, 
                        y=y_column,
                        title=f"{y_column} distribution by {x_column}"
                    )
            
            elif chart_type == "Stacked Bar":
                # Create crosstab for stacked bar
                cross_tab = pd.crosstab(df[x_column], df[y_column])
                fig = px.bar(
                    cross_tab.head(10), 
                    title=f"{x_column} vs {y_column} (Stacked)",
                    barmode='stack'
                )
            
            elif chart_type == "Heatmap":
                # Create crosstab for heatmap
                cross_tab = pd.crosstab(df[x_column], df[y_column])
                fig = px.imshow(
                    cross_tab,
                    title=f"Heatmap: {x_column} vs {y_column}",
                    aspect="auto"
                )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Show some basic stats about the relationship
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.write(f"**{x_column} Info:**")
                if x_is_numeric:
                    st.write(f"Range: {df[x_column].min():.2f} to {df[x_column].max():.2f}")
                    st.write(f"Mean: {df[x_column].mean():.2f}")
                else:
                    st.write(f"Unique values: {df[x_column].nunique()}")
                    st.write(f"Most common: {df[x_column].mode().iloc[0] if not df[x_column].mode().empty else 'N/A'}")
            
            with col_stats2:
                st.write(f"**{y_column} Info:**")
                if y_is_numeric:
                    st.write(f"Range: {df[y_column].min():.2f} to {df[y_column].max():.2f}")
                    st.write(f"Mean: {df[y_column].mean():.2f}")
                else:
                    st.write(f"Unique values: {df[y_column].nunique()}")
                    st.write(f"Most common: {df[y_column].mode().iloc[0] if not df[y_column].mode().empty else 'N/A'}")
        
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.info("Try selecting different columns or chart types.")
    
    # Correlation heatmap if multiple numeric columns
    if len(numeric_cols) > 1:
        st.markdown("---")
        st.markdown("### 🔥 Correlation Analysis")
        st.write("**Correlation Matrix (Numeric Columns Only)**")
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix of Numeric Columns",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show strongest correlations
        st.write("**Strongest Correlations:**")
        # Get correlation pairs (excluding diagonal)
        correlation_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                correlation_pairs.append((col1, col2, abs(corr_val), corr_val))
        
        # Sort by absolute correlation
        correlation_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Display top 5 correlations
        for i, (col1, col2, abs_corr, corr) in enumerate(correlation_pairs[:5]):
            direction = "🔴 Strong positive" if corr > 0.7 else "🔵 Strong negative" if corr < -0.7 else "🟫 Moderate positive" if corr > 0.3 else "🟪 Moderate negative" if corr < -0.3 else "🟫 Weak"
            st.write(f"{i+1}. **{col1}** vs **{col2}**: {corr:.3f} ({direction})")

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🤖 AI-Powered Timber Business Data Analyzer</h1>', unsafe_allow_html=True)
    
    # File upload only
    st.sidebar.title("📂 Upload Your Data")
    
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    df = None
    identifier_cols = []
    
    if uploaded_file is not None:
        df, identifier_cols = smart_csv_loader(uploaded_file)
        if df is not None:
            st.sidebar.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            if identifier_cols:
                st.sidebar.info(f"🔑 Detected {len(identifier_cols)} identifier column(s)")
    
    if df is None:
        st.warning("Please upload a CSV file to begin analysis.")
        st.markdown("""
        ### 🚀 What this tool does:
        - **Upload any CSV file** and instantly start analyzing it
        - **Smart identifier detection** - HS codes, product codes, etc. are treated correctly
        - **Chat with your data** using GenAI powered by multiple providers
        - **Get automatic insights** about patterns, trends, and anomalies  
        - **Generate actual plots and visualizations** when you ask for them
        - **No hardcoding** - works with any dataset automatically
        
        ### 🔧 Setup API Keys (Optional):
        Create a `.env` file in this directory with:
        ```
        GROQ_API_KEY=your_groq_key_here
        OPENAI_API_KEY=your_openai_key_here
        ANTHROPIC_API_KEY=your_anthropic_key_here
        ```
        
        ### 💡 Example questions you can ask:
        - "What are the top HS codes?" (analyze identifiers)
        - "Analyze the distribution of values" (for numeric data)
        - "What correlations exist between columns?"
        - "Describe the distribution patterns"
        - "Analyze trends over time"
        """)
        return
    
    # Initialize AI assistant
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = GenericDataAI()
    
    # Set identifier columns in AI assistant
    st.session_state.ai_assistant.identifier_columns = identifier_cols
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 AI Chat",
        "📊 Data Overview", 
        "📈 Quick Viz",
        "📊 Data Explorer"
    ])
    
    with tab1:
        create_ai_chat_section(df, identifier_cols)
    
    with tab2:
        create_data_overview(df, st.session_state.ai_assistant, identifier_cols)
    
    with tab3:
        create_quick_viz(df, identifier_cols)
    
    with tab4:
        create_data_explorer(df, identifier_cols)

if __name__ == "__main__":
    main()
