"""
Web Scraping Module for Business Contact Research
Separated from main application for better organization and debugging
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import asyncio
import importlib
import dotenv
from dotenv import load_dotenv


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
