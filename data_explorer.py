import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def create_data_explorer(df, identifier_cols):
    """
    Simplified Data Explorer with clean interface
    - Primary Filter (categorical columns)
    - Secondary Filter (categorical columns) 
    - Text Search
    """
    st.subheader("📊 Data Explorer")
    
    if df is None or len(df) == 0:
        st.warning("No data available to explore.")
        return
    
    # Get all categorical columns (including identifiers)
    categorical_cols = []
    
    # Add object type columns
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.extend(object_cols)
    
    # Add identifier columns
    if identifier_cols:
        categorical_cols.extend([col for col in identifier_cols if col not in categorical_cols])
    
    # Remove duplicates and sort
    categorical_cols = sorted(list(set(categorical_cols)))
    
    if not categorical_cols:
        st.info("No categorical columns found for filtering.")
        st.subheader("📈 Raw Dataset")
        st.dataframe(df.head(100), use_container_width=True, height=500)
        return
    
    # Simple 2-column layout for filters
    st.write("**🔍 Filter Your Data:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primary Filter**")
        primary_filter_col = st.selectbox(
            "Select Column", 
            ['None'] + categorical_cols,
            key="primary_filter_column"
        )
        
        primary_filter_value = 'All'
        if primary_filter_col != 'None':
            unique_values = ['All'] + sorted([str(val) for val in df[primary_filter_col].dropna().unique()])
            primary_filter_value = st.selectbox(
                f"Filter by {primary_filter_col}",
                unique_values,
                key="primary_filter_value"
            )
        
        # Text search for primary filter
        if primary_filter_col != 'None':
            primary_search = st.text_input(
                f"Search in {primary_filter_col}",
                placeholder="Enter search term...",
                key="primary_search"
            )
        else:
            primary_search = ""
    
    with col2:
        st.write("**Secondary Filter**")
        # Remove primary filter column from secondary options
        secondary_options = [col for col in categorical_cols if col != primary_filter_col]
        
        secondary_filter_col = st.selectbox(
            "Select Column",
            ['None'] + secondary_options,
            key="secondary_filter_column"
        )
        
        secondary_filter_value = 'All'
        if secondary_filter_col != 'None':
            unique_values = ['All'] + sorted([str(val) for val in df[secondary_filter_col].dropna().unique()])
            secondary_filter_value = st.selectbox(
                f"Filter by {secondary_filter_col}",
                unique_values,
                key="secondary_filter_value"
            )
        
        # Text search for secondary filter
        if secondary_filter_col != 'None':
            secondary_search = st.text_input(
                f"Search in {secondary_filter_col}",
                placeholder="Enter search term...",
                key="secondary_search"
            )
        else:
            secondary_search = ""
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply primary filter
    if primary_filter_col != 'None':
        if primary_filter_value != 'All':
            filtered_df = filtered_df[filtered_df[primary_filter_col].astype(str) == primary_filter_value]
        
        if primary_search:
            mask = filtered_df[primary_filter_col].astype(str).str.contains(
                primary_search, case=False, na=False
            )
            filtered_df = filtered_df[mask]
    
    # Apply secondary filter
    if secondary_filter_col != 'None':
        if secondary_filter_value != 'All':
            filtered_df = filtered_df[filtered_df[secondary_filter_col].astype(str) == secondary_filter_value]
        
        if secondary_search:
            mask = filtered_df[secondary_filter_col].astype(str).str.contains(
                secondary_search, case=False, na=False
            )
            filtered_df = filtered_df[mask]
    
    # Display results summary
    st.markdown("---")
    
    col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
    
    with col_summary1:
        st.metric("📊 Total Records", f"{len(filtered_df):,}")
    
    with col_summary2:
        original_count = len(df)
        if original_count > 0:
            percentage = (len(filtered_df) / original_count) * 100
            st.metric("📈 % of Total", f"{percentage:.1f}%")
        else:
            st.metric("📈 % of Total", "0%")
    
    with col_summary3:
        if primary_filter_col != 'None' and primary_filter_col in filtered_df.columns:
            unique_primary = filtered_df[primary_filter_col].nunique()
            st.metric(f"🔢 Unique {primary_filter_col[:10]}...", unique_primary)
        else:
            st.metric("🔢 Columns", len(filtered_df.columns))
    
    with col_summary4:
        if secondary_filter_col != 'None' and secondary_filter_col in filtered_df.columns:
            unique_secondary = filtered_df[secondary_filter_col].nunique()
            st.metric(f"🔢 Unique {secondary_filter_col[:10]}...", unique_secondary)
        else:
            completeness = (1 - filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns))) * 100 if len(filtered_df) > 0 else 0
            st.metric("✅ Completeness", f"{completeness:.1f}%")
    
    # Display filtered data
    st.subheader(f"📈 Filtered Dataset ({len(filtered_df):,} records)")
    
    if len(filtered_df) == 0:
        st.warning("No records match your filter criteria. Try adjusting your filters.")
        return
    
    # Display controls with safety check for small datasets
    col_display1, col_display2 = st.columns(2)
    with col_display1:
        # Safety check to prevent slider error when filtered data is small
        total_rows = len(filtered_df)
        if total_rows <= 10:
            # For very small datasets, just show all rows
            display_rows = total_rows
            st.info(f"Showing all {total_rows} rows (dataset is small)")
        else:
            # For larger datasets, provide slider
            min_display = min(10, total_rows)
            max_display = min(500, total_rows)
            default_display = min(100, total_rows)
            
            # Ensure min_value < max_value for slider
            if min_display >= max_display:
                display_rows = max_display
                st.info(f"Showing all {max_display} rows")
            else:
                display_rows = st.slider(
                    "Rows to display:", 
                    min_display, 
                    max_display, 
                    default_display
                )
    with col_display2:
        st.write(f"Showing {min(display_rows, len(filtered_df))} of {len(filtered_df)} total rows")
    
    # Smart column ordering - put filtered columns first
    columns_order = []
    if primary_filter_col != 'None':
        columns_order.append(primary_filter_col)
    if secondary_filter_col != 'None' and secondary_filter_col not in columns_order:
        columns_order.append(secondary_filter_col)
    
    # Add remaining columns
    remaining_cols = [col for col in filtered_df.columns if col not in columns_order]
    columns_order.extend(remaining_cols)
    
    # Display data
    st.dataframe(
        filtered_df[columns_order].head(display_rows),
        use_container_width=True,
        height=500
    )
    
    # Quick insights
    if len(filtered_df) > 0:
        st.markdown("---")
        st.subheader("💡 Quick Insights")
        
        col_insights1, col_insights2 = st.columns(2)
        
        with col_insights1:
            if primary_filter_col != 'None':
                st.write(f"**Top 5 {primary_filter_col}:**")
                top_values = filtered_df[primary_filter_col].value_counts().head(5)
                for value, count in top_values.items():
                    percentage = (count / len(filtered_df)) * 100
                    st.write(f"• **{value}**: {count} ({percentage:.1f}%)")
        
        with col_insights2:
            if secondary_filter_col != 'None':
                st.write(f"**Top 5 {secondary_filter_col}:**")
                top_values = filtered_df[secondary_filter_col].value_counts().head(5)
                for value, count in top_values.items():
                    percentage = (count / len(filtered_df)) * 100
                    st.write(f"• **{value}**: {count} ({percentage:.1f}%)")
            else:
                # Show data quality info instead
                st.write("**Data Quality:**")
                missing_data = filtered_df.isnull().sum()
                cols_with_missing = missing_data[missing_data > 0].head(3)
                if len(cols_with_missing) > 0:
                    for col, missing in cols_with_missing.items():
                        pct = (missing / len(filtered_df)) * 100
                        st.write(f"• {col}: {missing} missing ({pct:.1f}%)")
                else:
                    st.write("✅ No missing data found")
    
    # Download filtered data
    st.markdown("---")
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        if len(filtered_df) > 0:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Filtered Data",
                data=csv,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col_download2:
        # Web scraping option
        if len(filtered_df) > 0:
            if st.button("🌐 Research Business Contacts", help="Use AI to find business contact information"):
                perform_web_scraping(filtered_df)

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
        key="business_name_column_selector_explorer"
    )
    
    # Check unique business count
    unique_businesses = filtered_df[selected_column].dropna().nunique()
    if unique_businesses == 0:
        st.error(f"❌ No business names found in column '{selected_column}'")
        return
    
    st.info(f"📊 Found {unique_businesses} unique businesses to research in '{selected_column}'")
    
    # Research limit selection
    max_businesses = st.slider(
        "🎯 Maximum businesses to research:",
        min_value=1,
        max_value=min(20, unique_businesses),
        value=min(5, unique_businesses),
        help="Limit research to avoid high API costs. Each business costs ~$0.02-0.05",
        key="max_businesses_research_slider_explorer"
    )
    
    # Cost estimation
    estimated_cost = max_businesses * 0.03
    st.warning(f"💰 **Estimated API Cost:** ~${estimated_cost:.2f} (approx $0.03 per business)")
    
    # API Configuration check
    st.write("🔧 **API Configuration:**")
    
    # Helper function to get environment variables (works for both local .env and Streamlit secrets)
    def get_env_var(key, default=None):
        """Get environment variable from .env file (local) or Streamlit secrets (cloud)"""
        # First try regular environment variables (from .env or system)
        import os
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
    
    openai_key = get_env_var('OPENAI_API_KEY')
    tavily_key = get_env_var('TAVILY_API_KEY')
    
    # Simple key validation
    def is_valid_key(key, key_type):
        if not key or key.strip() == '':
            return False, "Key is empty or missing"
        if key_type == 'openai' and not key.startswith('sk-'):
            return False, "OpenAI key should start with 'sk-'"
        if key_type == 'tavily' and not key.startswith('tvly-'):
            return False, "Tavily key should start with 'tvly-'"
        return True, "Key format is valid"
    
    openai_valid, openai_reason = is_valid_key(openai_key, 'openai')
    tavily_valid, tavily_reason = is_valid_key(tavily_key, 'tavily')
    
    # Display status
    col_api1, col_api2 = st.columns(2)
    
    with col_api1:
        if openai_valid:
            st.success("✅ OpenAI API Key: Configured")
        else:
            st.error(f"❌ OpenAI API Key: {openai_reason}")
    
    with col_api2:
        if tavily_valid:
            st.success("✅ Tavily API Key: Configured")
        else:
            st.error(f"❌ Tavily API Key: {tavily_reason}")
    
    both_apis_configured = openai_valid and tavily_valid
    
    if not both_apis_configured:
        st.warning("⚠️ **Setup Required**: Please configure both API keys in your .env file before starting research.")
        return
    
    # Start research button
    if st.button(
        f"🚀 Start Research ({max_businesses} businesses)",
        type="primary",
        key="start_research_button_explorer"
    ):
        st.info("🔄 Starting business research...")
        
        try:
            # Add business researcher path to sys.path
            import sys
            business_researcher_path = r"C:\01_Projects\Teakwood_Business\Web_Scraping\business_contact_finder"
            if business_researcher_path not in sys.path:
                sys.path.append(business_researcher_path)
            
            # Import the streamlit business researcher
            from streamlit_business_researcher import research_businesses_from_dataframe
            import asyncio
            
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
                
                try:
                    # Handle async in Streamlit
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    status_text.info("🌐 Connecting to research APIs...")
                    progress_bar.progress(30)
                    
                    results_df, summary, csv_filename = loop.run_until_complete(run_research())
                    loop.close()
                    
                    progress_bar.progress(90)
                    status_text.success("✅ Research completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Research Error: {str(e)}")
                    return
            
            progress_bar.progress(100)
            status_text.success("✅ Research completed!")
            
            # Display results
            if results_df is not None and not results_df.empty:
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
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Research Results CSV",
                    data=csv_data,
                    file_name=f"business_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.balloons()
                st.success(f"🎉 Successfully researched {summary['successful']} businesses!")
            
            else:
                st.warning("⚠️ Research completed but no results were found.")
        
        except ImportError as e:
            st.error(f"❌ Could not import business researcher: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error during research: {str(e)}")
