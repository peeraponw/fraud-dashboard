import streamlit as st
import pandas as pd
import io

# Set page config
st.set_page_config(layout="wide", page_title="Anomaly Detection App")

# Sidebar
st.sidebar.title("Data Input")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'label_column' not in st.session_state:
    st.session_state.label_column = None

# Load data when file is uploaded
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        
        # Column selector for label
        columns = ['None'] + list(st.session_state.df.columns)
        st.session_state.label_column = st.sidebar.selectbox("Select label column", columns)
        
        if st.session_state.label_column == 'None':
            st.session_state.label_column = None
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Main content
tab1, tab2 = st.tabs(["Data View", "Dummy Tab"])

with tab1:
    st.title("Anomaly Detection Dashboard")
    
    if st.session_state.df is not None:
        # Create columns for key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(st.session_state.df))
        with col2:
            st.metric("Total Columns", len(st.session_state.df.columns))
        with col3:
            st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
        with col4:
            if st.session_state.label_column:
                st.metric("Unique Labels", st.session_state.df[st.session_state.label_column].nunique())
            else:
                st.metric("Unique Labels", "N/A")
        
        # Create two columns for filters and data display
        filter_col, data_col = st.columns([1, 3])
        
        with filter_col:
            st.subheader("Filters")
            # Add filter options for each column
            filters = {}
            for column in st.session_state.df.columns:
                if st.session_state.df[column].dtype == 'object':
                    filters[column] = st.multiselect(f"Filter {column}", options=st.session_state.df[column].unique())
                elif st.session_state.df[column].dtype in ['int64', 'float64']:
                    min_val, max_val = st.session_state.df[column].min(), st.session_state.df[column].max()
                    filters[column] = st.slider(f"Filter {column}", min_val, max_val, (min_val, max_val))
        
        with data_col:
            # Apply filters
            filtered_df = st.session_state.df.copy()
            for column, filter_value in filters.items():
                if filter_value:
                    if isinstance(filter_value, tuple):  # For numeric range
                        filtered_df = filtered_df[(filtered_df[column] >= filter_value[0]) & (filtered_df[column] <= filter_value[1])]
                    else:  # For categorical
                        filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
            
            # Display filtered dataframe
            st.dataframe(filtered_df, hide_index=True)
            
            # Download button for filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv",
            )

with tab2:
    st.title("Dummy Tab")
    st.write("This is a placeholder for the second tab. Content will be added later.")

