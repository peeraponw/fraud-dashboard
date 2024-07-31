import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from datetime import datetime, time, timedelta

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
        
        # Convert Tx Datetime to datetime format
        if 'Tx Datetime' in st.session_state.df.columns:
            st.session_state.df['Tx Datetime'] = pd.to_datetime(st.session_state.df['Tx Datetime'])
        
        # Column selector for label
        columns = ['None'] + list(st.session_state.df.columns)
        st.session_state.label_column = st.sidebar.selectbox("Select label column", columns)
        
        if st.session_state.label_column == 'None':
            st.session_state.label_column = None
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Main content
tab1, tab2 = st.tabs(["Data View", "Rules View"])

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
                if column in ['Remark']:
                    continue
                if st.session_state.df[column].dtype == 'datetime64[ns]':
                    start_date = st.date_input("Start Date", st.session_state.df[column].min().date())
                    start_time = st.slider('Dataset Start time', st.session_state.df[column].min().time(), format="HH:mm")
                    end_date = st.date_input("Dataset End Date", st.session_state.df[column].max().date())   
                    end_time = st.slider('End time', value=(st.session_state.df[column].max().time()), format="HH:mm")
                    start_datetime = datetime.combine(start_date, start_time)
                    end_datetime = datetime.combine(end_date, end_time)
                    filters[column] = (start_datetime, end_datetime)
                    
                    # filter hour range
                    day_start_time = st.slider("Day Start Time", min_value=time(0,0), max_value=time(23,59), value=time(0,0), format ="HH:mm")
                    day_end_time = st.slider("Day End Time", min_value=time(0,0), max_value=time(23,59), value=time(23,59), format ="HH:mm")
                    filters[column + '_time_range'] = (day_start_time, day_end_time)           
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
                    if isinstance(filter_value, tuple) and column.endswith('_time_range'):
                        column = column.replace('_time_range', '')
                        if filter_value[0] <= filter_value[1]: # if start time is less than end time
                            filtered_df = filtered_df[filtered_df[column].dt.time.between(filter_value[0], filter_value[1])]
                        else: # if the time range cover across midnight
                            filtered_df = filtered_df[(filtered_df[column].dt.time >= filter_value[0]) | (filtered_df[column].dt.time <= filter_value[1])]
                    elif isinstance(filter_value, tuple):  # For numeric range
                        filtered_df = filtered_df[(filtered_df[column] >= filter_value[0]) & (filtered_df[column] <= filter_value[1])]
                    else:  # For categorical
                        filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
            
            # Display filtered dataframe
            st.dataframe(filtered_df, hide_index=True, height=1600, use_container_width=True)
            
            # Download button for filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv",
            )

with tab2:
    # Date input
    date_col, _ = st.columns([1, 7])
    with date_col:
        last_date = st.date_input("Select the last date", datetime.now().date())
    # Display rules table
    st.title("Rules Dashboard")
    
    # # # 
    # # # Collections of rules
    # # # 
    def rule_10pm_to_4am(df):
        df['hour'] = df['Tx Datetime'].dt.hour
        return df[(df['hour'] >= 22) | (df['hour'] < 4)]

    def rule_rapid_card_use(df):
        return df.groupby(['Payer IP']).filter(lambda x: len(x['Card Number'].unique()) > 1)

    def rule_excessive_ip_switching(df):
        return df.groupby(['Card Number']).filter(lambda x: x['Payer IP'].nunique() > 3)

    def rule_escalating_transaction_amounts(df):
        return df.groupby(['Holder Name']).filter(lambda x: (x['Amount'].diff() > 0).sum() > 2)
    
    def rule_ml(df, model_path):
        df = df.copy()
        df['Tx Year'] = df['Tx Datetime'].dt.year
        df['Tx Month'] = df['Tx Datetime'].dt.month
        df['Tx Day'] = df['Tx Datetime'].dt.day
        df['Tx Hour'] = df['Tx Datetime'].dt.hour
        df['Tx Minute'] = df['Tx Datetime'].dt.minute
        df['Tx Second'] = df['Tx Datetime'].dt.second
        datetime_temp = df['Tx Datetime']
        df.drop(['Tx Datetime', 'Holder Name', 'Card Number', 'Remark', 'Payer IP'], axis=1, inplace=True)
        pipeline = joblib.load(model_path)
        y_pred = pipeline.predict(df)
        df['Tx Datetime'] = datetime_temp
        df['prediction'] = y_pred
        return df.loc[df['prediction'] == 1]
    
    # # # 
    # # # Apply rules
    # # # 
    # Helper function to apply a rule and calculate metrics
    def apply_rule(df, rule_func, input_date):
        rule_df = rule_func(df)
        today_count = rule_df[rule_df['Tx Datetime'].dt.date == input_date].shape[0]
        four_week_count = rule_df.groupby(rule_df['Tx Datetime'].dt.date).size().reindex(pd.date_range(end=input_date, periods=28), fill_value=0).tolist()
        last_event = rule_df['Tx Datetime'].max()
        return today_count, four_week_count, last_event

    # Apply rules and calculate metrics
    def apply_rules(df, input_date):
        rules = []

        rule_funcs = [
            ("10 PM to 4 AM", rule_10pm_to_4am),
            ("Rapid Card Use", rule_rapid_card_use),
            ("Excessive IP Switching", rule_excessive_ip_switching),
            ("Escalating Transaction Amounts", rule_escalating_transaction_amounts),
            ("Random Forest Classifier", lambda df: rule_ml(df, "rf_model.pkl")),
            ("Xgboost Classifier", lambda df: rule_ml(df, "xgb_model.pkl"))
        ]

        for rule_name, rule_func in rule_funcs:
            today_count, four_week_count, last_event = apply_rule(df, rule_func, input_date)
            rules.append({
                "Rule Name": rule_name,
                "Today Count": today_count,
                "4-week Count": four_week_count,
                "Last Event": last_event
            })

        return rules
    if st.session_state.df is not None:
        # Convert rules to DataFrame
        rules = apply_rules(st.session_state.df, last_date)
        rules_df = pd.DataFrame(rules)
        # st.dataframe(rule_excessive_ip_switching(st.session_state.df), hide_index=True, height=200, use_container_width=True)
        # st.dataframe(rules_df, hide_index=True, height=200, use_container_width=True)
        
        # Display rules table header
        header_col1, header_col2, header_col3, header_col4 = st.columns([2, 1, 4, 2])
        with header_col1:
            st.subheader("Rule Name")
        with header_col2:
            st.subheader("Today")
        with header_col3:
            st.subheader("Past 4-Week Activity")
        with header_col4:
            st.subheader("Last Detection")
        for index, row in rules_df.iterrows():
            col1, col2, col3, col4 = st.columns([2, 1, 4, 2])
            
            with col1:
                st.write(row['Rule Name'])
            with col2:
                st.subheader(row['Today Count'])
            with col3:
                chart_data = pd.DataFrame({'date': pd.date_range(end=last_date, periods=28), 'count': row['4-week Count']})
                plt.figure(figsize=(8, 2))
                sns.barplot(x='date', y='count', data=chart_data)
                plt.xticks(rotation=45)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
                plt.tight_layout()
                st.pyplot(plt)
            with col4:
                st.write(row['Last Event'])

