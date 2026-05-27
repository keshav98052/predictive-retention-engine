import streamlit as st
import pandas as pd
import datetime as dt
import plotly.express as px
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

# ==========================================
# 1. APP CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="AI Retention Engine", layout="wide")
st.title("🎯 Predictive Marketing & Retention Engine")

# Sidebar UI
st.sidebar.header("Data Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Transaction Dataset (CSV)", type="csv")
analysis_type = st.sidebar.selectbox(
    "Select Dashboard View", 
    ["1. Marketing Segmentation (RFM)", "2. Predictive Retention (CLV & Churn)"]
)

# ==========================================
# 2. THE AI BRAIN (CACHED FOR SPEED)
# ==========================================
# We use @st.cache_data so the ML model doesn't re-run every time you click a button
@st.cache_data
def process_data(file):
    # Load and clean data
    df = pd.read_csv(file, encoding='unicode_escape')
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df['Total_Price'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Generate Lifetimes Summary Matrix
    summary_data = summary_data_from_transaction_data(
        df, 'CustomerID', 'InvoiceDate', monetary_value_col='Total_Price', 
        observation_period_end=df['InvoiceDate'].max()
    )
    
    # Filter returning customers
    returning = summary_data[summary_data['frequency'] > 0].copy()
    
    # Train BG/NBD (Churn/Frequency)
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(returning['frequency'], returning['recency'], returning['T'])
    
    # Train Gamma-Gamma (Monetary Value)
    ggf = GammaGammaFitter(penalizer_coef=0)
    ggf.fit(returning['frequency'], returning['monetary_value'])
    
    # Predictions
    returning['Probability_Alive'] = bgf.conditional_probability_alive(
        returning['frequency'], returning['recency'], returning['T']
    )
    returning['Predicted_CLV_90_Days'] = ggf.customer_lifetime_value(
        bgf, returning['frequency'], returning['recency'], returning['T'], 
        returning['monetary_value'], time=3, discount_rate=0.01
    )
    
    # RFM Scoring Logic (Simplified for Streamlit)
    returning['R_Score'] = pd.qcut(returning['recency'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    returning['F_Score'] = pd.qcut(returning['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
    
    def assign_segment(row):
        r = int(row['R_Score'])
        f = int(row['F_Score'])
        if r >= 4 and f >= 4: return 'Champions'
        elif r <= 2 and f >= 3: return 'At Risk'
        elif r <= 2 and f <= 2: return 'Hibernating'
        else: return 'Loyal Customers'
        
    returning['Marketing_Action'] = returning.apply(assign_segment, axis=1)
    returning['High_Churn_Risk'] = returning['Probability_Alive'] < 0.30
    
    return returning

# ==========================================
# 3. THE INTERACTIVE DASHBOARD
# ==========================================
if uploaded_file is not None:
    with st.spinner("Training AI Models & Processing Data..."):
        df_processed = process_data(uploaded_file)
    
    if analysis_type == "1. Marketing Segmentation (RFM)":
        st.header("Tactical Marketing Segments")
        
        # The Slicer (Dropdown)
        segments = ["All"] + list(df_processed['Marketing_Action'].unique())
        selected_segment = st.selectbox("Filter by Segment:", segments)
        
        # Filter Logic
        if selected_segment != "All":
            filtered_df = df_processed[df_processed['Marketing_Action'] == selected_segment]
        else:
            filtered_df = df_processed
            
        # The RFM Grid (Scatter Plot)
        st.subheader("Customer Distribution Map")
        fig = px.scatter(
            filtered_df, x="frequency", y="monetary_value", color="Marketing_Action",
            hover_name=filtered_df.index, log_x=True, log_y=True,
            title="Frequency vs. Monetary Value (Log Scale)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # The Export List
        st.subheader("Export Action List")
        # Streamlit dataframes natively have a download/export button when users hover over them!
        st.dataframe(filtered_df[['Marketing_Action', 'recency', 'frequency', 'monetary_value']])

    elif analysis_type == "2. Predictive Retention (CLV & Churn)":
        st.header("Predictive Retention (Next 90 Days)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # High-Churn Warning List
            st.subheader("🚨 High-Churn Warning List")
            st.write("Customers with < 30% probability of returning.")
            churn_risk_df = df_processed[df_processed['High_Churn_Risk'] == True]
            churn_risk_df = churn_risk_df.sort_values(by='Predicted_CLV_90_Days', ascending=False)
            
            # Formatting to currency and percentage for the UI
            display_df = churn_risk_df[['Marketing_Action', 'Probability_Alive', 'Predicted_CLV_90_Days']].copy()
            display_df['Probability_Alive'] = (display_df['Probability_Alive'] * 100).round(1).astype(str) + '%'
            display_df['Predicted_CLV_90_Days'] = '$' + display_df['Predicted_CLV_90_Days'].round(2).astype(str)
            
            st.dataframe(display_df, height=400)
            
        with col2:
            # Future Value Forecast
            st.subheader("💰 Future Value Forecast by Segment")
            clv_by_segment = df_processed.groupby('Marketing_Action')['Predicted_CLV_90_Days'].sum().reset_index()
            
            fig2 = px.bar(
                clv_by_segment, x='Marketing_Action', y='Predicted_CLV_90_Days', 
                color='Marketing_Action', title="Forecasted Revenue (90 Days)"
            )
            st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 Please upload a transaction CSV in the sidebar to run the AI engine.")