import streamlit as st
import pandas as pd
import numpy as np
from clustering_utils import (
    load_and_process_data, 
    create_elbow_plot, 
    create_cluster_plot,
    create_confusion_matrix,
    create_feature_distribution,
    get_classification_metrics
)

st.set_page_config(
    page_title="Online Shoppers Clustering Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Load custom CSS styles"""
    try:
        with open('styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                color: #2c3e50;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #3498db;
                margin-bottom: 1rem;
            }
            .section-header {
                color: #34495e;
                border-bottom: 2px solid #3498db;
                padding-bottom: 0.5rem;
                margin-top: 2rem;
                margin-bottom: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)

@st.cache_data
def get_processed_data():
    """Cached function to load and process data"""
    return load_and_process_data()

def render_sidebar(results):
    """Render the sidebar with dataset information and metrics"""
    st.sidebar.header("📊 Dashboard Navigation")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📋 Dataset Information")
    st.sidebar.info(f"""
    **Dataset Shape:** {results['data'].shape[0]} rows × {results['data'].shape[1]} columns
    
    **Features Used:**
    - {results['feature_names'][0]}
    - {results['feature_names'][1]}
    
    **Missing Values:** {results['data'].isnull().sum().sum()}
    
    **Algorithm:** K-Means Clustering (k=2)
    """)
    
    st.sidebar.subheader("🎯 Key Metrics")
    st.sidebar.metric("Adjusted Rand Index", f"{results['adj_rand_score']:.4f}")
    st.sidebar.metric("Silhouette Score", f"{results['silhouette_score']:.4f}")
    st.sidebar.metric("Cluster 0 Size", f"{np.sum(results['y_means'] == 0):,}")
    st.sidebar.metric("Cluster 1 Size", f"{np.sum(results['y_means'] == 1):,}")

def render_elbow_tab(results):
    """Render the elbow analysis tab"""
    st.markdown('<h2 class="section-header">Elbow Method Analysis</h2>', unsafe_allow_html=True)
    st.markdown("""
    The elbow method helps determine the optimal number of clusters by plotting the Within-Cluster Sum of Squares (WCSS) 
    against the number of clusters. The "elbow" point indicates the optimal k value.
    """)
    
    fig_elbow = create_elbow_plot(results)
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    st.info("💡 **Insight:** The elbow appears around k=2, suggesting 2 clusters is optimal for this dataset.")

def render_clustering_tab(results):
    """Render the clustering results tab"""
    st.markdown('<h2 class="section-header">Clustering Results</h2>', unsafe_allow_html=True)
    st.markdown("""
    Interactive visualization of the K-means clustering results showing customer segments based on their behavior patterns.
    """)
    
    fig_cluster = create_cluster_plot(results)
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🟡 Uninterested Customers:**
        - Lower engagement patterns
        - May need targeted marketing
        """)
    with col2:
        st.markdown("""
        **🔴 Target Customers:**
        - Higher engagement patterns  
        - More likely to make purchases
        """)

def render_evaluation_tab(results):
    """Render the model evaluation tab"""
    st.markdown('<h2 class="section-header">Model Evaluation</h2>', unsafe_allow_html=True)
    
    fig_cm = create_confusion_matrix(results)
    st.plotly_chart(fig_cm, use_container_width=True)
    
    st.subheader("📊 Classification Report")
    class_report = get_classification_metrics(results)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{class_report['accuracy']:.3f}")
    with col2:
        st.metric("Macro Avg Precision", f"{class_report['macro avg']['precision']:.3f}")
    with col3:
        st.metric("Macro Avg Recall", f"{class_report['macro avg']['recall']:.3f}")

def render_data_overview_tab(results):
    """Render the data overview tab"""
    st.markdown('<h2 class="section-header">Data Overview</h2>', unsafe_allow_html=True)
    
    st.subheader("📋 Dataset Preview")
    st.dataframe(results['data'].head(10), use_container_width=True)
    
    st.subheader("📊 Summary Statistics")
    st.dataframe(results['data'].describe(), use_container_width=True)
    
    st.subheader("🔍 Data Types")
    dtype_df = pd.DataFrame({
        'Column': results['data'].columns,
        'Data Type': results['data'].dtypes.values,
        'Non-Null Count': results['data'].count().values,
        'Null Count': results['data'].isnull().sum().values
    })
    st.dataframe(dtype_df, use_container_width=True)

def render_feature_analysis_tab(results):
    """Render the feature analysis tab"""
    st.markdown('<h2 class="section-header">Feature Analysis</h2>', unsafe_allow_html=True)
    
    fig_dist = create_feature_distribution(results)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.subheader("📈 Feature Statistics")
    feature_stats = results['data'].iloc[:, [5, 6]].describe()
    st.dataframe(feature_stats, use_container_width=True)

def render_footer():
    """Render the footer"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>🛒 Online Shoppers Clustering Dashboard | Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    load_css()
    
    st.markdown('<h1 class="main-header">🛒 Online Shoppers Clustering Dashboard</h1>', unsafe_allow_html=True)
    
    try:
        with st.spinner('Loading and processing data...'):
            results = get_processed_data()
    except FileNotFoundError:
        st.error("❌ Could not find 'online_shoppers_intention.csv'. Please make sure the file is in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    render_sidebar(results)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Elbow Analysis", 
        "🎯 Clustering Results", 
        "📊 Model Evaluation", 
        "📋 Data Overview", 
        "📉 Feature Analysis"
    ])
    
    with tab1:
        render_elbow_tab(results)
    
    with tab2:
        render_clustering_tab(results)
    
    with tab3:
        render_evaluation_tab(results)
    
    with tab4:
        render_data_overview_tab(results)
    
    with tab5:
        render_feature_analysis_tab(results)
    
    render_footer()

if __name__ == "__main__":
    main()