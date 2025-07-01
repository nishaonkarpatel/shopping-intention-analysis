import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

def load_and_process_data():
    """
    Load and process the online shoppers intention dataset.
    
    Returns:
        dict: Dictionary containing processed data and analysis results
    """
    try:
        # Load the data
        data = pd.read_csv('online_shoppers_intention.csv')
        
        # Select features for clustering (columns 5 and 6)
        # These should be ProductRelated_Duration and BounceRates
        x = data.iloc[:, [5, 6]].values
        feature_names = [data.columns[5], data.columns[6]]
        
        # Calculate WCSS for elbow method
        wcss = []
        k_range = range(1, 11)
        for i in k_range:
            km = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
                       n_init=10, random_state=0, algorithm='lloyd')
            km.fit(x)
            wcss.append(km.inertia_)
        
        # Apply K-means with 2 clusters
        km = KMeans(n_clusters=2, init='k-means++', max_iter=300, 
                   n_init=10, random_state=0, algorithm='lloyd')
        y_means = km.fit_predict(x)
        
        # Calculate evaluation metrics
        le = LabelEncoder()
        labels_true = le.fit_transform(data['Revenue'])
        labels_pred = y_means
        
        adj_rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)
        silhouette_score = metrics.silhouette_score(x, labels_pred)
        
        # Confusion matrix
        cm = confusion_matrix(labels_true, labels_pred)
        
        return {
            'data': data,
            'x': x,
            'feature_names': feature_names,
            'wcss': wcss,
            'k_range': k_range,
            'y_means': y_means,
            'km': km,
            'labels_true': labels_true,
            'labels_pred': labels_pred,
            'adj_rand_score': adj_rand_score,
            'silhouette_score': silhouette_score,
            'cm': cm
        }
    except FileNotFoundError:
        raise FileNotFoundError("Could not find 'online_shoppers_intention.csv'. Please make sure the file is in the same directory as this script.")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def create_elbow_plot(results):
    """
    Create elbow plot for K-means clustering.
    
    Args:
        results (dict): Results dictionary from load_and_process_data
        
    Returns:
        plotly.graph_objects.Figure: Elbow plot figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(results['k_range']),
        y=results['wcss'],
        mode='lines+markers',
        name='WCSS',
        line=dict(color='#3498db', width=3),
        marker=dict(size=10, color='#e74c3c')
    ))
    fig.update_layout(
        title='The Elbow Method for Optimal K',
        xaxis_title='Number of Clusters',
        yaxis_title='Within-Cluster Sum of Squares (WCSS)',
        template='plotly_white',
        height=400,
        font=dict(size=12)
    )
    return fig

def create_cluster_plot(results):
    """
    Create scatter plot showing clustering results.
    
    Args:
        results (dict): Results dictionary from load_and_process_data
        
    Returns:
        plotly.graph_objects.Figure: Cluster visualization figure
    """
    x = results['x']
    y_means = results['y_means']
    km = results['km']
    feature_names = results['feature_names']
    
    fig = go.Figure()
    
    # Add cluster points
    colors = ['#f39c12', '#e91e63']
    labels = ['Uninterested Customers', 'Target Customers']
    
    for i in range(2):
        cluster_points = x[y_means == i]
        fig.add_trace(go.Scatter(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            mode='markers',
            name=labels[i],
            marker=dict(color=colors[i], size=8, opacity=0.7),
            hovertemplate=f'<b>{labels[i]}</b><br>' +
                         f'{feature_names[0]}: %{{x}}<br>' +
                         f'{feature_names[1]}: %{{y}}<extra></extra>'
        ))
    
    # Add centroids
    fig.add_trace(go.Scatter(
        x=km.cluster_centers_[:, 0],
        y=km.cluster_centers_[:, 1],
        mode='markers',
        name='Centroids',
        marker=dict(color='#2c3e50', size=20, symbol='x', line=dict(width=3)),
        hovertemplate='<b>Centroid</b><br>' +
                     f'{feature_names[0]}: %{{x}}<br>' +
                     f'{feature_names[1]}: %{{y}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{feature_names[0]} vs {feature_names[1]} - K-Means Clustering',
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        template='plotly_white',
        height=500,
        font=dict(size=12)
    )
    return fig

def create_confusion_matrix(results):
    """
    Create confusion matrix heatmap.
    
    Args:
        results (dict): Results dictionary from load_and_process_data
        
    Returns:
        plotly.graph_objects.Figure: Confusion matrix figure
    """
    cm = results['cm']
    
    # Create subplots for both raw and normalized confusion matrices
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Raw Counts', 'Normalized Percentages'),
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Raw confusion matrix
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=['Predicted: No Purchase', 'Predicted: Purchase'],
            y=['Actual: No Purchase', 'Actual: Purchase'],
            colorscale='Blues',
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig.add_trace(
        go.Heatmap(
            z=cm_normalized,
            x=['Predicted: No Purchase', 'Predicted: Purchase'],
            y=['Actual: No Purchase', 'Actual: Purchase'],
            colorscale='Blues',
            showscale=True,
            text=np.round(cm_normalized, 3),
            texttemplate="%{text}",
            textfont={"size": 16},
            hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Percentage: %{z:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Confusion Matrix Analysis',
        height=400,
        template='plotly_white',
        font=dict(size=12)
    )
    return fig

def create_feature_distribution(results):
    """
    Create histogram plots for feature distributions.
    
    Args:
        results (dict): Results dictionary from load_and_process_data
        
    Returns:
        plotly.graph_objects.Figure: Feature distribution figure
    """
    data = results['data']
    feature_names = results['feature_names']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=tuple(feature_names)
    )
    
    # Feature 1 distribution
    fig.add_trace(
        go.Histogram(
            x=data.iloc[:, 5],
            name=feature_names[0],
            nbinsx=30,
            marker_color='#3498db',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Feature 2 distribution
    fig.add_trace(
        go.Histogram(
            x=data.iloc[:, 6],
            name=feature_names[1],
            nbinsx=30,
            marker_color='#e74c3c',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Feature Distributions',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    return fig

def get_classification_metrics(results):
    """
    Calculate and return classification metrics.
    
    Args:
        results (dict): Results dictionary from load_and_process_data
        
    Returns:
        dict: Classification report as dictionary
    """
    return classification_report(
        results['labels_true'], 
        results['labels_pred'], 
        target_names=['No Purchase', 'Purchase'], 
        output_dict=True
    )