import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objs as go

# Recive a 2d matrix 1 row for elements and n embedding values
# You can create it like this np.vstack(encodings.values())
def visualize_3D(data, labels, title="Title!!!", method='tSNE', method_settings={"perplexity":30.0, "random_state":0}):
    # Reduction
    model = []
    if method == 'tSNE':
        model = TSNE(n_components=3, perplexity=30, **method_settings)
    elif method == 'PCA':
        model = PCA(n_components=3, **method_settings) 
    else:
        raise Exception("Invalid method provided")
    
    reduced_vectors = model.fit_transform(data)

    # Visualization
    trace = go.Scatter3d(
        x=reduced_vectors[:,0],
        y=reduced_vectors[:,1],
        z=reduced_vectors[:,2],
        mode='markers',
        marker=dict(
            color=labels,
            size=5,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title=title)
        )
    )

    # create the layout for the plot
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title=dict(text=title)
    )

    # create the figure and plot it
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()


def visualize_embedding_2D(data, labels, title="Title!!!", method='tSNE', method_settings={ "random_state":0}):
    # Reduction
    model = []
    if method == 'tSNE':
        model = TSNE(n_components=3, perplexity=30, **method_settings)
    elif method == 'PCA':
        model = PCA(n_components=3, **method_settings) 
    else:
        raise Exception("Invalid method provided")
    
    reduced_vectors = model.fit_transform(data)
    
    # Visualization
    trace = go.Scatter(
        x=reduced_vectors[:,0],
        y=reduced_vectors[:,1],
        mode='markers',
        marker=dict(
            color=labels,
            size=5,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title=title)
        )
    )

    # create the layout for the plot
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
        ),
        title=dict(text=title)
    )

    # create the figure and plot it
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()