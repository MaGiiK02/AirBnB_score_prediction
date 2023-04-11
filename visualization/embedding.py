import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objs as go

# Receive a 2d matrix 1 row for elements and n embedding values
# You can create it like this np.vstack(encodings.values())
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objs as go


# a function to reduce the dimensionality of the data using t-SNE or PCA
def reduce_dimensionality(data, method='tSNE', method_settings={"perplexity":30.0, "random_state":0}):
    # Reduction
    model = None
    if method == 'tSNE':
        model = TSNE(n_components=3, **method_settings)
    elif method == 'PCA':
        model = PCA(n_components=3, **method_settings) 
    else:
        raise Exception("Invalid method provided")
    
    reduced_vectors = model.fit_transform(data)
    return reduced_vectors, model

# a function that takes the reduced vectors and plots them in 3D
def visualize_3D(reduced_vectors, labels, title="Title!!!"):
    # you have to initialize the reduced vectors yourself outside this function
    # and pass it to this function
    if reduced_vectors is None:
        raise Exception("Reduced vectors not provided")
    
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

# a function that takes the reduced vectors and plots them in 2D
def visualize_2D(reduced_vectors, labels, title="Title!!!"):
    # you have to initialize the reduced vectors yourself outside this function
    if reduced_vectors is None:
        raise Exception("Reduced vectors not provided")
    
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