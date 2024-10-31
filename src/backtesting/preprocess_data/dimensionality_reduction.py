from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from umap import UMAP

def reduce_dimensionality(data: pd.DataFrame, 
                          method: str = 'pca',
                          n_components: int = None,
                          target_variance: float = 0.95,
                          **kwargs) -> pd.DataFrame:
    """
    Reduce dimensionality of data using various techniques.
    
    Args:
        data (pd.DataFrame): Input data
        method (str, optional): Dimensionality reduction method to use. Options:
            'pca', 'ica', 'svd', 'gaussian_rp', 'sparse_rp', 'tsne', 'isomap', 'lle', 'umap'. 
            Defaults to 'pca'.
        n_components (int, optional): Number of components to keep. Defaults to None.
        target_variance (float, optional): Target explained variance for PCA. Defaults to 0.95.
        **kwargs: Additional keyword arguments for the specific reduction method.
        
    Returns:
        pd.DataFrame: Dimensionality-reduced data
    """
    if method == 'pca':
        if n_components is None:
            pca = PCA(n_components=target_variance)
        else:
            pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        
    elif method == 'ica':
        ica = FastICA(n_components=n_components, **kwargs)
        reduced_data = ica.fit_transform(data)
        
    elif method == 'svd':
        svd = TruncatedSVD(n_components=n_components, **kwargs)
        reduced_data = svd.fit_transform(data)
        
    elif method == 'gaussian_rp':
        grp = GaussianRandomProjection(n_components=n_components, **kwargs)
        reduced_data = grp.fit_transform(data)
        
    elif method == 'sparse_rp':
        srp = SparseRandomProjection(n_components=n_components, **kwargs)
        reduced_data = srp.fit_transform(data)
        
    elif method == 'tsne':
        tsne = TSNE(n_components=n_components, **kwargs)
        reduced_data = tsne.fit_transform(data)
        
    elif method == 'isomap':
        isomap = Isomap(n_components=n_components, **kwargs)
        reduced_data = isomap.fit_transform(data)
        
    elif method == 'lle':
        lle = LocallyLinearEmbedding(n_components=n_components, **kwargs)
        reduced_data = lle.fit_transform(data)
        
    elif method == 'umap':
        umap = UMAP(n_components=n_components, **kwargs)
        reduced_data = umap.fit_transform(data)
        
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
    reduced_df = pd.DataFrame(reduced_data, columns=[f"{method}_{i+1}" for i in range(reduced_data.shape[1])])
    
    return reduced_df
