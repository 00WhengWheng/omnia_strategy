import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, List

class VAEVisualizer:
    def __init__(self, vae_model, original_data: np.ndarray, timestamps: np.ndarray = None):
        """
        Inizializza il visualizzatore
        
        Args:
            vae_model: Il modello VAE addestrato
            original_data: Dati originali usati per il training
            timestamps: Array di timestamps per i dati (opzionale)
        """
        self.vae = vae_model
        self.original_data = original_data
        self.timestamps = timestamps if timestamps is not None else np.arange(len(original_data))
        self.latent_repr = self.vae.encode(original_data)
        
    def reduce_dimensions(self, method: str = 't-sne') -> np.ndarray:
        """Riduce la dimensionalità dello spazio latente per visualizzazione"""
        if method.lower() == 't-sne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2, random_state=42)
            
        return reducer.fit_transform(self.latent_repr)
    
    def identify_clusters(self, n_clusters: int = 5) -> np.ndarray:
        """Identifica cluster nello spazio latente"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(self.latent_repr)
    
    def plot_latent_space_2d(self, 
                            color_by: str = 'cluster',
                            n_clusters: int = 5,
                            dimension_reduction: str = 't-sne',
                            interactive: bool = True) -> None:
        """
        Visualizza lo spazio latente 2D
        
        Args:
            color_by: 'cluster' o 'time'
            n_clusters: numero di cluster se color_by='cluster'
            dimension_reduction: 't-sne' o 'pca'
            interactive: usa plotly per plot interattivo
        """
        reduced_data = self.reduce_dimensions(dimension_reduction)
        
        if color_by == 'cluster':
            colors = self.identify_clusters(n_clusters)
            color_label = 'Cluster'
        else:
            colors = self.timestamps
            color_label = 'Time'
            
        if interactive:
            fig = px.scatter(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                color=colors,
                title=f'Latent Space Visualization ({dimension_reduction.upper()})',
                labels={'color': color_label},
                template='plotly_dark'
            )
            fig.show()
        else:
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                reduced_data[:, 0],
                reduced_data[:, 1],
                c=colors,
                cmap='viridis'
            )
            plt.colorbar(scatter, label=color_label)
            plt.title(f'Latent Space Visualization ({dimension_reduction.upper()})')
            plt.xlabel(f'{dimension_reduction.upper()} 1')
            plt.ylabel(f'{dimension_reduction.upper()} 2')
            plt.show()
            
    def plot_latent_trajectories(self, window_size: int = 50) -> None:
        """Visualizza le traiettorie nello spazio latente"""
        reduced_data = self.reduce_dimensions('pca')
        
        fig = go.Figure()
        
        # Aggiungi tutti i punti come scatter
        fig.add_trace(go.Scatter(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            mode='markers',
            name='All Points',
            marker=dict(size=4, color='gray', opacity=0.5)
        ))
        
        # Visualizza l'ultima finestra come traiettoria
        last_window = reduced_data[-window_size:]
        
        fig.add_trace(go.Scatter(
            x=last_window[:, 0],
            y=last_window[:, 1],
            mode='lines+markers',
            name=f'Last {window_size} points',
            line=dict(color='red', width=2),
            marker=dict(size=6, color='red')
        ))
        
        fig.update_layout(
            title='Latent Space Trajectories',
            template='plotly_dark',
            showlegend=True
        )
        
        fig.show()
        
    def plot_reconstruction_quality(self, n_samples: int = 5) -> None:
        """Visualizza la qualità della ricostruzione"""
        # Seleziona random samples
        idx = np.random.choice(len(self.original_data), n_samples, replace=False)
        samples = self.original_data[idx]
        
        # Ricostruisci i samples
        reconstructed = self.vae.reconstruct(samples)
        
        # Plot
        plt.figure(figsize=(15, 3*n_samples))
        
        for i in range(n_samples):
            # Original
            plt.subplot(n_samples, 2, 2*i + 1)
            plt.plot(samples[i])
            plt.title(f'Original Sample {i+1}')
            plt.grid(True)
            
            # Reconstructed
            plt.subplot(n_samples, 2, 2*i + 2)
            plt.plot(reconstructed[i])
            plt.title(f'Reconstructed Sample {i+1}')
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()
        
    def plot_latent_distribution(self) -> None:
        """Visualizza la distribuzione delle dimensioni latenti"""
        n_dims = self.latent_repr.shape[1]
        n_cols = min(4, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 3*n_rows))
        
        for i in range(n_dims):
            plt.subplot(n_rows, n_cols, i+1)
            sns.histplot(self.latent_repr[:, i], kde=True)
            plt.title(f'Latent Dimension {i+1}')
            
        plt.tight_layout()
        plt.show()

# Esempio di utilizzo
if __name__ == "__main__":
    """
    # Assumendo di avere il VAE addestrato e i dati
    vae = TradingVAE(input_dim=n_features, latent_dim=32)
    vae.fit(x_train)
    
    # Creazione visualizzatore
    visualizer = VAEVisualizer(vae, x_train, timestamps=dates)
    
    # Visualizzazioni diverse
    visualizer.plot_latent_space_2d(color_by='cluster', n_clusters=5)
    visualizer.plot_latent_space_2d(color_by='time')
    visualizer.plot_latent_trajectories(window_size=50)
    visualizer.plot_reconstruction_quality(n_samples=5)
    visualizer.plot_latent_distribution()
    """

    '''
    # Preparazione dati
    timestamps = pd.date_range(start='2023-01-01', periods=len(x_train), freq='D')

    # Creazione visualizzatore
    visualizer = VAEVisualizer(vae_model=vae, 
                            original_data=x_train, 
                            timestamps=timestamps)

    # Analisi dei pattern
    visualizer.plot_latent_space_2d(color_by='time')  # Vedi evoluzione temporale
    visualizer.plot_latent_space_2d(color_by='cluster')  # Identifica regimi di mercato
    visualizer.plot_latent_trajectories()  # Analizza movimenti recenti
    '''