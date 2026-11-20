"""
Phase 2 - Step 5: Dimensionality Reduction Module.
Implements:
1. Linear PCA for variance explanation, noise reduction, and downstream input generation.
2. Non-linear Manifold Projection (t-SNE / 2D Projections) for structural visualization.
Strictly enforces Zero Data Leakage: PCA fitted exclusively on Train.
Compatible with scikit-learn 1.4+.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class DimensionalityReducer:
    """
    Manages linear and non-linear dimensionality reduction pipelines.
    """

    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.is_fitted = False

    def fit_pca(self, X_train: np.ndarray):
        """Fit PCA strictly on the training embeddings."""
        print(f"[*] Fitting PCA (n_components={self.n_components}) strictly on Train embeddings...")
        self.pca.fit(X_train)
        self.is_fitted = True
        explained_var = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"    -> Cumulative Explained Variance (50 components): {explained_var:.2f}%")
        return self

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        """Projects dense embeddings into compressed PCA space."""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted on Train before transform.")
        return self.pca.transform(X)

    def compute_2d_manifold(self, X_pca: np.ndarray, sample_size: int = 4000) -> np.ndarray:
        """
        Computes 2D t-SNE projection from top PCA components
        for rapid, publication-quality 2D manifold visualization.
        """
        n_samples = min(sample_size, len(X_pca))
        print(f"[*] Projecting top PCA components to 2D Manifold using t-SNE (N={n_samples})...")
        X_sub = X_pca[:n_samples, :30]
        
        # Uses max_iter (standard across scikit-learn versions)
        tsne = TSNE(
            n_components=2,
            perplexity=35,
            learning_rate="auto",
            init="random",
            random_state=self.random_state,
            max_iter=1000
        )
        coords_2d = tsne.fit_transform(X_sub)
        return coords_2d