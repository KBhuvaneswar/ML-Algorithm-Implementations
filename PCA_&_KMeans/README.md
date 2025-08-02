# Machine Learning: PCA and K-Means Clustering

## Overview
This repository showcases the application of two fundamental unsupervised learning techniques: Principal Component Analysis (PCA) for dimensionality reduction and K-means clustering (using Lloyd's algorithm) for data partitioning. It explores their implementation, visualization of results, and critical evaluation on different datasets.

## Part I: Principal Component Analysis (PCA)

### Dataset
The MNIST dataset, a collection of 28x28 grayscale images of handwritten digits (0 to 9), was used. A subset of 1000 images was randomly selected for this assignment, with 100 samples from each digit class (0-9) to ensure a balanced analysis.
* **Total samples:** 1000
* **Image dimensions:** 28x28 pixels
* **Classes:** 10 (digits 0 to 9)
* **Samples per class:** 100

**Note:** The MNIST dataset was obtained from Hugging Face datasets. It is not included in this repository, but can be freely downloaded from public sources.

### Methodology
The PCA algorithm was implemented from scratch following these steps:
1.  **Data Preprocessing:** Each 28x28 image was flattened into a 1D array of 784 pixels, resulting in a dataset of shape (1000, 784).

2.  **Data Centering:** The mean of each feature was subtracted from the corresponding feature values to ensure the first principal component captures maximum variance.

3.  **Covariance Matrix Calculation:** The covariance matrix of the centered data was computed to understand feature relationships.

4.  **Eigenvalue & Eigenvector Calculation:** Eigenvectors and eigenvalues of the covariance matrix were calculated. Eigenvectors represent principal components, and eigenvalues indicate explained variance.

5.  **Sorting Components:** Eigenvectors were sorted in decreasing order of their corresponding eigenvalues to identify the most important components.

6.  **Selecting Top 'k' Components:** The top `k=10` principal components were selected.

7.  **Data Projection:** The centered data was projected onto the selected principal components to reduce dimensionality.

### Visualization of Principal Components
The top 10 principal components were reshaped back into 28x28 images and visualized. These images represent the directions in feature space that capture the most variance, offering insights into the underlying patterns of the MNIST digits.

### Explained Variance by Principal Components
The proportion of variance explained by each principal component was calculated and analyzed:

| Principal Component | Explained Variance (%) |
| :------------------ | :--------------------- |
| Principal Component 1 | 9.67% |
| Principal Component 2 | 7.24% |
| Principal Component 3 | 6.47% |
| Principal Component 4 | 5.34% |
| Principal Component 5 | 4.78% |
| Principal Component 6 | 4.44% |
| Principal Component 7 | 3.34% |
| Principal Component 8 | 2.96% |
| Principal Component 9 | 2.80% |
| Principal Component 10 | 2.38% |

**Analysis:** The first principal component explains the highest variance (9.67%), with diminishing returns for subsequent components, indicating that the initial components capture most of the data's essential information.

### Reconstructing and Visualizing the Dataset
The dataset was reconstructed using various numbers of principal components ($d \in \{10, 25, 50, 100, 150\}$) to observe the trade-off between dimensionality reduction and image quality.

* **Reconstruction Process:** A custom `reconstruct_image` function projected the reduced-dimensional data back into the original feature space and added the dataset's mean.

* **Observations on Reconstruction Quality:**
    * **d=10:** Reconstructed images retain only prominent features, appearing blurry.
    * **d=25:** Images are more refined, but finer details are still missing.
    * **d=50:** Significant improvement in clarity, capturing much of the original structure.
    * **d=100:** Images are almost indistinguishable from originals, retaining most details.
    * **d=150:** Closely match original images, achieving near-complete reconstruction.

### Cumulative Variance Explained
The cumulative variance explained by the principal components shows how much total variance is captured as more components are included:

| Number of Components (n) | Cumulative Variance Explained |
| :----------------------- | :---------------------------- |
| 10 | 49.42% |
| 25 | 70.37% |
| 50 | 83.85% |
| 100 | 92.64% |
| 150 | 95.94% |

The curve of cumulative variance explained rises steeply initially and then flattens, indicating that a relatively small number of components capture a large portion of the variance.

### Optimal Dimensionality for Downstream Classification Tasks
For a downstream digit classification task, an optimal dimension of **50-100** principal components is recommended. This range offers a good balance between retaining significant information (capturing 83.85% to 92.64% of variance) and maintaining computational efficiency, avoiding the increased costs associated with higher dimensions while minimizing information loss from very low dimensions.


## Part II: K-means Clustering (Lloyd's Algorithm)

### Dataset
The dataset used for clustering is `cm_dataset_2.csv`, consisting of 1000 data points in a two-dimensional space ($\mathbb{R}^2$). Each data point has two features (Feature 1 on X-axis, Feature 2 on Y-axis). The spatial distribution suggests it's suitable for clustering analysis.

**Note:** The `cm_dataset_2.csv` file is **not included** in this repository.

### Lloyd's Algorithm Implementation
The `lloyds_kmeans` function was implemented to cluster a dataset into `k` clusters.

* **Input Arguments:**
    * `data`: NumPy array of the dataset.
    * `n_clusters`: Number of clusters (default 2).
    * `max_iters`: Maximum iterations for convergence (default 100).

* **Output:**
    * `cluster_labels`: Array of cluster assignments for each data point.
    * `centroids`: Final coordinates of the cluster centers.

* **Algorithm Steps:**
    1.  **Initialization:** Randomly select `k` data points as initial cluster centroids.
    2.  **Cluster Assignment:** Assign each data point to the nearest centroid based on Euclidean distance.
    3.  **Centroid Update:** Update centroids to the mean position of all points assigned to that cluster.
    4.  **Convergence Check:** Stop when centroids no longer change (using `np.allclose`) or `max_iters` is reached.

* **Error Function:** `calculate_error` function computes the total squared Euclidean distance between data points and their assigned centroids, quantifying cluster compactness.

### Random Initializations and Convergence (K=2)
Lloyd's algorithm was run with 5 different random initializations for $K=2$. For each trial, the error function was plotted against iterations, and the final cluster configurations were visualized.

* **Convergence Behavior:** The error function typically reached a stable value within 20-50 iterations, showing quick convergence. Trials with poorer initializations sometimes showed slower convergence and higher final error.

* **Cluster Assignments:** Most data points were consistently assigned to the same clusters across trials. However, some points near cluster boundaries exhibited variability due to different initial centroids.

* **Centroid Stability:** Final centroid locations were largely similar across trials, indicating robust clustering results despite initial differences.

### Visualizing Voronoi Regions for K-means Clustering
Voronoi regions were generated for cluster centroids obtained with $K \in \{2, 3, 4, 5\}$.

* **Approach:** After applying Lloyd's algorithm for each K, the `scipy.spatial.Voronoi` function was used to compute the Voronoi tessellation.

* **Purpose:** These regions visualize the influence of each cluster centroid in the feature space, showing how data points are partitioned based on their proximity to centroids.

### Evaluation of Lloyd's Algorithm for this Dataset
**Conclusion:** Lloyd's algorithm for K-means clustering is **not the most suitable method** for `cm_dataset_2.csv`.

* **Justification:** Lloyd's algorithm makes assumptions about spherical, convex clusters of similar variance, which this dataset's complex, non-linear structure does not meet.

* **Recommendation:** **Spectral clustering** is recommended as a powerful alternative.
    * **Advantages of Spectral Clustering:**
        * Handles non-convex clusters.
        * Accurately represents complex relationships in the data.
        * Adapts to irregular cluster shapes and varying densities.
    * For intricate or non-linear patterns present in this dataset, spectral clustering is highly recommended for achieving meaningful and robust partitions.

## Libraries Used
* NumPy (for array manipulations, mathematical operations)
* Matplotlib (for plotting visualizations like explained variance, reconstructed images, cluster plots, error plots)
* Hugging Face `datasets` library (for MNIST download)
* `scipy.spatial.Voronoi` (for Voronoi tessellation)

## How to Run
To run the code:
1.  Ensure you have Python, NumPy, Matplotlib, and `scipy` installed.
2.  **For PCA:**
    * Ensure the `datasets` library is installed (`pip install datasets`).
    * The `pca.ipynb` file will handle downloading the MNIST dataset automatically.
    * Execute the code cells.