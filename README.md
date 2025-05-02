# Project
from sklearn.mixture import GaussianMixture
# Apply GMM
gmm = GaussianMixture(n_components=4, random_state=42)
y_gmm = gmm.fit_predict(X)
# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='coolwarm', alpha=0.7)
plt.title("Gaussian Mixture Model Clustering")
plt.show()
