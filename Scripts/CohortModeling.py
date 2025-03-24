#!/usr/bin env python
# -*- coding: utf-8 -*-

"""
Author: Michael Garancher
Date: 2025-03-01
Description: Clustering and Predictive modeling for customer churn
"""


import sys
from pathlib import Path
import warnings
import logging

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

import matplotlib.pyplot as plt	

warnings.filterwarnings('ignore')

# Set pandas display options
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

# Set the root directory of the project
ROOT:Path = Path(__file__).resolve().parents[1]



class KMeansClustering:
	"""
	KMeansClustering class for performing KMeans clustering with hyperparameter tuning, evaluation, and visualization.
	
	Attributes:
		df (pd.DataFrame): The input dataframe.
		X (pd.DataFrame): The feature matrix after preprocessing.
		_y (pd.Series): The target feature used for evaluation.
		Y (np.ndarray): Encoded target feature if categorical, otherwise same as _y.
		n_components (np.ndarray): Array of components for PCA and Factor Analysis.
		n_clusters (np.ndarray): Array of cluster numbers for KMeans.
		best_model (Pipeline): The best model found during hyperparameter tuning.
		best_params (dict): The best parameters found during hyperparameter tuning.
	
	Methods:
		__init__(df: pd.DataFrame, target_feature: str, sample_size: float = 1.0):
			Initializes the KMeansClustering instance with the given dataframe, target feature, and sample size.
		n_components:
			Property getter and setter for n_components attribute.
		n_clusters:
			Property getter and setter for n_clusters attribute.
		safe_silhouette(estimator, X, y=None):
			Static method to calculate the silhouette score safely, handling edge cases.
		feature_selection(method, threshold):
			Selects features based on the specified method and threshold.
			Returns the instance with the selected features.
		predict():
			Predicts clusters for new data using the best model found during hyperparameter tuning.
			Returns a dataframe with an additional 'Cluster' column containing cluster assignments.
		fit_predict():
			Performs hyperparameter tuning for KMeans clustering, fits the model, and predicts clusters.
			Returns a dataframe with an additional 'Cluster' column containing cluster assignments.
		evaluate():
			Evaluates the model by calculating the silhouette score and prints it to the console.
			Returns a dictionary of evaluation metrics.
		explain_clusters(df_with_clusters):
			Generates a summary of key characteristics for each cluster in the given dataframe.
			Returns a dataframe with cluster statistics.
		visualize_clusters(df_with_clusters, save_path: Path = None):
			Visualizes clusters in a given DataFrame using PCA for dimensionality reduction.
			Displays a scatter plot of the clusters using PCA components.
		save_model(cls, filename: str):
			Class method to save the model to a file.
		load_model(filepath: Path):
			Class method to load the model from a file.
	"""
	def __init__(self, df: pd.DataFrame, target_feature: str, sample_size: float = 1.0):
		# Sample data if requested
		self.df = df.sample(frac=sample_size, random_state=42).reset_index(drop=True)
		
		# Get features (numeric columns without target)
		self.X = self.df.select_dtypes(include=[int, float]).copy()
		if target_feature in self.X.columns:
			self.X = self.X.drop(columns=target_feature)
		
		# Get target
		self._y = df[target_feature].copy()
		self.Y = LabelEncoder().fit_transform(self._y) if self._y.dtype == 'object' else self._y

		# Scorer for hyperparameter tuning
		self.n_components = np.arange(5, 20, 5)  # PCA and Factor Analysis components
		self.n_clusters = np.arange(5, 11, 2)  # KMeans clusters

		# Best model and parameters init
		self.best_model = None
		self.best_params = None

	@property
	def n_components(self) -> np.ndarray:
		return self._n_components
	@n_components.setter
	def n_components(self, n_components:np.ndarray) -> None:
		if n_components.size == 0:
			raise ValueError("n_components cannot be empty")
		self._n_components = n_components

	@property
	def n_clusters(self) -> np.ndarray:
		return self._n_clusters
	@n_clusters.setter
	def n_clusters(self, n_clusters:np.ndarray) -> None:
		if n_clusters.size == 0:
			raise ValueError("n_clusters cannot be empty")
		self._n_clusters = n_clusters

	@classmethod
	def safe_silhouette(cls, estimator, X, y=None) -> float:
		""" Safely compute the silhouette score for the given estimator and data."""
		# First apply the full pipeline to get clusters
		labels = estimator.predict(X)
		unique_labels = np.unique(labels)
		
		# Only calculate score if we have at least 2 clusters and each has >1 sample
		if len(unique_labels) > 1 and all(np.sum(labels == label) > 1 for label in unique_labels):
			return silhouette_score(X, labels)
		return -1  # Return a bad score instead of failing
	
	def feature_selection(self, method='variance', threshold=0.01):
		"""Select features based on specified method."""		
		if method == 'variance':
			selector = VarianceThreshold(threshold=threshold)
			self.X = pd.DataFrame(selector.fit_transform(self.X), 
								columns=self.X.columns[selector.get_support()])
		elif method == 'kbest' and self._y is not None:
			selector = SelectKBest(f_classif, k=min(20, self.X.shape[1]))
			self.X = pd.DataFrame(selector.fit_transform(self.X, self.Y),
								columns=self.X.columns[selector.get_support()])	
		print(f"Selected {self.X.shape[1]} features using {method} method")
		return self
	
	def predict(self) -> pd.DataFrame:
		""" Predict clusters for new data."""
		if self.best_model is None:
			raise ValueError("Model has not been trained yet. Please call fit_predict() first.")
		df_with_clusters = self.df.copy(deep=True)
		try:
			df_with_clusters['Cluster'] = self.best_model.predict(self.X)
			print("Clusters predicted successfully\n")
			return df_with_clusters
		except Exception as e:
			raise ValueError(f"Error predicting clusters: {e}")

	def fit_predict(self) -> pd.DataFrame:
		"""Perform hyperparameter tuning for KMeans clustering, fit the model, and predict clusters."""
		
		print("\nWorking on hyperparameter tuning for KMeans clustering...")
		
		# Create pipeline with scaling, dimensionality reduction, and clustering
		pipe = Pipeline([
			("scaling", StandardScaler()),
			("reduce_dim", "passthrough"),
			("clustering", KMeans(random_state=42, n_init=10))
		])
		
		param_grid = {
			"reduce_dim": [PCA(), FactorAnalysis()],
			"reduce_dim__n_components": self.n_components,
			"clustering__n_clusters": self.n_clusters
		}
		
		# Use safer scoring approach
		grid = GridSearchCV(
			pipe, 
			param_grid=param_grid,
			scoring=self.safe_silhouette,
			n_jobs=-1, 
			cv=3,  
			verbose=1
		)
		
		# Fit the model
		grid.fit(self.X)

		# Get the best model and parameters
		self.best_params = grid.best_params_
		self.best_model = grid.best_estimator_
		
		print("\n",
			f"# Sample Size: {self.X.shape[0]}",
			f"# Best Parameters: {self.best_params}",
			f"# Best Score: {grid.best_score_:.4f}",
			sep="\n", end="\n\n"
		)

		# Get predictions for the entire dataset
		df_with_clusters = self.df.copy(deep=True)
		try:
			df_with_clusters['Cluster'] = self.best_model.predict(self.X)
			print("Clusters predicted successfully\n")
			return df_with_clusters
		except Exception as e:
			raise ValueError(f"Error predicting clusters: {e}\n")
		
	def evaluate(self) -> None:
		"""	Evaluate the model by calculating the silhouette score."""
		
		if self.best_model is None:
			raise ValueError("Model has not been trained yet.")
		
		labels = self.best_model.predict(self.X)
		metrics = {}
		
		# Silhouette score (cluster cohesion criterion)
		metrics['silhouette'] = self.safe_silhouette(self.best_model, self.X)
		
		# Calinski-Harabasz Index (variance ratio criterion)
		metrics['calinski_harabasz'] = calinski_harabasz_score(self.X, labels)
		
		# Davies-Bouldin Index (cluster separation criterion)
		metrics['davies_bouldin'] = davies_bouldin_score(self.X, labels)
		
		# Inertia (KMeans only)
		if hasattr(self.best_model.named_steps['clustering'], 'inertia_'):
			metrics['inertia'] = self.best_model.named_steps['clustering'].inertia_
		
		# Output results
		print("# Cluster Evaluation Metrics:")
		for name, value in metrics.items():
			print(f"  - {name.replace('_', ' ').title()}: {value:.4f}")
			
		return metrics

	def explain_clusters(self, df_with_clusters):
		"""Generate a summary of key characteristics for each cluster."""
		result = {}
		for cluster in sorted(df_with_clusters['Cluster'].unique()):
			cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
			result[f'Cluster {cluster}'] = {
				'size': len(cluster_data),
				'size_percent': len(cluster_data) / len(df_with_clusters) * 100
			}
			if self._y is not None:
				target_counts = cluster_data[self._y.name].value_counts(normalize=True)
				for target, pct in target_counts.items():
					result[f'Cluster {cluster}'][f'{self._y.name}_{target}'] = pct * 100
			for col in self.X.columns:
				result[f'Cluster {cluster}'][f'avg_{col}'] = cluster_data[col].mean()
		return pd.DataFrame(result).T

	def visualize_clusters(self, df_with_clusters, save_path:Path=None):
		""" Visualize clusters in a given DataFrame using PCA for dimensionality reduction."""
		# Reduce dimensions to 3D 
		pca = PCA(n_components=3)
		X_pca = pca.fit_transform(
			StandardScaler().fit_transform(
				df_with_clusters.drop(columns=['Cluster']).select_dtypes(include=[int, float])
			)
		)
		
		# Plot projected data points
		fig = plt.figure(figsize=(15, 10))
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=df_with_clusters['Cluster'], 
					cmap='viridis', s=10, alpha=0.7)
		fig.colorbar(ax.collections[0], label='Cluster')
		fig.suptitle('Customer Segments Visualization (PCA)')
		ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
		ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
		ax.set_zlabel(f'PCA Component 3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
		
		# Save plot if filepath is provided
		if save_path:
			if not save_path.parent.exists():
				save_path.parent.mkdir(parents=True, exist_ok=True)
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"\nCluster 3D plot saved to {save_path}")
		else:
			plt.show()

	def save_model(self, filepath:Path):
		""" Save model to a file."""
		if not filepath.parent.exists():
			filepath.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(self.best_model, filepath)
		print(f"\nModel saved to {filepath}")

	def load_model(self, filepath:Path):
		""" Load model from a .pkl file."""
		if not filepath.exists():
			raise FileNotFoundError(f"Model file not found: {filepath}")
		self.best_model = joblib.load(filepath)
		print(f"\nModel loaded from {filepath}")



if __name__ == "__main__":
	...
	# Example usage
	# clustering = KMeansClustering(df, target_feature='Churn', sample_size=0.5)
	# df_with_clusters = clustering.fit_predict()


	
	
