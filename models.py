"""
Recommender Model Module
Contains the core recommender model using cosine similarity
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class LaptopRecommenderModel:
    """Core recommender model using content-based filtering"""
    
    def __init__(self, processed_df, feature_matrix):
        """
        Initialize the recommender model
        
        Parameters:
        -----------
        processed_df : pandas.DataFrame
            Preprocessed laptop data
        feature_matrix : numpy.ndarray
            Feature matrix for similarity calculation
        """
        self.processed_df = processed_df
        self.feature_matrix = feature_matrix
    
    def calculate_similarity(self, laptop_idx, candidate_indices=None):
        """
        Calculate cosine similarity between a laptop and all others
        
        Parameters:
        -----------
        laptop_idx : int
            Index of the reference laptop
        candidate_indices : list, optional
            Indices of candidate laptops to compare with.
            If None, compares with all laptops.
            
        Returns:
        --------
        numpy.ndarray
            Similarity scores
        """
        if candidate_indices is None:
            # Compare with all laptops
            similarities = cosine_similarity(
                [self.feature_matrix[laptop_idx]],
                self.feature_matrix
            )[0]
        else:
            # Compare only with candidate laptops
            candidate_matrix = self.feature_matrix[candidate_indices]
            similarities = cosine_similarity(
                [self.feature_matrix[laptop_idx]],
                candidate_matrix
            )[0]
        
        return similarities
    
    def get_similar_laptops(self, laptop_idx, n_recommendations=10):
        """
        Get similar laptops based on content-based filtering
        
        Parameters:
        -----------
        laptop_idx : int
            Index of the reference laptop
        n_recommendations : int
            Number of recommendations to return
            
        Returns:
        --------
        list
            List of dictionaries with similar laptop information
        """
        # Calculate cosine similarity
        similarities = self.calculate_similarity(laptop_idx)
        
        # Get top N similar laptops (excluding the input laptop itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'index': int(idx),
                'name': self.processed_df.iloc[idx]['Name'],
                'brand': self.processed_df.iloc[idx]['Brand'],
                'price': float(self.processed_df.iloc[idx]['Price']),
                'rating': float(self.processed_df.iloc[idx]['Rating']),
                'similarity_score': float(similarities[idx])
            })
        
        return recommendations
    
    def calculate_hybrid_score(self, similarities, ratings, similarity_weight=0.7):
        """
        Calculate hybrid score combining similarity and ratings
        
        Parameters:
        -----------
        similarities : numpy.ndarray
            Similarity scores
        ratings : numpy.ndarray
            Rating values
        similarity_weight : float
            Weight for similarity (0-1). Rating weight = 1 - similarity_weight
            
        Returns:
        --------
        numpy.ndarray
            Combined scores
        """
        # Normalize ratings to 0-1 scale
        ratings_norm = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-8)
        
        # Combined score: weighted combination
        combined_scores = similarity_weight * similarities + (1 - similarity_weight) * ratings_norm
        
        return combined_scores

