"""
Recommendation Engine Module
Handles filtering, ranking, and generating recommendations
"""

import pandas as pd
import numpy as np
from models import LaptopRecommenderModel


class RecommendationEngine:
    """Handles recommendation logic and filtering"""
    
    def __init__(self, processed_df, feature_matrix, model):
        """
        Initialize the recommendation engine
        
        Parameters:
        -----------
        processed_df : pandas.DataFrame
            Preprocessed laptop data
        feature_matrix : numpy.ndarray
            Feature matrix
        model : LaptopRecommenderModel
            The recommender model instance
        """
        self.processed_df = processed_df
        self.feature_matrix = feature_matrix
        self.model = model
    
    def filter_by_preferences(self, max_price=None, min_rating=None, 
                             brand=None, min_ram=None, min_storage=None,
                             processor_brand=None, graphics_brand=None,
                             use_case=None):
        """
        Knowledge-based filtering by user preferences
        
        Parameters:
        -----------
        max_price : float, optional
            Maximum price filter
        min_rating : float, optional
            Minimum rating filter
        brand : str, optional
            Brand filter
        min_ram : int, optional
            Minimum RAM filter
        min_storage : int, optional
            Minimum storage filter
        processor_brand : str, optional
            Processor brand filter
        graphics_brand : str, optional
            Graphics brand filter
        use_case : str, optional
            Use case filter ('gaming', 'business', 'budget')
            
        Returns:
        --------
        list
            List of indices matching the filters
        """
        df = self.processed_df.copy()
        
        # Apply filters
        if max_price is not None:
            df = df[df['Price'] <= max_price]
        
        if min_rating is not None:
            df = df[df['Rating'] >= min_rating]
        
        if brand is not None:
            df = df[df['Brand'].str.contains(brand, case=False, na=False)]
        
        if min_ram is not None:
            df = df[df['RAM_GB'] >= min_ram]
        
        if min_storage is not None:
            df = df[df['Storage_capacity_GB'] >= min_storage]
        
        if processor_brand is not None:
            df = df[df['Processor_brand'].str.contains(processor_brand, case=False, na=False)]
        
        if graphics_brand is not None:
            df = df[df['Graphics_brand'].str.contains(graphics_brand, case=False, na=False)]
        
        # Use case based filtering
        if use_case == 'gaming':
            # Gaming laptops: NVIDIA (≥4GB) OR AMD RX dedicated (≥4GB)
            has_nvidia = (
                df['Graphics_brand'].str.contains('NVIDIA', case=False, na=False) &
                (df['Graphics_GB'] >= 4)
            )
            has_amd_dedicated = (
                df['Graphics_brand'].str.contains('AMD', case=False, na=False) & 
                (df['Graphics_GB'] >= 4) &
                (df['Graphics_name'].str.contains('RX|Radeon RX', case=False, na=False))
            )
            df = df[has_nvidia | has_amd_dedicated]
        elif use_case == 'business':
            # Business laptops: good RAM
            df = df[df['RAM_GB'] >= 8]
        elif use_case == 'budget':
            # Budget laptops: lower price
            df = df[df['Price'] <= 50000]
        
        return df.index.tolist()
    
    def hybrid_recommend(self, laptop_idx=None, max_price=None, min_rating=None,
                        brand=None, min_ram=None, min_storage=None,
                        processor_brand=None, graphics_brand=None,
                        use_case=None, n_recommendations=10,
                        similarity_weight=0.7):
        """
        Hybrid recommendation combining content-based and knowledge-based filtering
        
        Parameters:
        -----------
        laptop_idx : int, optional
            Index of reference laptop
        max_price : float, optional
            Maximum price filter
        min_rating : float, optional
            Minimum rating filter
        brand : str, optional
            Brand filter
        min_ram : int, optional
            Minimum RAM filter
        min_storage : int, optional
            Minimum storage filter
        processor_brand : str, optional
            Processor brand filter
        graphics_brand : str, optional
            Graphics brand filter
        use_case : str, optional
            Use case filter
        n_recommendations : int
            Number of recommendations
        similarity_weight : float
            Weight for similarity in hybrid score (0-1)
            
        Returns:
        --------
        list or str
            List of recommendation dictionaries or error message
        """
        # Step 1: Knowledge-based filtering
        filtered_indices = self.filter_by_preferences(
            max_price=max_price, min_rating=min_rating, brand=brand,
            min_ram=min_ram, min_storage=min_storage,
            processor_brand=processor_brand, graphics_brand=graphics_brand,
            use_case=use_case
        )
        
        if len(filtered_indices) == 0:
            return "No laptops found matching your criteria. Please try relaxing your filters."
        
        # Step 2: Content-based similarity (if reference laptop provided)
        if laptop_idx is not None and laptop_idx in self.processed_df.index:
            # Calculate similarity only for filtered laptops
            similarities = self.model.calculate_similarity(laptop_idx, filtered_indices)
            
            # Get ratings for filtered laptops
            ratings = self.processed_df.iloc[filtered_indices]['Rating'].values
            
            # Calculate hybrid score
            combined_scores = self.model.calculate_hybrid_score(
                similarities, ratings, similarity_weight
            )
            
            # Get top recommendations
            top_indices = np.argsort(combined_scores)[::-1][:n_recommendations]
            recommended_indices = [filtered_indices[i] for i in top_indices]
        else:
            # If no reference laptop, rank by rating and price
            filtered_df = self.processed_df.iloc[filtered_indices].copy()
            max_price_val = filtered_df['Price'].max()
            price_score = 1 - (filtered_df['Price'] / max_price_val)
            combined_scores = filtered_df['Rating'] * 0.7 + price_score * 0.3
            
            top_indices = combined_scores.nlargest(n_recommendations).index
            recommended_indices = top_indices.tolist()
        
        # Format recommendations
        recommendations = self._format_recommendations(recommended_indices)
        
        return recommendations
    
    def _format_recommendations(self, indices):
        """
        Format recommendation results
        
        Parameters:
        -----------
        indices : list
            List of laptop indices
            
        Returns:
        --------
        list
            List of formatted recommendation dictionaries
        """
        recommendations = []
        for idx in indices:
            row = self.processed_df.iloc[idx]
            recommendations.append({
                'name': row['Name'],
                'brand': row['Brand'],
                'price': f"₹{row['Price']:,.0f}",
                'rating': row['Rating'],
                'ram': f"{row['RAM_GB']} GB",
                'storage': f"{row['Storage_capacity_GB']} GB {row['Storage_type']}",
                'processor': f"{row['Processor_name']} {row.get('Processor_variant', '')}",
                'graphics': row['Graphics_name'],
                'display': f"{row['Display_size_inches']}\" ({row['Horizontal_pixel']}x{row['Vertical_pixel']})"
            })
        
        return recommendations
    
    def find_laptop_by_name(self, laptop_name):
        """
        Find laptop index by name (partial match)
        
        Parameters:
        -----------
        laptop_name : str
            Name or partial name of laptop
            
        Returns:
        --------
        int or None
            Index of matching laptop, or None if not found
        """
        matches = self.processed_df[
            self.processed_df['Name'].str.contains(laptop_name, case=False, na=False)
        ]
        if len(matches) > 0:
            return matches.index[0]
        return None

