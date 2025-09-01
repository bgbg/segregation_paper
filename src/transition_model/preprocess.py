"""Data preprocessing for voter transition analysis.

This module handles building A/B/Other/Abstain tensors from election data
for use in hierarchical ecological inference models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


def compute_categories(df: pd.DataFrame, party_columns: Dict[str, str]) -> pd.DataFrame:
    """Compute the four categories: Shas, Agudat Israel, Others, Abstained.
    
    Args:
        df: Election results DataFrame with party vote counts
        party_columns: Mapping of party names to column names
        
    Returns:
        DataFrame with computed categories
    """
    df = df.copy()
    
    # Extract party votes
    shas_votes = df[party_columns.get('shas', 'party_shas')].fillna(0)
    agudat_votes = df[party_columns.get('agudat_israel', 'party_agudat_israel')].fillna(0)
    
    # Compute categories
    df['A_shas'] = shas_votes
    df['B_agudat'] = agudat_votes
    df['Other'] = df['legal'] - shas_votes - agudat_votes
    df['Abstained'] = np.maximum(0, df['can_vote'] - df['legal'])
    
    return df


def build_station_tensors(
    df1: pd.DataFrame, 
    df2: pd.DataFrame,
    target_cities: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build station-level tensors for adjacent election pairs.
    
    Args:
        df1: Election t data with computed categories
        df2: Election t+1 data with computed categories  
        target_cities: Cities to include (None for all)
        
    Returns:
        Tuple of (x1, x2, n1, n2) tensors shaped [stations, 4]
    """
    categories = ['A_shas', 'B_agudat', 'Other', 'Abstained']
    
    # Filter cities if specified
    if target_cities:
        df1 = df1[df1['city'].isin(target_cities)]
        df2 = df2[df2['city'].isin(target_cities)]
    
    # Align stations between elections
    common_stations = set(df1['station_id']) & set(df2['station_id'])
    df1 = df1[df1['station_id'].isin(common_stations)]
    df2 = df2[df2['station_id'].isin(common_stations)]
    
    # Sort by station_id for alignment
    df1 = df1.sort_values('station_id')
    df2 = df2.sort_values('station_id')
    
    # Build tensors
    x1 = df1[categories].values
    x2 = df2[categories].values
    n1 = x1.sum(axis=1)
    n2 = x2.sum(axis=1)
    
    return x1, x2, n1, n2


def prepare_hierarchical_data(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    target_cities: List[str]
) -> Dict[str, np.ndarray]:
    """Prepare data for hierarchical model with city groupings.
    
    Args:
        df1: Election t data
        df2: Election t+1 data
        target_cities: Cities to model
        
    Returns:
        Dictionary with tensors organized by city
    """
    data = {}
    
    # Country-wide data
    x1_country, x2_country, n1_country, n2_country = build_station_tensors(df1, df2)
    data['country'] = {
        'x1': x1_country,
        'x2': x2_country, 
        'n1': n1_country,
        'n2': n2_country
    }
    
    # City-specific data
    for city in target_cities:
        city_df1 = df1[df1['city'] == city]
        city_df2 = df2[df2['city'] == city]
        
        if len(city_df1) > 0 and len(city_df2) > 0:
            x1_city, x2_city, n1_city, n2_city = build_station_tensors(city_df1, city_df2)
            data[city] = {
                'x1': x1_city,
                'x2': x2_city,
                'n1': n1_city, 
                'n2': n2_city
            }
    
    return data