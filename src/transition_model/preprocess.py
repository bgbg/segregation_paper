"""Data preprocessing for voter transition analysis.

This module handles building A/B/Other/Abstain tensors from election data
for use in hierarchical ecological inference models.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, List, Optional


def load_config(config_path: str = "data/config.yaml") -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_harmonized_data(election_num: int, config: Dict) -> pd.DataFrame:
    """Load harmonized election data from parquet file.
    
    Args:
        election_num: Election number (e.g., 20, 21, etc.)
        config: Configuration dictionary
        
    Returns:
        Harmonized DataFrame
    """
    interim_path = Path(config['paths']['interim_data']) / f"harmonized_{election_num}.parquet"
    
    if not interim_path.exists():
        raise FileNotFoundError(f"Harmonized data not found: {interim_path}")
    
    return pd.read_parquet(interim_path)


def compute_categories(df: pd.DataFrame, party_columns: Dict[str, str] = None) -> pd.DataFrame:
    """Compute the four categories: Shas, Agudat Israel, Others, Abstained.
    
    Note: This function is deprecated in favor of data_harmonizer.compute_voter_categories()
    but kept for backward compatibility.
    
    Args:
        df: Election results DataFrame with party vote counts
        party_columns: Mapping of party names to column names (optional)
        
    Returns:
        DataFrame with computed categories
    """
    df = df.copy()
    
    # If categories already exist (from harmonization), return as is
    if all(col in df.columns for col in ['A_shas', 'B_agudat', 'Other', 'Abstained']):
        return df
    
    # Fallback to old computation for backward compatibility
    if party_columns is None:
        party_columns = {}
    
    # Extract party votes
    shas_votes = df[party_columns.get('shas', 'party_shas')].fillna(0)
    agudat_votes = df[party_columns.get('agudat_israel', 'party_agudat_israel')].fillna(0)
    
    # Compute categories with corrected abstained calculation
    df['A_shas'] = shas_votes
    df['B_agudat'] = agudat_votes
    df['Other'] = df['legal'] - shas_votes - agudat_votes
    df['Abstained'] = np.maximum(0, df['can_vote'] - df['legal'])  # Corrected: can_vote - legal
    
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
    
    # Determine station ID column name
    station_col = None
    for col in ['station_id', 'ballot_code', 'ballot_id']:
        if col in df1.columns and col in df2.columns:
            station_col = col
            break
    
    if station_col is None:
        # If no station ID column, assume rows are already aligned
        if len(df1) != len(df2):
            raise ValueError("DataFrames have different lengths and no station ID column found")
    else:
        # Align stations between elections
        common_stations = set(df1[station_col]) & set(df2[station_col])
        df1 = df1[df1[station_col].isin(common_stations)]
        df2 = df2[df2[station_col].isin(common_stations)]
        
        # Sort by station_id for alignment
        df1 = df1.sort_values(station_col)
        df2 = df2.sort_values(station_col)
    
    # Build tensors
    x1 = df1[categories].values
    x2 = df2[categories].values
    n1 = x1.sum(axis=1)
    n2 = x2.sum(axis=1)
    
    return x1, x2, n1, n2


def prepare_hierarchical_data(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    target_cities: List[str] = None,
    config: Dict = None
) -> Dict[str, np.ndarray]:
    """Prepare data for hierarchical model with city groupings.
    
    Args:
        df1: Election t data
        df2: Election t+1 data
        target_cities: Cities to model (if None, load from config)
        config: Configuration dictionary (if None, load default)
        
    Returns:
        Dictionary with tensors organized by city
    """
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Use target cities from config if not provided
    if target_cities is None:
        target_cities = config['cities']['target_cities']
    
    data = {}
    
    # Country-wide data
    x1_country, x2_country, n1_country, n2_country = build_station_tensors(df1, df2)
    data['country'] = {
        'x1': x1_country,
        'x2': x2_country, 
        'n1': n1_country,
        'n2': n2_country
    }
    
    # City-specific data - match by English name in lowercase
    for city in target_cities:
        # For harmonized data, we expect city filtering to have been applied already
        # So we look for any city that matches (case-insensitive)
        city_df1 = df1[df1.get('city_name', '').str.lower().str.contains(city.lower(), na=False)]
        city_df2 = df2[df2.get('city_name', '').str.lower().str.contains(city.lower(), na=False)]
        
        if len(city_df1) > 0 and len(city_df2) > 0:
            x1_city, x2_city, n1_city, n2_city = build_station_tensors(city_df1, city_df2)
            data[city] = {
                'x1': x1_city,
                'x2': x2_city,
                'n1': n1_city, 
                'n2': n2_city
            }
    
    return data


def prepare_transition_data(pair_tag: str, config_path: str = "data/config.yaml") -> Dict[str, np.ndarray]:
    """Prepare data for a specific transition pair using harmonized data.
    
    Args:
        pair_tag: Transition pair identifier (e.g., 'kn20_21')
        config_path: Path to configuration file
        
    Returns:
        Dictionary with hierarchical data for the transition pair
    """
    config = load_config(config_path)
    
    # Parse election numbers from pair tag
    if not pair_tag.startswith('kn') or '_' not in pair_tag:
        raise ValueError(f"Invalid pair tag format: {pair_tag}. Expected format: 'knXX_YY'")
    
    elections_part = pair_tag[2:]  # Remove 'kn' prefix
    election_t, election_t1 = elections_part.split('_')
    election_t, election_t1 = int(election_t), int(election_t1)
    
    # Load harmonized data for both elections
    df1 = load_harmonized_data(election_t, config)
    df2 = load_harmonized_data(election_t1, config)
    
    if df1.empty or df2.empty:
        raise ValueError(f"No harmonized data available for transition {pair_tag}")
    
    # Prepare hierarchical data
    return prepare_hierarchical_data(df1, df2, config=config)