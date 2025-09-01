"""Data harmonization pipeline for voter transition analysis.

This module handles the complete data harmonization process:
- Loading raw election CSV files
- Applying Hebrew-to-English column mappings
- Computing voter categories with corrected abstained calculation
- Applying homogenic filtering
- Filtering for target cities
- Saving harmonized data as parquet files
"""

import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.vote_utils import is_homogenic

logger = logging.getLogger(__name__)


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


def load_column_mappings(columns_path: str) -> Dict[str, str]:
    """Load Hebrew-to-English column mappings.
    
    Args:
        columns_path: Path to columns mapping CSV file
        
    Returns:
        Dictionary mapping Hebrew column names to English equivalents
    """
    df = pd.read_csv(columns_path, encoding='utf-8')
    # Create mapping from Hebrew to English
    mapping = dict(zip(df['heb'], df['eng']))
    return mapping


def load_city_mappings(cities_path: str) -> Dict[str, str]:
    """Load Hebrew-to-English city name mappings.
    
    Args:
        cities_path: Path to cities mapping CSV file
        
    Returns:
        Dictionary mapping Hebrew city names to English equivalents
    """
    df = pd.read_csv(cities_path, encoding='utf-8')
    # Create mapping from Hebrew to English (lowercase for matching)
    mapping = dict(zip(df['city_name'], df['city_name_english'].str.lower()))
    return mapping


def apply_column_mappings(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Apply Hebrew-to-English column name mappings to DataFrame.
    
    Args:
        df: Raw DataFrame with Hebrew column names
        column_mapping: Hebrew to English column mapping
        
    Returns:
        DataFrame with English column names
    """
    # Create reverse mapping for columns that exist in the DataFrame
    available_mappings = {heb: eng for heb, eng in column_mapping.items() 
                         if heb in df.columns}
    
    if not available_mappings:
        logger.warning("No column mappings found for this DataFrame")
        return df
    
    # Apply mapping
    df_mapped = df.rename(columns=available_mappings)
    logger.info(f"Mapped {len(available_mappings)} columns from Hebrew to English")
    
    return df_mapped


def compute_voter_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the four voter categories: Shas, Agudat Israel, Others, Abstained.
    
    This implements the corrected abstained computation:
    abstained = can_vote - legal (treats invalid votes as abstention)
    
    Args:
        df: DataFrame with election data and English column names
        
    Returns:
        DataFrame with computed voter categories
    """
    df = df.copy()
    
    # Extract party votes (handle missing columns gracefully)
    shas_votes = df.get('party_shas', 0)
    agudat_votes = df.get('party_agudat_israel', 0) 
    
    # Handle missing values
    if isinstance(shas_votes, pd.Series):
        shas_votes = shas_votes.fillna(0)
    if isinstance(agudat_votes, pd.Series):
        agudat_votes = agudat_votes.fillna(0)
    
    # Compute categories
    df['A_shas'] = shas_votes
    df['B_agudat'] = agudat_votes
    df['Other'] = df['legal'] - shas_votes - agudat_votes
    
    # Corrected abstained computation: can_vote - legal
    # This treats invalid/void votes as abstention
    df['Abstained'] = np.maximum(0, df['can_vote'] - df['legal'])
    
    # Ensure non-negative values
    df['A_shas'] = np.maximum(0, df['A_shas'])
    df['B_agudat'] = np.maximum(0, df['B_agudat']) 
    df['Other'] = np.maximum(0, df['Other'])
    
    return df


def apply_homogenic_filtering(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """Apply homogenic filtering to keep only polling stations with high Haredi concentration.
    
    Args:
        df: DataFrame with voter categories
        threshold: Minimum fraction of Haredi votes (Shas + Agudat Israel)
        
    Returns:
        Filtered DataFrame
    """
    # Check if we have the required columns
    haredi_parties = ['A_shas', 'B_agudat']
    if not all(col in df.columns for col in haredi_parties):
        logger.warning("Cannot apply homogenic filtering - missing Haredi party columns")
        return df
    
    # Calculate Haredi fraction directly
    haredi_votes = df['A_shas'] + df['B_agudat']
    haredi_fraction = haredi_votes / df['legal']
    
    # Apply homogenic filter
    homogenic_mask = haredi_fraction > threshold
    
    df_filtered = df[homogenic_mask].copy()
    
    logger.info(f"Homogenic filtering: kept {len(df_filtered)}/{len(df)} stations "
               f"({len(df_filtered)/len(df)*100:.1f}%) with >{threshold*100}% Haredi votes")
    
    return df_filtered


def filter_target_cities(df: pd.DataFrame, target_cities: List[str], 
                        city_mapping: Dict[str, str]) -> pd.DataFrame:
    """Filter DataFrame to include only target cities.
    
    Args:
        df: DataFrame with city information
        target_cities: List of target city names (English, lowercase)
        city_mapping: Hebrew to English city name mapping
        
    Returns:
        Filtered DataFrame with only target cities
    """
    if 'city_name' not in df.columns:
        logger.warning("Cannot filter cities - no city_name column found")
        return df
    
    # Create reverse mapping for matching
    english_to_hebrew = {eng.lower(): heb for heb, eng in city_mapping.items()}
    
    # Find Hebrew names for target cities
    target_hebrew_cities = []
    for city in target_cities:
        city_lower = city.lower().strip()
        if city_lower in english_to_hebrew:
            target_hebrew_cities.append(english_to_hebrew[city_lower])
        else:
            logger.warning(f"Target city '{city}' not found in city mapping")
    
    if not target_hebrew_cities:
        logger.warning("No target cities found in mapping - returning all cities")
        return df
    
    # Filter for target cities
    city_mask = df['city_name'].isin(target_hebrew_cities)
    df_filtered = df[city_mask].copy()
    
    logger.info(f"City filtering: kept {len(df_filtered)}/{len(df)} stations "
               f"from {len(target_hebrew_cities)} target cities")
    
    return df_filtered


def harmonize_election_data(election_num: int, config: Dict, 
                          force: bool = False) -> pd.DataFrame:
    """Harmonize data for a single election.
    
    Args:
        election_num: Election number (e.g., 20, 21, 22, etc.)
        config: Configuration dictionary
        force: Whether to force reprocessing even if output exists
        
    Returns:
        Harmonized DataFrame
    """
    # Define paths
    raw_path = Path(config['paths']['raw_data']) / f"results_{election_num}.csv"
    interim_path = Path(config['paths']['interim_data']) / f"harmonized_{election_num}.parquet"
    
    # Check if output already exists
    if interim_path.exists() and not force:
        logger.info(f"Harmonized data for election {election_num} already exists, loading...")
        return pd.read_parquet(interim_path)
    
    # Check if raw data exists
    if not raw_path.exists():
        logger.warning(f"Raw data file not found: {raw_path}")
        return pd.DataFrame()  # Return empty DataFrame
    
    logger.info(f"Processing election {election_num} data...")
    
    # Load raw data
    df = pd.read_csv(raw_path, encoding='utf-8')
    logger.info(f"Loaded {len(df)} stations from {raw_path}")
    
    # Load mappings
    column_mapping = load_column_mappings(config['paths']['columns_mapping'])
    city_mapping = load_city_mappings(config['paths']['cities_mapping'])
    
    # Apply column mappings
    df = apply_column_mappings(df, column_mapping)
    
    # Compute voter categories with corrected abstained calculation
    df = compute_voter_categories(df)
    
    # Apply homogenic filtering if enabled
    if config['data']['homogenic_filtering']['enabled']:
        threshold = config['data']['homogenic_filtering']['threshold']
        df = apply_homogenic_filtering(df, threshold)
    
    # Filter for target cities
    target_cities = config['cities']['target_cities']
    df = filter_target_cities(df, target_cities, city_mapping)
    
    # Add election number for reference
    df['election'] = election_num
    
    # Ensure interim directory exists
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save harmonized data
    df.to_parquet(interim_path, index=False)
    logger.info(f"Saved harmonized data to {interim_path}")
    
    return df


def harmonize_all_elections(config_path: str = "data/config.yaml", 
                           force: bool = False) -> Dict[int, pd.DataFrame]:
    """Harmonize data for all configured elections.
    
    Args:
        config_path: Path to configuration file
        force: Whether to force reprocessing
        
    Returns:
        Dictionary mapping election numbers to harmonized DataFrames
    """
    config = load_config(config_path)
    elections = config['data']['elections']
    
    harmonized_data = {}
    
    for election_num in elections:
        df = harmonize_election_data(election_num, config, force)
        if not df.empty:
            harmonized_data[election_num] = df
            logger.info(f"Successfully harmonized election {election_num}: {len(df)} stations")
        else:
            logger.warning(f"No data available for election {election_num}")
    
    logger.info(f"Harmonization complete: processed {len(harmonized_data)} elections")
    return harmonized_data


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test harmonization with available data
    logger.info("Testing data harmonization pipeline...")
    
    harmonized_data = harmonize_all_elections(force=False)
    
    if harmonized_data:
        logger.info("Harmonization test successful!")
        for election_num, df in harmonized_data.items():
            logger.info(f"Election {election_num}: {len(df)} stations, "
                       f"{df['A_shas'].sum():.0f} Shas, "
                       f"{df['B_agudat'].sum():.0f} Agudat, "
                       f"{df['Other'].sum():.0f} Other, "
                       f"{df['Abstained'].sum():.0f} Abstained")
    else:
        logger.warning("No election data was harmonized - check if raw data files exist")