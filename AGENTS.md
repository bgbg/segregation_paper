# AGENTS.md

This document describes the available tools and capabilities for the Voter Segregation Analysis project.

## Data Analysis Tools

### Core Analysis Functions

#### `dissimilarity_analysis.py`
Core data loading and Monte Carlo simulation functions for segregation analysis

#### `vote_utils.py`
Segregation metrics calculation and statistical testing utilities

#### `plot_utils.py`
Hebrew text support and matplotlib styling utilities

## Data Sources

### Election Data
- Knesset elections 15-25 (2003-2025)
- Ballot box level results
- City and neighborhood breakdowns
- Party vote counts

### Data Structure
- `columns.csv`: Hebrew-to-English column mapping for election data
- Raw CSV files with Hebrew column names
- Normalized DataFrames with English column names

### Demographic Data
- Israeli cities (Hebrew/English names)
- Haredi population estimates
- Geographic boundaries

## Analysis Capabilities

### Segregation Metrics
- **Dissimilarity Index**: Measures spatial segregation between groups
- **Duncan & Duncan Index**: Alternative dissimilarity measure
- **Homogeneity Analysis**: Identifies areas with concentrated voting patterns
- **Statistical Significance**: Monte Carlo simulations for p-values

## Coding Instructions

- **CLI Parsing**: Prefer `defopt` for command-line argument parsing
- **Code Style**: Follow PEP 8 and Black formatting conventions
- **Error Handling**: Don't catch exceptions unless you can fix them. Prefer early failure
- **Code Design**: Prefer simplicity and readability. Don't overengineer

## GitHub Interactions

- **GitHub CLI**: Use `gh` command to interact with GitHub
- **Pull Requests**: Don't use emojis in PR titles or descriptions
- **Commit Messages**: Don't use emojis or sign with "created with AI" or similar

## Dependencies

### Core Libraries
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scipy`: Statistical functions
- `matplotlib`: Basic plotting
- `seaborn`: Statistical visualizations

### Specialized Libraries
- `cachier`: Function result caching
- `tqdm`: Progress bars
- `bidi`: Bidirectional text support

## Data Processing Pipeline

1. **Data Loading**: Raw CSV files → Cleaned DataFrames
2. **Preprocessing**: Column normalization, missing data handling
3. **Analysis**: Segregation calculations, statistical tests
4. **Visualization**: Maps, charts, statistical plots
5. **Export**: Results to various formats

