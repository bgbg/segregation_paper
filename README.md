# Voter Segregation Analysis in Israeli Elections

This repository contains the code and analysis used in the paper:

**Ethnic Divisions Within Unity: Insights into Intra-Group Segregation from Israel's Ultra-Orthodox Society**  
Published in Social Sciences, 2025, 14(3), 169  
https://doi.org/10.3390/socsci14030169

## Project Overview

This project analyzes voting patterns and segregation in Israeli elections, with a particular focus on Ultra-Orthodox (Haredi) communities. The analysis uses dissimilarity indices and other segregation metrics to quantify the degree of voter segregation across different cities and elections.

The project examines:
- Spatial segregation of Ultra-Orthodox voters in Israeli cities
- Temporal trends in voter segregation across multiple Knesset elections
- Factors associated with higher or lower levels of voter segregation
- Statistical significance of observed segregation patterns

## Main Findings

The research examines intra-group segregation patterns within Israel's Ultra-Orthodox society, revealing:

- Significant ethnic-based divisions within the seemingly unified Ultra-Orthodox community
- Distinct patterns of residential and social segregation between different Ultra-Orthodox subgroups
- Analysis of how these divisions manifest while maintaining overall religious group cohesion
- Quantitative evidence of intra-group segregation patterns
- Insights into how social unity and ethnic segregation can coexist within religious communities

For detailed findings and methodology, please refer to the [published paper](https://doi.org/10.3390/socsci14030169).

## Repository Structure

- `src/`: Source code for the analysis
  - `dissimilarity_analysis.py`: Core functions for calculating and analyzing dissimilarity indices
  - `vote_utils.py`: Utilities for processing voting data and calculating various segregation metrics
  - `plot_utils.py`: Utilities for plotting and visualization, including handling bidirectional text

- `notebooks/`: Jupyter notebooks containing the analysis
  - `voter_separation.ipynb`: Main analysis of voter segregation patterns
  - `voter_participation.ipynb`: Analysis of voter participation patterns
  - `explore_ralab_separation.ipynb`: Exploration of specific separation patterns
  - `presentation.ipynb`: Visualizations and findings for presentation purposes

- `data/`: Data files used in the analysis
  - `external/`: External data sources including election results and demographic information

## Dependencies

The project requires the following main dependencies:

```
numpy
pandas
pyarrow
scikit-learn
statsmodels
seaborn
matplotlib
cachier
tqdm
jupyter
```

A complete list of dependencies is available in the `requirements.txt` file.

## Setup and Usage

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks to reproduce the analysis:
   ```
   jupyter notebook notebooks/voter_separation.ipynb
   ```

## Citation

If you use this code or analysis in your research, please cite the original paper:

```bibtex
@article{gorelik2025ethnic,
    title={Ethnic Divisions Within Unity: Insights into Intra-Group Segregation from Israel's Ultra-Orthodox Society},
    author={Gorelik, Boris},
    journal={Social Sciences},
    volume={14},
    number={3},
    pages={169},
    year={2025},
    publisher={MDPI},
    doi={10.3390/socsci14030169}
}
```

Or in text format:

Gorelik, B. (2025). Ethnic Divisions Within Unity: Insights into Intra-Group Segregation from Israel's Ultra-Orthodox Society. *Social Sciences*, *14*(3), 169. https://doi.org/10.3390/socsci14030169

## License

See the [LICENSE](LICENSE) file for details.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 