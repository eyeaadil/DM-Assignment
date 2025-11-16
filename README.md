# Titanic Survival Prediction

This project performs a comprehensive analysis of the Titanic dataset, including exploratory data analysis, visualization, and predictive modeling to determine passenger survival.

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the analysis script:
```bash
python titanic_analysis.py
```

This will:
1. Download the Titanic dataset (if not already present)
2. Perform exploratory data analysis
3. Generate visualizations (saved as PNG files)
4. Train a Random Forest classifier
5. Display model performance metrics

## Output Files

- `titanic_eda.png`: Exploratory data analysis visualizations
- `feature_importance.png`: Feature importance plot from the model
- `titanic.csv`: Local copy of the dataset

## Project Structure

- `titanic_analysis.py`: Main analysis script
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter (optional, for interactive exploration)
# DM-Assignment
