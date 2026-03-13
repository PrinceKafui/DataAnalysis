# Titanic Passenger Data Analysis

## Overview
As a data enthusiast and software developer, I created this comprehensive data analysis project to deepen my understanding of statistical analysis and data visualization using Python. This project analyzes the famous Titanic dataset to uncover patterns in passenger survival based on factors like age, class, gender, and fare.

The software performs a complete data analysis pipeline including:
- Data loading and exploration
- Data cleaning and preprocessing
- Statistical summary computation
- Multiple data visualizations
- Insight generation and reporting

[Software Demo Video]()

## Data Source
The analysis uses the Titanic dataset from Seaborn's built-in datasets, which contains information about 891 passengers including:
- Survival status (0 = No, 1 = Yes)
- Passenger class (1st, 2nd, 3rd)
- Name, Sex, Age
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Ticket number and fare
- Cabin number and port of embarkation

## Visualizations Created
1. **Age Distribution Histogram** - Shows age distribution of passengers, separated by survival status
2. **Survival by Class and Sex** - Bar chart comparing survival rates across passenger classes and genders
3. **Fare vs Age Scatter Plot** - Explores relationship between fare paid, passenger age, and survival

## Key Findings
- Only 38.4% of passengers survived the disaster
- First-class passengers had a 62.9% survival rate vs 24.2% for third-class
- Women survived at a much higher rate (74.2%) than men (18.9%)
- Children (<18) had better survival odds (53.4%) than adults (37.2%)
- Passengers paying higher fares had significantly better survival rates

## Development Environment

**Tools Used:**
- Python 3.9+
- Visual Studio Code
- Git for version control
- Jupyter Notebook (for initial exploration)

**Libraries:**
- Pandas - Data manipulation and analysis
- Matplotlib - Data visualization
- Seaborn - Statistical data visualization
- NumPy - Numerical computing

## Installation and Usage

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the analysis: `python titanic_analysis.py`
4. View outputs:
   - Console output for statistical summaries
   - `visualizations/` folder for saved plots
   - `output.txt` for saved statistics

## Useful Websites
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Titanic Dataset Description](https://www.kaggle.com/c/titanic/data)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)