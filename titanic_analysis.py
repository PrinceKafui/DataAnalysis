"""
Titanic Passenger Data Analysis
Author: Prince Kafui Amexo
Date: March 2026
Description: Comprehensive analysis of Titanic passenger data including
data cleaning, statistical analysis, and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TitanicDataAnalyzer:
    """A class to analyze Titanic passenger data"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.df = None
        self.cleaned_df = None
        print("=" * 60)
        print("TITANIC PASSENGER DATA ANALYSIS")
        print("=" * 60)
        
    def load_data(self):
        """Load the Titanic dataset from seaborn"""
        print("\n[1] LOADING DATASET...")
        print("-" * 40)
        
        try:
            # Load dataset from seaborn (built-in)
            self.df = sns.load_dataset('titanic')
            print(f"✓ Dataset loaded successfully!")
            print(f"  - Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            print(f"  - Columns: {list(self.df.columns)}")
            return self.df
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return None
    
    def explore_data(self):
        """Initial data exploration"""
        print("\n[2] INITIAL DATA EXPLORATION...")
        print("-" * 40)
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nData Info:")
        print(self.df.info())
        
        print("\nBasic Statistics (numerical columns):")
        print(self.df.describe())
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        missing_with_data = missing[missing > 0]
        if len(missing_with_data) > 0:
            print(missing_with_data)
        else:
            print("No missing values found")
        
    def clean_data(self):
        """Clean and prepare the dataset"""
        print("\n[3] DATA CLEANING...")
        print("-" * 40)
        
        # Create a copy to avoid modifying original
        self.cleaned_df = self.df.copy()
        
        # Check missing values before cleaning
        print("\nMissing values before cleaning:")
        missing_before = self.cleaned_df.isnull().sum()
        missing_before_with_data = missing_before[missing_before > 0]
        if len(missing_before_with_data) > 0:
            print(missing_before_with_data)
        else:
            print("None")
        
        # 1. Handle missing 'age' values - fill with median age
        median_age = self.cleaned_df['age'].median()
        self.cleaned_df['age'] = self.cleaned_df['age'].fillna(median_age)
        print(f"\n✓ Filled missing ages with median age: {median_age:.1f}")
        
        # 2. Handle missing 'embarked' values - fill with mode (most common)
        mode_embarked = self.cleaned_df['embarked'].mode()[0]
        self.cleaned_df['embarked'] = self.cleaned_df['embarked'].fillna(mode_embarked)
        print(f"✓ Filled missing embarked with mode: '{mode_embarked}'")
        
        # 3. Handle missing 'embark_town' - fill with mode
        mode_town = self.cleaned_df['embark_town'].mode()[0]
        self.cleaned_df['embark_town'] = self.cleaned_df['embark_town'].fillna(mode_town)
        print(f"✓ Filled missing embark_town with mode: '{mode_town}'")
        
        # 4. Drop 'deck' column (too many missing values - 77%)
        if 'deck' in self.cleaned_df.columns:
            self.cleaned_df = self.cleaned_df.drop('deck', axis=1)
            print("✓ Dropped 'deck' column (77% missing values)")
        
        # 5. Check for duplicates
        duplicates = self.cleaned_df.duplicated().sum()
        if duplicates > 0:
            self.cleaned_df = self.cleaned_df.drop_duplicates()
            print(f"✓ Removed {duplicates} duplicate rows")
        else:
            print("✓ No duplicate rows found")
        
        # Verify no missing values remain
        print("\nMissing values after cleaning:")
        missing_after = self.cleaned_df.isnull().sum()
        missing_after_with_data = missing_after[missing_after > 0]
        if len(missing_after_with_data) == 0:
            print("None - All missing values handled successfully!")
        else:
            print(missing_after_with_data)
        
        return self.cleaned_df
    
    def compute_statistics(self):
        """Compute and display summary statistics"""
        print("\n[4] SUMMARY STATISTICS")
        print("=" * 40)
        
        # Overall survival rate
        survival_rate = (self.cleaned_df['survived'].mean() * 100)
        print(f"\nOVERALL SURVIVAL RATE: {survival_rate:.1f}%")
        print(f"Total Passengers Analyzed: {len(self.cleaned_df)}")
        
        # Statistics by passenger class
        print("\n" + "=" * 40)
        print("STATISTICS BY PASSENGER CLASS")
        print("=" * 40)
        
        class_stats = self.cleaned_df.groupby('pclass').agg({
            'survived': ['mean', 'count'],
            'age': 'mean',
            'fare': 'mean'
        }).round(2)
        
        class_stats.columns = ['Survival Rate', 'Count', 'Avg Age', 'Avg Fare']
        class_stats['Survival Rate'] = class_stats['Survival Rate'] * 100
        print(class_stats)
        
        # Statistics by sex
        print("\n" + "=" * 40)
        print("STATISTICS BY SEX")
        print("=" * 40)
        
        sex_stats = self.cleaned_df.groupby('sex').agg({
            'survived': ['mean', 'count'],
            'age': 'mean',
            'fare': 'mean'
        }).round(2)
        
        sex_stats.columns = ['Survival Rate', 'Count', 'Avg Age', 'Avg Fare']
        sex_stats['Survival Rate'] = sex_stats['Survival Rate'] * 100
        print(sex_stats)
        
        # Statistics by class and sex combined
        print("\n" + "=" * 40)
        print("SURVIVAL RATE BY CLASS AND SEX (%)")
        print("=" * 40)
        
        class_sex_survival = self.cleaned_df.pivot_table(
            values='survived', 
            index='pclass', 
            columns='sex', 
            aggfunc=lambda x: (x.mean() * 100).round(1)
        )
        print(class_sex_survival)
        
        # Age distribution statistics
        print("\n" + "=" * 40)
        print("AGE DISTRIBUTION STATISTICS")
        print("=" * 40)
        
        print(f"Mean Age: {self.cleaned_df['age'].mean():.1f} years")
        print(f"Median Age: {self.cleaned_df['age'].median():.1f} years")
        print(f"Min Age: {self.cleaned_df['age'].min():.1f} years")
        print(f"Max Age: {self.cleaned_df['age'].max():.1f} years")
        print(f"Standard Deviation: {self.cleaned_df['age'].std():.1f} years")
        
        # Fare statistics
        print("\n" + "=" * 40)
        print("FARE STATISTICS")
        print("=" * 40)
        
        print(f"Mean Fare: ${self.cleaned_df['fare'].mean():.2f}")
        print(f"Median Fare: ${self.cleaned_df['fare'].median():.2f}")
        print(f"Min Fare: ${self.cleaned_df['fare'].min():.2f}")
        print(f"Max Fare: ${self.cleaned_df['fare'].max():.2f}")
        
        # Save statistics to file
        self._save_statistics()
        
    def _save_statistics(self):
        """Save statistics to a text file"""
        try:
            with open('output.txt', 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("TITANIC DATA ANALYSIS - STATISTICAL SUMMARY\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Dataset Shape: {self.cleaned_df.shape}\n")
                f.write(f"Total Passengers: {len(self.cleaned_df)}\n")
                f.write(f"Survival Rate: {self.cleaned_df['survived'].mean()*100:.1f}%\n\n")
                
                f.write("SURVIVAL BY CLASS:\n")
                class_stats = self.cleaned_df.groupby('pclass')['survived'].mean() * 100
                for pclass, rate in class_stats.items():
                    f.write(f"  Class {pclass}: {rate:.1f}%\n")
                
                f.write("\nSURVIVAL BY SEX:\n")
                sex_stats = self.cleaned_df.groupby('sex')['survived'].mean() * 100
                for sex, rate in sex_stats.items():
                    f.write(f"  {sex.capitalize()}: {rate:.1f}%\n")
                
                f.write("\nAGE STATISTICS:\n")
                f.write(f"  Mean: {self.cleaned_df['age'].mean():.1f}\n")
                f.write(f"  Median: {self.cleaned_df['age'].median():.1f}\n")
                f.write(f"  Range: {self.cleaned_df['age'].min():.1f} - {self.cleaned_df['age'].max():.1f}\n")
                
            print("\n✓ Statistics saved to 'output.txt'")
        except Exception as e:
            print(f"\n✗ Error saving statistics: {e}")
    
    def create_visualizations(self):
        """Create and save data visualizations"""
        print("\n[5] CREATING VISUALIZATIONS...")
        print("-" * 40)
        
        # Create visualizations directory
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')
            print("✓ Created 'visualizations' folder")
        
        # Visualization 1: Age Distribution Histogram
        print("\n1. Creating Age Distribution plot...")
        try:
            plt.figure(figsize=(12, 6))
            
            # Split by survival
            survived_ages = self.cleaned_df[self.cleaned_df['survived']==1]['age'].dropna()
            not_survived_ages = self.cleaned_df[self.cleaned_df['survived']==0]['age'].dropna()
            
            plt.hist([survived_ages, not_survived_ages], 
                    bins=20, 
                    label=['Survived', 'Did not survive'], 
                    alpha=0.7, 
                    edgecolor='black',
                    color=['#2ecc71', '#e74c3c'])
            
            plt.xlabel('Age (years)', fontsize=12)
            plt.ylabel('Number of Passengers', fontsize=12)
            plt.title('Age Distribution of Titanic Passengers by Survival', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('visualizations/age_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: visualizations/age_distribution.png")
        except Exception as e:
            print(f"  ✗ Error creating age distribution plot: {e}")
        
        # Visualization 2: Survival Rate by Passenger Class and Sex
              # Visualization 2: Survival Rate by Passenger Class and Sex
        print("2. Creating Class & Sex Survival plot...")
        try:
            plt.figure(figsize=(12, 6))
            
            # Calculate survival rates
            survival_by_class_sex = self.cleaned_df.groupby(['pclass', 'sex'])['survived'].mean().unstack() * 100
            
            # Create bar plot
            ax = survival_by_class_sex.plot(kind='bar', rot=0, color=['#3498db', '#e84342'])
            plt.xlabel('Passenger Class', fontsize=12)
            plt.ylabel('Survival Rate (%)', fontsize=12)
            plt.title('Survival Rate by Passenger Class and Sex', fontsize=14, fontweight='bold')
            plt.legend(title='Sex', loc='upper right')
            plt.xticks(rotation=0)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', padding=3)
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('visualizations/class_survival.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: visualizations/class_survival.png")
        except Exception as e:
            print(f"  ✗ Error creating class survival plot: {e}")
            print("  ✓ Saved: visualizations/class_survival.png")
        except Exception as e:
            print(f"  ✗ Error creating class survival plot: {e}")
        
        # Visualization 3: Fare vs Age scatter plot
        print("3. Creating Fare vs Age scatter plot...")
        try:
            plt.figure(figsize=(14, 8))
            
            # Create scatter plot with different colors for survival
            colors = {0: '#e74c3c', 1: '#2ecc71'}
            for survived, color in colors.items():
                subset = self.cleaned_df[self.cleaned_df['survived'] == survived]
                plt.scatter(subset['age'], subset['fare'], 
                           c=color, 
                           label='Survived' if survived else 'Did not survive',
                           alpha=0.6, 
                           edgecolors='black', 
                           linewidth=0.5,
                           s=50)  # point size
            
            plt.xlabel('Age (years)', fontsize=12)
            plt.ylabel('Fare ($)', fontsize=12)
            plt.title('Fare vs Age by Survival Status', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Use log scale for better visualization of fare distribution
            plt.yscale('log')
            
            # Add a horizontal line at median fare
            median_fare = self.cleaned_df['fare'].median()
            plt.axhline(y=median_fare, color='gray', linestyle='--', alpha=0.5, label=f'Median Fare (${median_fare:.2f})')
            
            plt.tight_layout()
            plt.savefig('visualizations/fare_vs_age.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: visualizations/fare_vs_age.png")
        except Exception as e:
            print(f"  ✗ Error creating fare vs age plot: {e}")
        
        # Visualization 4: Survival Heatmap (Bonus)
        print("4. Creating Survival Heatmap...")
        try:
            plt.figure(figsize=(10, 6))
            
            # Create a correlation matrix for numeric columns
            numeric_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
            corr_matrix = self.cleaned_df[numeric_cols].corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       square=True,
                       linewidths=1,
                       cbar_kws={"shrink": 0.8})
            
            plt.title('Correlation Heatmap of Titanic Features', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: visualizations/correlation_heatmap.png")
        except Exception as e:
            print(f"  ✗ Error creating heatmap: {e}")
        
        print("\n✓ All visualizations saved to 'visualizations/' folder")
        
    def generate_report(self):
        """Generate a final analysis report"""
        print("\n[6] FINAL ANALYSIS REPORT")
        print("=" * 60)
        
        print("\nKEY FINDINGS:")
        print("-" * 40)
        
        # Finding 1: Overall survival
        survival_rate = self.cleaned_df['survived'].mean() * 100
        print(f"1️⃣  Overall, only {survival_rate:.1f}% of passengers survived the Titanic disaster.")
        
        # Finding 2: Class disparity
        class1_rate = self.cleaned_df[self.cleaned_df['pclass']==1]['survived'].mean() * 100
        class2_rate = self.cleaned_df[self.cleaned_df['pclass']==2]['survived'].mean() * 100
        class3_rate = self.cleaned_df[self.cleaned_df['pclass']==3]['survived'].mean() * 100
        print(f"2️⃣  There was a significant class disparity: {class1_rate:.1f}% of first-class passengers survived, compared to {class2_rate:.1f}% in second-class and only {class3_rate:.1f}% in third-class.")
        
        # Finding 3: Gender disparity
        female_rate = self.cleaned_df[self.cleaned_df['sex']=='female']['survived'].mean() * 100
        male_rate = self.cleaned_df[self.cleaned_df['sex']=='male']['survived'].mean() * 100
        print(f"3️⃣  Gender played a crucial role: {female_rate:.1f}% of women survived, while only {male_rate:.1f}% of men survived.")
        
        # Finding 4: Age insights
        child_rate = self.cleaned_df[self.cleaned_df['age'] < 18]['survived'].mean() * 100
        adult_rate = self.cleaned_df[self.cleaned_df['age'] >= 18]['survived'].mean() * 100
        print(f"4️⃣  Children had a higher survival rate: {child_rate:.1f}% of children survived vs {adult_rate:.1f}% of adults.")
        
        # Finding 5: Fare impact
        high_fare = self.cleaned_df[self.cleaned_df['fare'] > self.cleaned_df['fare'].median()]['survived'].mean() * 100
        low_fare = self.cleaned_df[self.cleaned_df['fare'] <= self.cleaned_df['fare'].median()]['survived'].mean() * 100
        print(f"5️⃣  Passengers who paid above-median fares had a {high_fare:.1f}% survival rate, compared to {low_fare:.1f}% for those paying below-median fares.")
        
        # Finding 6: Family size impact (Bonus)
        print(f"6️⃣  Family size mattered: Passengers alone had a {self.cleaned_df[self.cleaned_df['alone']==True]['survived'].mean()*100:.1f}% survival rate, while those with family had {self.cleaned_df[self.cleaned_df['alone']==False]['survived'].mean()*100:.1f}%.")
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE - Check the 'visualizations' folder for charts!")
        print("=" * 60)
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "=" * 60)
        print("STARTING TITANIC DATA ANALYSIS")
        print("=" * 60)
        
        # Load data
        if self.load_data() is None:
            print("✗ Failed to load data. Exiting.")
            return
        
        # Explore data
        self.explore_data()
        
        # Clean data
        self.clean_data()
        
        # Compute statistics
        self.compute_statistics()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("✓ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nOutput files created:")
        print("  - output.txt (statistical summary)")
        print("  - visualizations/age_distribution.png")
        print("  - visualizations/class_survival.png")
        print("  - visualizations/fare_vs_age.png")
        print("  - visualizations/correlation_heatmap.png")

# Main execution
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = TitanicDataAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis()