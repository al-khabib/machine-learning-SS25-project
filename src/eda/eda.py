import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# %% Load and examine dataset


def load_and_examine_data():
    # Load the dataset
    df = pd.read_csv('bodyPerformance.csv')

    print(f"Dataset Shape: {df.shape}")
    print(f"Features: {df.shape[1]-1}, Samples: {df.shape[0]}")
    print(f"\nColumns: {list(df.columns)}")

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nBasic Statistics:")
    print(df.describe().round(2))

    return df

# %% Data quality assessment


def assess_data_quality(df):

    # Check for missing values
    print("Missing Values:")
    missing_values = df.isnull().sum()
    print(missing_values)

    # Check for zero values (potential measurement errors)
    print("\nZero Values Analysis (Potential Measurement Errors):")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    zero_analysis = {}

    for col in numerical_cols:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            zero_percentage = (zero_count/len(df)*100)
            print(f"{col:25}: {zero_count:4d} zeros ({zero_percentage:5.2f}%)")
            zero_analysis[col] = {'count': zero_count,
                                  'percentage': zero_percentage}

    # Check for negative flexibility values (potentially valid)
    negative_flexibility = (df['sit and bend forward_cm'] < 0).sum()
    print(
        f"\nNegative flexibility values: {negative_flexibility} ({negative_flexibility/len(df)*100:.2f}%)")
    print("Note: Negative values might represent limited flexibility (valid measurements)")

    # Check for extreme outliers
    print("\nPotential Extreme Outliers:")
    extreme_outliers = {
        'Very high flexibility (>100cm)': (df['sit and bend forward_cm'] > 100).sum(),
        'Very high sit-ups (>75)': (df['sit-ups counts'] > 75).sum(),
        'Very high body fat (>50%)': (df['body fat_%'] > 50).sum(),
        'Very low body fat (<5%)': (df['body fat_%'] < 5).sum()
    }

    for condition, count in extreme_outliers.items():
        print(f"{condition:30}: {count}")

    # Data quality summary
    samples_with_bio_zeros = len(df[(df['diastolic'] == 0) | (df['systolic'] == 0) |
                                    (df['gripForce'] == 0) | (df['broad jump_cm'] == 0)])

    print(f"\nDATA QUALITY SUMMARY:")
    print(f"Total samples: {len(df):,}")
    print(
        f"Samples with biological measurement zeros: {samples_with_bio_zeros}")
    print(f"Samples with negative flexibility: {negative_flexibility}")
    print(f"High quality samples: {len(df) - samples_with_bio_zeros:,}")
    print(
        f"Overall data quality: {((len(df) - samples_with_bio_zeros)/len(df)*100):.1f}%")

    return zero_analysis, extreme_outliers

# %% Target variable analysis


def analyze_target_variable(df):
    """Analyze the target variable (class) distribution"""
    print("\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS")
    print("="*60)

    # Class distribution
    class_counts = df['class'].value_counts().sort_index()
    class_percentages = df['class'].value_counts(
        normalize=True).sort_index() * 100

    print("Class Distribution:")
    for class_name in ['A', 'B', 'C', 'D']:
        count = class_counts[class_name]
        percentage = class_percentages[class_name]
        print(
            f"Class {class_name} (Performance Level): {count:,} samples ({percentage:.1f}%)")

    print(f"\nClass Balance Assessment:")
    max_diff = class_counts.max() - class_counts.min()
    print(f"Maximum difference between classes: {max_diff} samples")
    print(f"Dataset is {'well-balanced' if max_diff < 50 else 'imbalanced'}")

    return class_counts, class_percentages

# %% Gender analysis


def analyze_gender_distribution(df):
    """Analyze gender distribution and its relationship with performance"""
    print("\n" + "="*60)
    print("GENDER DISTRIBUTION ANALYSIS")
    print("="*60)

    # Basic gender distribution
    gender_counts = df['gender'].value_counts()
    gender_percentages = df['gender'].value_counts(normalize=True) * 100

    print("Gender Distribution:")
    for gender in ['M', 'F']:
        count = gender_counts[gender]
        percentage = gender_percentages[gender]
        print(f"{gender:1} ({'Male' if gender == 'M' else 'Female'}): {count:,} samples ({percentage:.1f}%)")

    # Gender vs Class crosstab
    print("\nGender vs Performance Class:")
    gender_class_crosstab = pd.crosstab(
        df['gender'], df['class'], normalize='index') * 100
    print(gender_class_crosstab.round(1))

    # Statistical test for independence
    chi2, p_value, dof, expected = chi2_contingency(
        pd.crosstab(df['gender'], df['class']))
    print(f"\nChi-square test for independence:")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significant association: {'Yes' if p_value < 0.05 else 'No'}")

    return gender_class_crosstab

# %% Feature distributions analysis


def analyze_feature_distributions(df):
    """Analyze distributions of all numerical features"""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('age')  # We'll handle age separately

    print("Numerical Features Summary:")
    for feature in numerical_features:
        data = df[feature]
        print(f"\n{feature}:")
        print(f"  Mean: {data.mean():.2f}")
        print(f"  Median: {data.median():.2f}")
        print(f"  Std: {data.std():.2f}")
        print(f"  Range: {data.min():.2f} - {data.max():.2f}")
        print(f"  Skewness: {stats.skew(data):.2f}")
        print(f"  Kurtosis: {stats.kurtosis(data):.2f}")

    return numerical_features

# %% Performance patterns across classes


def analyze_performance_patterns(df):
    """Analyze how features vary across performance classes"""
    print("\n" + "="*60)
    print("PERFORMANCE PATTERNS ACROSS CLASSES")
    print("="*60)

    numerical_features = ['age', 'height_cm', 'weight_kg', 'body fat_%',
                          'diastolic', 'systolic', 'gripForce',
                          'sit and bend forward_cm', 'sit-ups counts', 'broad jump_cm']

    # Group by class and calculate means
    class_means = df.groupby('class')[numerical_features].mean()

    print("Mean values by performance class:")
    print(class_means.round(2))

    # Calculate performance trends (A=best, D=worst)
    print("\nPerformance Trends (A→B→C→D):")
    class_order = ['A', 'B', 'C', 'D']

    for feature in numerical_features:
        values = [class_means.loc[cls, feature] for cls in class_order]
        trend = "↑" if values[0] > values[-1] else "↓"
        range_val = max(values) - min(values)
        print(f"{feature:25}: {trend} Range: {range_val:.2f}")

    return class_means

# %% Correlation analysis


def analyze_correlations(df):
    """Analyze correlations between features"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)

    # Select numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation_matrix = df[numerical_features].corr()

    print("Strong correlations (|r| > 0.5):")
    strong_corr_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                strong_corr_pairs.append((feature1, feature2, corr_val))
                print(f"{feature1:25} ↔ {feature2:25}: r = {corr_val:.3f}")

    return correlation_matrix, strong_corr_pairs

# %% Age analysis


def analyze_age_patterns(df):
    """Analyze age-related patterns"""
    print("\n" + "="*60)
    print("AGE PATTERN ANALYSIS")
    print("="*60)

    # Age statistics by class
    age_by_class = df.groupby('class')['age'].agg(
        ['mean', 'median', 'std']).round(2)
    print("Age statistics by performance class:")
    print(age_by_class)

    # Age groups analysis
    df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 65],
                             labels=['21-30', '31-40', '41-50', '51-64'])

    age_group_class = pd.crosstab(
        df['age_group'], df['class'], normalize='index') * 100
    print("\nAge group vs Performance class (% within age group):")
    print(age_group_class.round(1))

    return age_by_class, age_group_class

# %% BMI analysis


def analyze_bmi_patterns(df):
    """Calculate and analyze BMI patterns"""
    print("\n" + "="*60)
    print("BMI ANALYSIS")
    print("="*60)

    # Calculate BMI
    df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2

    # BMI categories
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'

    df['bmi_category'] = df['bmi'].apply(categorize_bmi)

    # BMI by performance class
    bmi_by_class = df.groupby('class')['bmi'].agg(
        ['mean', 'median', 'std']).round(2)
    print("BMI statistics by performance class:")
    print(bmi_by_class)

    # BMI category distribution
    bmi_cat_dist = df['bmi_category'].value_counts()
    print(f"\nBMI Category Distribution:")
    for category, count in bmi_cat_dist.items():
        percentage = (count / len(df)) * 100
        print(f"{category:12}: {count:,} ({percentage:.1f}%)")

    return bmi_by_class

# %% Statistical tests


def perform_statistical_tests(df):
    """Perform statistical tests to identify significant differences"""
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)

    numerical_features = ['age', 'height_cm', 'weight_kg', 'body fat_%',
                          'gripForce', 'sit and bend forward_cm',
                          'sit-ups counts', 'broad jump_cm']

    print("ANOVA F-tests for differences across performance classes:")
    print("(Testing if means differ significantly between classes A, B, C, D)")

    for feature in numerical_features:
        groups = [df[df['class'] == cls]
                  [feature].values for cls in ['A', 'B', 'C', 'D']]
        f_stat, p_value = stats.f_oneway(*groups)
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"{feature:25}: F = {f_stat:7.2f}, p = {p_value:.6f} {significance}")

    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")

# %% Main execution function


def main():
    """Main function to run complete EDA"""
    # Load and examine data
    df = load_and_examine_data()

    # Data quality assessment
    zero_analysis, extreme_outliers = assess_data_quality(df)

    # Target variable analysis
    class_counts, class_percentages = analyze_target_variable(df)

    # Gender analysis
    gender_class_crosstab = analyze_gender_distribution(df)

    # Feature distributions
    numerical_features = analyze_feature_distributions(df)

    # Performance patterns
    class_means = analyze_performance_patterns(df)

    # Correlation analysis
    correlation_matrix, strong_corr_pairs = analyze_correlations(df)

    # Age patterns
    age_by_class, age_group_class = analyze_age_patterns(df)

    # BMI analysis
    bmi_by_class = analyze_bmi_patterns(df)

    # Statistical tests
    perform_statistical_tests(df)

    print("\n" + "="*60)
    print("EDA COMPLETE - Ready for Visualization and Modeling")
    print("="*60)

    return df


# %% Visualizations (optional, can be added later)
# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Create custom color palette for performance classes
performance_colors = {
    'A': '#2ca02c',  # Green
    'B': '#1f77b4',  # Blue
    'C': '#ff7f0e',  # Orange
    'D': '#d62728'   # Red
}

# %% Target variable visualization


def visualize_target_distribution(df):
    """Visualize the target variable distribution"""
    plt.figure(figsize=(12, 6))

    # Plot 1: Bar chart of class distribution
    plt.subplot(1, 2, 1)
    class_counts = df['class'].value_counts().sort_index()
    bars = plt.bar(class_counts.index, class_counts.values,
                   color=[performance_colors[c] for c in class_counts.index])

    plt.title('Performance Class Distribution', fontsize=14)
    plt.xlabel('Performance Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height+50,
                 f'{height:,}', ha='center', va='bottom', fontsize=11)

    # Plot 2: Pie chart of class distribution
    plt.subplot(1, 2, 2)
    class_percentages = df['class'].value_counts(
        normalize=True).sort_index() * 100
    plt.pie(class_percentages, labels=class_percentages.index, autopct='%1.1f%%',
            colors=[performance_colors[c] for c in class_percentages.index],
            startangle=90, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    plt.title('Performance Class Percentage', fontsize=14)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# %% Feature distributions visualization


def visualize_feature_distributions(df):
    """Visualize distributions of key numerical features"""
    # Select key features
    features_to_plot = ['age', 'height_cm', 'weight_kg', 'body fat_%',
                        'gripForce', 'sit and bend forward_cm', 'sit-ups counts', 'broad jump_cm']

    # Create figure with multiple histograms
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle('Feature Distributions', fontsize=16, y=0.98)
    axes = axes.flatten()

    for i, feature in enumerate(features_to_plot):
        sns.histplot(df[feature], kde=True, ax=axes[i],
                     color='#1f77b4', alpha=0.7)

        # Calculate mean and median
        mean_val = df[feature].mean()
        median_val = df[feature].median()

        # Add mean and median lines
        axes[i].axvline(mean_val, color='r', linestyle='-', linewidth=1.5,
                        label=f'Mean: {mean_val:.1f}')
        axes[i].axvline(median_val, color='g', linestyle='--', linewidth=1.5,
                        label=f'Median: {median_val:.1f}')

        axes[i].set_title(f'Distribution of {feature}', fontsize=12)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Now create boxplots by class
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle('Feature Distributions by Performance Class',
                 fontsize=16, y=0.98)
    axes = axes.flatten()

    for i, feature in enumerate(features_to_plot):
        # Create boxplot with custom colors
        boxprops = {'edgecolor': 'black', 'linewidth': 1.5}
        medianprops = {'color': 'black', 'linewidth': 2}
        whiskerprops = {'color': 'black', 'linewidth': 1.5}
        capprops = {'color': 'black', 'linewidth': 1.5}

        sns.boxplot(x='class', y=feature, data=df, ax=axes[i],
                    palette=performance_colors,
                    boxprops=boxprops, medianprops=medianprops,
                    whiskerprops=whiskerprops, capprops=capprops)

        axes[i].set_title(f'{feature} by Class', fontsize=12)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('feature_boxplots_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()

# %% Gender analysis visualization


def visualize_gender_analysis(df):
    """Visualize gender distribution and its relationship with performance"""
    plt.figure(figsize=(18, 6))

    # Plot 1: Gender distribution
    plt.subplot(1, 3, 1)
    gender_counts = df['gender'].value_counts()
    plt.pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%',
            colors=['#1f77b4', '#e377c2'], startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    plt.title('Gender Distribution', fontsize=14)

    # Plot 2: Performance class by gender
    plt.subplot(1, 3, 2)
    gender_class = pd.crosstab(df['gender'], df['class'])
    gender_class_percentage = pd.crosstab(
        df['gender'], df['class'], normalize='index') * 100

    ax = gender_class_percentage.plot(kind='bar', color=[
                                      performance_colors[c] for c in gender_class_percentage.columns])
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xticklabels(['Male', 'Female'], rotation=0)
    ax.set_title('Performance Class by Gender', fontsize=14)
    plt.legend(title='Class', title_fontsize=12)

    # Plot 3: Key metrics by gender
    plt.subplot(1, 3, 3)
    metrics = ['gripForce', 'sit-ups counts', 'broad jump_cm']
    metric_means = df.groupby('gender')[metrics].mean()

    ax = metric_means.T.plot(kind='bar', color=['#1f77b4', '#e377c2'])
    ax.set_title('Key Performance Metrics by Gender', fontsize=14)
    ax.set_ylabel('Mean Value', fontsize=12)
    plt.xticks(rotation=25, ha='right')

    # Add percentage differences
    for i, metric in enumerate(metrics):
        male_val = metric_means.loc['M', metric]
        female_val = metric_means.loc['F', metric]
        percentage_diff = ((male_val - female_val) / female_val) * 100
        plt.text(i-0.1, max(male_val, female_val) + 5,
                 f'+{percentage_diff:.0f}%', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('gender_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additional plot: Performance metrics by gender and class
    metrics = ['gripForce', 'sit-ups counts',
               'broad jump_cm', 'sit and bend forward_cm']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Metrics by Gender and Class',
                 fontsize=16, y=0.98)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.barplot(x='class', y=metric, hue='gender', data=df,
                    palette=['#1f77b4', '#e377c2'], ax=axes[i])
        axes[i].set_title(f'{metric} by Class and Gender', fontsize=14)
        axes[i].set_xlabel('Performance Class', fontsize=12)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].legend(title='Gender', labels=['Male', 'Female'])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('gender_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

# %% Correlation analysis visualization


def visualize_correlations(df):
    """Visualize correlations between features"""
    # Select numerical features
    numerical_features = ['age', 'height_cm', 'weight_kg', 'body fat_%',
                          'diastolic', 'systolic', 'gripForce',
                          'sit and bend forward_cm', 'sit-ups counts', 'broad jump_cm']

    # Compute correlation matrix
    correlation_matrix = df[numerical_features].corr()

    # Create a custom diverging colormap
    cmap = LinearSegmentedColormap.from_list('custom_div',
                                             [(0, '#d62728'), (0.5, '#ffffff'), (1, '#2ca02c')])

    # Visualize correlation matrix
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap=cmap,
                vmin=-1, vmax=1, center=0, square=True, mask=mask,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'})

    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.xticks(fontsize=11, rotation=45, ha='right')
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create scatter plots for strongly correlated features
    strong_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                strong_corr_pairs.append((feature1, feature2, corr_val))

    # Create scatter plots for strongly correlated pairs
    if len(strong_corr_pairs) > 0:
        rows = (len(strong_corr_pairs) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(16, 4*rows))
        fig.suptitle(
            'Scatter Plots for Strongly Correlated Features', fontsize=16, y=0.99)

        if rows > 1:
            axes = axes.flatten()

        for i, (feature1, feature2, corr_val) in enumerate(strong_corr_pairs):
            ax = axes[i] if rows > 1 else axes[i % 2]
            scatter = sns.scatterplot(x=feature1, y=feature2, data=df,
                                      hue='class', palette=performance_colors,
                                      alpha=0.6, s=50, ax=ax)

            # Add regression line
            sns.regplot(x=feature1, y=feature2, data=df,
                        scatter=False, ax=ax, color='black', line_kws={'linewidth': 2})

            ax.set_title(
                f'{feature1} vs {feature2} (r = {corr_val:.2f})', fontsize=13)
            ax.grid(True, alpha=0.3)

            # Add legend with better placement
            if i == 0:
                scatter.legend(title='Class', bbox_to_anchor=(
                    1.05, 1), loc='upper left')
            else:
                ax.get_legend().remove()

        # Hide any unused subplots
        if len(strong_corr_pairs) < len(axes):
            for j in range(len(strong_corr_pairs), len(axes)):
                axes[j].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig('strong_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

# %% Performance metrics visualization


def visualize_performance_metrics(df):
    """Visualize performance metrics across classes"""
    metrics = ['gripForce', 'sit-ups counts',
               'broad jump_cm', 'sit and bend forward_cm']

    # Create figure with line plots showing performance trends
    plt.figure(figsize=(14, 10))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)

        # Calculate means for each class
        means = df.groupby('class')[metric].mean().reindex(
            ['A', 'B', 'C', 'D'])

        # Plot line with markers
        plt.plot(['A', 'B', 'C', 'D'], means.values, marker='o', markersize=10,
                 linewidth=2, color='#1f77b4')

        # Add data points
        for j, (cls, val) in enumerate(means.items()):
            plt.scatter(
                j, val, s=150, color=performance_colors[cls], edgecolor='black', zorder=5)
            plt.text(j, val+means.std()*0.1,
                     f'{val:.1f}', ha='center', fontsize=12)

        plt.title(f'{metric} by Performance Class', fontsize=14)
        plt.xlabel('Performance Class', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(ticks=range(4), labels=['A', 'B', 'C', 'D'], fontsize=12)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('performance_metrics_by_class.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Now create radar plot for all classes
    metrics = ['gripForce', 'sit-ups counts', 'broad jump_cm',
               'sit and bend forward_cm', 'body fat_%', 'age']

    # Normalize the data for radar plot
    def normalize(series):
        if series.name in ['body fat_%', 'age']:  # Lower is better
            return (series.max() - series) / (series.max() - series.min())
        else:  # Higher is better
            return (series - series.min()) / (series.max() - series.min())

    # Calculate normalized means for each class
    radar_data = {}
    for metric in metrics:
        class_means = df.groupby('class')[metric].mean()
        radar_data[metric] = normalize(class_means)

    # Convert to DataFrame
    radar_df = pd.DataFrame(radar_data).T

    # Create radar plot
    plt.figure(figsize=(10, 10))

    # Number of variables
    categories = list(radar_df.index)
    N = len(categories)

    # Create angle for each variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the plot

    # Create the plot
    ax = plt.subplot(111, polar=True)

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)

    # Draw y-axis labels (radial labels)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5",
               "0.75", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)

    # Plot each class
    for cls in ['A', 'B', 'C', 'D']:
        values = radar_df[cls].values.tolist()
        values += values[:1]  # Close the plot

        ax.plot(angles, values, linewidth=2,
                label=f'Class {cls}', color=performance_colors[cls])
        ax.fill(angles, values, alpha=0.25, color=performance_colors[cls])

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Performance Metrics Radar Plot by Class', fontsize=15)

    plt.tight_layout()
    plt.savefig('performance_radar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# %% Age analysis visualization


def visualize_age_analysis(df):
    """Visualize age-related patterns"""
    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 65],
                             labels=['21-30', '31-40', '41-50', '51-64'])

    # Plot 1: Age distribution
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.histplot(df['age'], kde=True, bins=20, color='#1f77b4')
    plt.title('Age Distribution', fontsize=14)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Plot 2: Age group distribution
    plt.subplot(1, 3, 2)
    age_group_counts = df['age_group'].value_counts().sort_index()
    sns.barplot(x=age_group_counts.index,
                y=age_group_counts.values, palette='viridis')
    plt.title('Age Group Distribution', fontsize=14)
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Add count labels
    for i, count in enumerate(age_group_counts):
        plt.text(i, count+50, f'{count:,}', ha='center', fontsize=11)

    # Plot 3: Age vs class
    plt.subplot(1, 3, 3)
    sns.boxplot(x='class', y='age', data=df, palette=performance_colors)
    plt.title('Age Distribution by Performance Class', fontsize=14)
    plt.xlabel('Performance Class', fontsize=12)
    plt.ylabel('Age', fontsize=12)

    plt.tight_layout()
    plt.savefig('age_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additional plot: Performance class distribution by age group
    plt.figure(figsize=(14, 8))

    age_class_crosstab = pd.crosstab(
        df['age_group'], df['class'], normalize='index') * 100

    # Plot as stacked bar chart
    ax = age_class_crosstab.plot(kind='bar', stacked=True,
                                 color=[performance_colors[c]
                                        for c in age_class_crosstab.columns],
                                 figsize=(14, 8))

    plt.title('Performance Class Distribution by Age Group', fontsize=16)
    plt.xlabel('Age Group', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.legend(title='Class', fontsize=12, title_fontsize=14)

    # Add percentage labels on bars
    for c in age_class_crosstab.columns:
        for i, p in enumerate(age_class_crosstab[c]):
            if p > 5:  # Only show percentage if it's > 5%
                plt.text(i, age_class_crosstab.iloc[i][:c].sum() + p/2,
                         f'{p:.1f}%', ha='center', fontsize=11)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('age_group_class_distribution.png',
                dpi=300, bbox_inches='tight')
    plt.close()

# %% Body composition visualization


def visualize_body_composition(df):
    """Visualize body composition metrics and their relationship with performance"""
    # Calculate BMI
    df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2

    # BMI categories
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'

    df['bmi_category'] = df['bmi'].apply(categorize_bmi)

    # Create figure with 4 plots
    plt.figure(figsize=(18, 12))

    # Plot 1: BMI distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['bmi'], kde=True, bins=30, color='#1f77b4')
    plt.axvline(x=18.5, color='green', linestyle='--', linewidth=1)
    plt.axvline(x=25, color='orange', linestyle='--', linewidth=1)
    plt.axvline(x=30, color='red', linestyle='--', linewidth=1)

    plt.text(16.5, plt.ylim()[1]*0.9, 'Underweight', rotation=90, va='top')
    plt.text(21.5, plt.ylim()[1]*0.9, 'Normal', rotation=90, va='top')
    plt.text(27, plt.ylim()[1]*0.9, 'Overweight', rotation=90, va='top')
    plt.text(33, plt.ylim()[1]*0.9, 'Obese', rotation=90, va='top')

    plt.title('BMI Distribution', fontsize=14)
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Plot 2: Body fat percentage by class
    plt.subplot(2, 2, 2)
    sns.boxplot(x='class', y='body fat_%', data=df, palette=performance_colors)
    plt.title('Body Fat Percentage by Performance Class', fontsize=14)
    plt.xlabel('Performance Class', fontsize=12)
    plt.ylabel('Body Fat Percentage (%)', fontsize=12)

    # Calculate body fat means for annotation
    bf_means = df.groupby('class')['body fat_%'].mean()
    for i, cls in enumerate(['A', 'B', 'C', 'D']):
        plt.text(i, bf_means[cls]+1,
                 f'{bf_means[cls]:.1f}%', ha='center', fontsize=11)

    # Plot 3: BMI category distribution
    plt.subplot(2, 2, 3)
    bmi_cat_counts = df['bmi_category'].value_counts().sort_index()
    colors = ['#9467bd', '#2ca02c', '#ff7f0e',
              '#d62728']  # Purple, Green, Orange, Red

    plt.pie(bmi_cat_counts, labels=bmi_cat_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    plt.title('BMI Category Distribution', fontsize=14)
    plt.axis('equal')

    # Plot 4: BMI category by performance class
    plt.subplot(2, 2, 4)
    bmi_class = pd.crosstab(
        df['bmi_category'], df['class'], normalize='index') * 100

    # Reorder categories
    cat_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    bmi_class = bmi_class.reindex(cat_order)

    bmi_class.plot(kind='bar', stacked=False,
                   color=[performance_colors[c] for c in bmi_class.columns])
    plt.title('Performance Class Distribution by BMI Category', fontsize=14)
    plt.xlabel('BMI Category', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Class')

    plt.tight_layout()
    plt.savefig('body_composition.png', dpi=300, bbox_inches='tight')
    plt.close()

# %% Feature importance visualization


def visualize_feature_importance(df):
    """Visualize feature importance based on statistical significance"""
    # Calculate statistical significance of features by comparing classes A and D
    features = ['age', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic', 'systolic',
                'gripForce', 'sit and bend forward_cm', 'sit-ups counts', 'broad jump_cm']

    # Calculate Cohen's d effect size
    effect_sizes = {}
    for feature in features:
        a_values = df[df['class'] == 'A'][feature]
        d_values = df[df['class'] == 'D'][feature]

        # Cohen's d = (Mean1 - Mean2) / Pooled Standard Deviation
        mean_diff = a_values.mean() - d_values.mean()
        pooled_std = np.sqrt(((a_values.std()**2) + (d_values.std()**2)) / 2)
        effect_size = abs(mean_diff / pooled_std)
        effect_sizes[feature] = effect_size

    # Sort by effect size
    effect_sizes = {k: v for k, v in sorted(
        effect_sizes.items(), key=lambda item: item[1], reverse=True)}

    # Create bar chart
    plt.figure(figsize=(14, 8))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(effect_sizes)))
    bars = plt.bar(effect_sizes.keys(), effect_sizes.values(), color=colors)

    plt.title('Feature Importance (Effect Size between Class A and D)', fontsize=16)
    plt.xlabel('Feature', fontsize=14)
    plt.ylabel("Cohen's d Effect Size", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add effect size values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height+0.05,
                 f'{height:.2f}', ha='center', fontsize=11)

    # Add effect size interpretation guide
    plt.text(0.98, 0.95, 'Effect Size Interpretation:\n'
             'Small: d = 0.2\n'
             'Medium: d = 0.5\n'
             'Large: d = 0.8\n'
             'Very Large: d > 1.2',
             transform=plt.gca().transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

# %% 2D Feature space visualization


def visualize_2d_feature_space(df):
    """Visualize the feature space in 2D using the most important features"""
    # Create combinations of the most important features
    important_features = ['sit-ups counts',
                          'broad jump_cm', 'gripForce', 'body fat_%']
    feature_pairs = [
        ('sit-ups counts', 'broad jump_cm'),
        ('gripForce', 'body fat_%'),
        ('sit-ups counts', 'body fat_%'),
        ('broad jump_cm', 'gripForce')
    ]

    # Create scatter plots for each feature pair
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        '2D Feature Space Visualization by Performance Class', fontsize=16, y=0.98)
    axes = axes.flatten()

    for i, (feature1, feature2) in enumerate(feature_pairs):
        scatter = sns.scatterplot(x=feature1, y=feature2, hue='class', data=df,
                                  palette=performance_colors, s=70, alpha=0.7, ax=axes[i])

        axes[i].set_title(f'{feature1} vs {feature2}', fontsize=14)
        axes[i].set_xlabel(feature1, fontsize=12)
        axes[i].set_ylabel(feature2, fontsize=12)
        axes[i].grid(True, alpha=0.3)

        # Only show legend in first plot
        if i > 0:
            axes[i].get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('2d_feature_space.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create additional visualization with contours for decision boundaries
    plt.figure(figsize=(12, 10))

    # Focus on sit-ups and broad jump
    x = df['sit-ups counts']
    y = df['broad jump_cm']

    # Create scatter plot
    plt.scatter(x, y, c=[performance_colors[c] for c in df['class']],
                alpha=0.7, s=70, edgecolor='none')

    # Add contour plots for each class
    for cls in ['A', 'B', 'C', 'D']:
        class_data = df[df['class'] == cls]

        if len(class_data) > 10:  # Need enough points for density estimation
            # Create a 2D density
            x_class = class_data['sit-ups counts']
            y_class = class_data['broad jump_cm']

            # Calculate kernel density estimate
            try:
                xmin, xmax = x.min(), x.max()
                ymin, ymax = y.min(), y.max()

                # Create meshgrid
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x_class, y_class])
                kernel = stats.gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)

                # Plot contours
                plt.contour(xx, yy, f, levels=5,
                            colors=performance_colors[cls], alpha=0.7)

                # Add class label at the center of each cluster
                plt.text(x_class.mean(), y_class.mean(), cls,
                         color='white', fontsize=20, fontweight='bold',
                         ha='center', va='center',
                         bbox=dict(boxstyle='circle', facecolor=performance_colors[cls], alpha=0.7))
            except:
                # Skip if kernel density fails
                pass

    plt.title('Performance Class Clusters and Decision Boundaries', fontsize=16)
    plt.xlabel('Sit-ups Count', fontsize=14)
    plt.ylabel('Broad Jump (cm)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=performance_colors[c], label=f'Class {c}')
                       for c in ['A', 'B', 'C', 'D']]
    plt.legend(handles=legend_elements, fontsize=12)

    plt.tight_layout()
    plt.savefig('decision_boundaries.png', dpi=300, bbox_inches='tight')
    plt.close()

# %% Main visualization function


def visualize_all(df):
    """Execute all visualization functions"""
    print("="*60)
    print("GENERATING VISUALIZATIONS FOR BODY PERFORMANCE DATASET")
    print("="*60)

    print("1. Visualizing target distribution...")
    visualize_target_distribution(df)

    print("2. Visualizing feature distributions...")
    visualize_feature_distributions(df)

    print("3. Visualizing gender analysis...")
    visualize_gender_analysis(df)

    print("4. Visualizing correlations...")
    visualize_correlations(df)

    print("5. Visualizing performance metrics...")
    visualize_performance_metrics(df)

    print("6. Visualizing age analysis...")
    visualize_age_analysis(df)

    print("7. Visualizing body composition...")
    visualize_body_composition(df)

    print("8. Visualizing feature importance...")
    visualize_feature_importance(df)

    print("9. Visualizing 2D feature space...")
    visualize_2d_feature_space(df)

    print("\nAll visualizations complete! Files saved in the current directory.")
    print("="*60)


# %% Execute if run as main script
if __name__ == "__main__":
    df = main()

    visualize_all(df)
