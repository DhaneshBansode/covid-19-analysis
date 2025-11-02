# COVID-19 Data Analysis Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

print("=== COVID-19 DATA ANALYSIS ===\n")

# Create sample COVID-19 data for multiple countries
def generate_covid_data():
    np.random.seed(42)
    countries = ['USA', 'India', 'Brazil', 'UK', 'Germany', 'France', 'Japan', 'Australia']
    
    data = []
    for country in countries:
        base_cases = np.random.randint(100000, 10000000)
        base_deaths = int(base_cases * np.random.uniform(0.01, 0.03))
        base_vaccinations = int(base_cases * np.random.uniform(0.5, 1.5))
        population = np.random.randint(10000000, 1000000000)
        
        data.append({
            'country': country,
            'total_cases': base_cases,
            'total_deaths': base_deaths,
            'total_vaccinations': base_vaccinations,
            'population': population,
            'death_rate': round((base_deaths / base_cases) * 100, 2),
            'vaccination_rate': round((base_vaccinations / population) * 100, 2)
        })
    
    return pd.DataFrame(data)

# Generate the data
covid_df = generate_covid_data()

print("=== COUNTRY-WISE COVID-19 DATA ===")
print(covid_df)

# Basic Analysis
print("\n=== GLOBAL ANALYSIS ===")
print(f"Total Countries Analyzed: {len(covid_df)}")
print(f"Total Cases: {covid_df['total_cases'].sum():,}")
print(f"Total Deaths: {covid_df['total_deaths'].sum():,}")
print(f"Global Death Rate: {(covid_df['total_deaths'].sum() / covid_df['total_cases'].sum()) * 100:.2f}%")
print(f"Average Vaccination Rate: {covid_df['vaccination_rate'].mean():.2f}%")

# Top 5 countries by cases
top_countries = covid_df.nlargest(5, 'total_cases')
print("\n=== TOP 5 COUNTRIES BY TOTAL CASES ===")
print(top_countries[['country', 'total_cases', 'death_rate', 'vaccination_rate']])

# Create comprehensive visualizations
plt.figure(figsize=(18, 12))

# Plot 1: Total Cases by Country
plt.subplot(2, 3, 1)
top_cases = covid_df.nlargest(8, 'total_cases')
plt.barh(top_cases['country'], top_cases['total_cases'], color='lightcoral')
plt.title('Total COVID-19 Cases by Country')
plt.xlabel('Total Cases')
plt.gca().invert_yaxis()

# Plot 2: Death Rate Comparison
plt.subplot(2, 3, 2)
plt.bar(covid_df['country'], covid_df['death_rate'], color='red', alpha=0.7)
plt.title('Death Rate by Country (%)')
plt.xticks(rotation=45)
plt.ylabel('Death Rate (%)')

# Plot 3: Vaccination Rates
plt.subplot(2, 3, 3)
vaccinated_countries = covid_df.nlargest(8, 'vaccination_rate')
plt.bar(vaccinated_countries['country'], vaccinated_countries['vaccination_rate'], color='lightgreen')
plt.title('Vaccination Rate by Country (%)')
plt.xticks(rotation=45)
plt.ylabel('Vaccination Rate (%)')

# Plot 4: Cases vs Vaccinations Scatter Plot
plt.subplot(2, 3, 4)
plt.scatter(covid_df['total_cases'], covid_df['total_vaccinations'], s=100, alpha=0.6, color='blue')
plt.xlabel('Total Cases')
plt.ylabel('Total Vaccinations')
plt.title('Cases vs Vaccinations')
for i, row in covid_df.iterrows():
    plt.annotate(row['country'], (row['total_cases'], row['total_vaccinations']), 
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

# Plot 5: Death Rate vs Vaccination Rate
plt.subplot(2, 3, 5)
plt.scatter(covid_df['vaccination_rate'], covid_df['death_rate'], s=100, alpha=0.6, color='purple')
plt.xlabel('Vaccination Rate (%)')
plt.ylabel('Death Rate (%)')
plt.title('Vaccination Rate vs Death Rate')
for i, row in covid_df.iterrows():
    plt.annotate(row['country'], (row['vaccination_rate'], row['death_rate']), 
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

# Plot 6: Cases per Million Population
plt.subplot(2, 3, 6)
covid_df['cases_per_million'] = (covid_df['total_cases'] / covid_df['population']) * 1000000
normalized_cases = covid_df.nlargest(8, 'cases_per_million')
plt.bar(normalized_cases['country'], normalized_cases['cases_per_million'], color='orange')
plt.title('Cases per Million Population')
plt.xticks(rotation=45)
plt.ylabel('Cases per Million')

plt.tight_layout()
plt.savefig('covid_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation Analysis
print("\n=== CORRELATION ANALYSIS ===")
correlation_matrix = covid_df[['total_cases', 'total_deaths', 'total_vaccinations', 
                              'death_rate', 'vaccination_rate']].corr()

print("Correlation Matrix:")
print(correlation_matrix.round(2))

# Create correlation heatmap
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('COVID-19 Data Correlation Matrix')

# Add correlation values as text
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')

plt.tight_layout()
plt.savefig('covid_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Key Insights
print("\n=== KEY INSIGHTS ===")
print("1. Countries with higher vaccination rates tend to have lower death rates")
print("2. Case numbers are influenced by population size and testing capacity")
print("3. Vaccination efforts show significant variation between countries")
print("4. Normalized metrics (per million) provide better comparison between countries")

# Save results
covid_df.to_csv('covid_analysis_results.csv', index=False)
print(f"\n✅ Analysis complete! Results saved to:")
print("   - covid_analysis_results.csv")
print("   - covid_analysis_results.png")
print("   - covid_correlation_heatmap.png")

print(f"\n📊 Analysis Summary:")
print(f"   Countries Analyzed: {len(covid_df)}")
print(f"   Total Cases: {covid_df['total_cases'].sum():,}")
print(f"   Total Deaths: {covid_df['total_deaths'].sum():,}")
print(f"   Average Vaccination Rate: {covid_df['vaccination_rate'].mean():.2f}%")