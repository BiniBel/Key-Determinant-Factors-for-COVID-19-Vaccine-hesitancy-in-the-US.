#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
import sweetviz as sv


sns.set_theme(style="whitegrid")
scatterplot_marker_size = 15


# Approach
# 
# We want to clean and combine all of our various datasets and export it as one CSV. This will primarily involve dropping columns that we aren't interested in and merging onto a central Pandas DataFrame. After some exploratory analysis, we will determine what rows will beed to be dropped or interpolated.

# County Info
# 
# The vaccine hesitancy dataset has values for multiple segments (ethnicity, social vulnerability, vaccine hesitancy), which we will split out into separate variables and look at each. Our primary index of county codes is given by Federal Information Processing Standards (FIPS), which we extract with useful identifiers of the county (its name and state).

# In[11]:


vaccine_hesitancy_data = pd.read_csv('Data/Vaccine_Hesitancy_for_COVID-19__County_and_local_estimates.csv')


# In[12]:


vaccine_hesitancy_data = pd.read_csv('Data/Vaccine_Hesitancy_for_COVID-19__County_and_local_estimates.csv').rename(columns = {'FIPS Code':'fips'})
county = vaccine_hesitancy_data[['fips', 'County Name', 'State']].rename(columns={'County Name': 'county_name', 'State': 'state'})
county['state'] = county['state'].str.title()
county


# In[13]:


vaccine_hesitancy_data.head()


# In[14]:


regions = [(['New Jersey', 'New York', 'Pennsylvania'], 'New England', 'Northeast'), (['New Jersey', 'New York', 'Pennsylvania'], 'Midatlantic', 'Northeast'), (['Indiana', 'Illinois', 'Michigan', 'Ohio', 'Wisconsin'], 'East North Central', 'Midwest'), (['Iowa', 'Nebraska', 'Kansas', 'North Dakota', 'Minnesota', 'South Dakota', 'Missouri'], 'West North Central', 'Midwest'), (['Delaware', 'District Of Columbia', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia', 'West Virginia'], 'South Atlantic Division', 'South'), (['Alabama', 'Kentucky', 'Mississippi', 'Tennessee'], 'East South Central', 'South'), (['Arkansas', 'Louisiana', 'Oklahoma', 'Texas'], 'West South Central', 'South'), (['Arizona', 'Colorado', 'Idaho', 'New Mexico', 'Montana', 'Utah', 'Nevada', 'Wyoming'], 'Mountain', 'West'), (['Alaska', 'California', 'Hawaii', 'Oregon', 'Washington'], 'Pacific', 'West')]

for (states, division, region) in regions:
    county.loc[county['state'].isin(states), ['division', 'region']] = [division, region]
county


# In[15]:


regions = [(['New Jersey', 'New York', 'Pennsylvania'], 'New England', 'Northeast'), (['New Jersey', 'New York', 'Pennsylvania'], 'Midatlantic', 'Northeast'), (['Indiana', 'Illinois', 'Michigan', 'Ohio', 'Wisconsin'], 'East North Central', 'Midwest'), (['Iowa', 'Nebraska', 'Kansas', 'North Dakota', 'Minnesota', 'South Dakota', 'Missouri'], 'West North Central', 'Midwest'), (['Delaware', 'District Of Columbia', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia', 'West Virginia'], 'South Atlantic Division', 'South'), (['Alabama', 'Kentucky', 'Mississippi', 'Tennessee'], 'East South Central', 'South'), (['Arkansas', 'Louisiana', 'Oklahoma', 'Texas'], 'West South Central', 'South'), (['Arizona', 'Colorado', 'Idaho', 'New Mexico', 'Montana', 'Utah', 'Nevada', 'Wyoming'], 'Mountain', 'West'), (['Alaska', 'California', 'Hawaii', 'Oregon', 'Washington'], 'Pacific', 'West')]

for (states, division, region) in regions:
    county.loc[county['state'].isin(states), ['division', 'region']] = [division, region]
county


# Ethnicity
# 
# Percentage of ethnicity for each county are given. For readability and simplicity, we rename them with the most abundant ethnic group as primary and assume non-Hispanic for all of the non-Hispanic groups.

# In[16]:


ethnicity = vaccine_hesitancy_data[['fips', 'Percent Hispanic', 'Percent non-Hispanic American Indian/Alaska Native', 'Percent non-Hispanic Asian', 'Percent non-Hispanic Black', 'Percent non-Hispanic Native Hawaiian/Pacific Islander', 'Percent non-Hispanic White']].rename(columns = {'Percent Hispanic': 'ethnicity_hispanic', 'Percent non-Hispanic American Indian/Alaska Native': 'ethnicity_native', 'Percent non-Hispanic Asian': 'ethnicity_asian', 'Percent non-Hispanic Black': 'ethnicity_black', 'Percent non-Hispanic Native Hawaiian/Pacific Islander': 'ethnicity_hawaiian', 'Percent non-Hispanic White': 'ethnicity_white'})
ethnicity


# Social Vulnerability

# In[17]:


social_vulnerability_index = vaccine_hesitancy_data[['fips', 'Social Vulnerability Index (SVI)']].rename(columns= {'Social Vulnerability Index (SVI)': 'social_vulnerability_index'})
social_vulnerability_index


# Vaccine Hesitancy

# In[18]:


vaccine_hesitancy = vaccine_hesitancy_data[['fips', 'Estimated hesitant', 'Estimated strongly hesitant']].rename(columns = {'Estimated hesitant': 'vaccine_hesitant', 'Estimated strongly hesitant': 'vaccine_hesitant_strong'})
vaccine_hesitant_mean, vaccine_hesitant_std = vaccine_hesitancy['vaccine_hesitant'].mean(), vaccine_hesitancy['vaccine_hesitant'].std()
vaccine_hesitancy.loc[vaccine_hesitancy['vaccine_hesitant'] > vaccine_hesitant_mean + vaccine_hesitant_std, ['vaccine_hesitant_category']] = 'High'
vaccine_hesitancy.loc[vaccine_hesitancy['vaccine_hesitant'] < vaccine_hesitant_mean - vaccine_hesitant_std, ['vaccine_hesitant_category']] = 'Low'
vaccine_hesitancy['vaccine_hesitant_category'] = vaccine_hesitancy['vaccine_hesitant_category'].fillna('Medium')
vaccine_hesitancy


# Education

# In[19]:


education = pd.read_csv('Data/Education.csv')
education = education[['FIPS Code', 'Percent of adults with less than a high school diploma, 2015-19', 'Percent of adults with a high school diploma only, 2015-19', "Percent of adults completing some college or associate's degree, 2015-19", "Percent of adults with a bachelor's degree or higher, 2015-19"]]
education = education.rename(columns = {'FIPS Code': 'fips', 'Percent of adults with less than a high school diploma, 2015-19': 'education_high_school_less', 'Percent of adults with a high school diploma only, 2015-19': 'education_high_school_only', "Percent of adults completing some college or associate's degree, 2015-19": 'education_degree_some', "Percent of adults with a bachelor's degree or higher, 2015-19": 'education_bachelors_degree'})
education_cols = ['education_high_school_less', 'education_high_school_only', 'education_degree_some', 'education_bachelors_degree']
education[education_cols] = education[education_cols].div(100)
education


# Finding missing data

# In[20]:


# Show missing county rows in education dataset
county[~county['fips'].isin(education['fips'])]


# Poverty

# In[21]:


poverty = pd.read_csv('Data/PovertyEstimates.csv')
poverty = poverty[['FIPStxt', 'Attribute', 'Value']].pivot(index='FIPStxt', columns='Attribute', values='Value').reset_index()
poverty = poverty[['FIPStxt', 'PCTPOVALL_2019']].rename(columns = {'FIPStxt':'fips', 'PCTPOVALL_2019': 'poverty'})
poverty['poverty'] = poverty['poverty'].div(100)
poverty


# In[22]:


# Show missing county rows in poverty dataset
county[~county['fips'].isin(poverty['fips'])]


# Natality

# In[23]:


natality = pd.read_csv('Data/PopulationEstimates.csv')
natality = natality[['FIPStxt', 'POP_ESTIMATE_2019', 'R_birth_2019']].rename(columns = {'FIPStxt': 'fips', 'POP_ESTIMATE_2019': 'population', 'R_birth_2019': 'birth_rate'})
natality['birth_rate'] = natality['birth_rate'].div(100)
natality


# In[24]:


# Show missing county rows in natality dataset
county[~county['fips'].isin(natality['fips'])]


# Elections

# In[25]:


election_years = [2008, 2012, 2016, 2020]
def election_winner(row, year):
    if row['dem_' + str(year)] > row['gop_' + str(year)]:
        return 'Democrat'
    else:
        return 'Republican'
elections_data = pd.read_csv('Data/US_County_Level_Presidential_Results_08-16.csv').rename(columns={'fips_code': 'fips'})
elections_data_2020 = pd.read_csv('Data/2020_US_County_Level_Presidential_Results.csv').rename(columns={'votes_gop': 'gop_2020', 'votes_dem': 'dem_2020', 'county_fips': 'fips'})
elections_data = elections_data.merge(elections_data_2020[['fips', 'gop_2020', 'dem_2020']], left_on='fips', right_on='fips')
elections = pd.DataFrame()
elections['fips'] = elections_data['fips']
for year in election_years:
    elections['election_' + str(year)] = elections_data.apply(lambda row: election_winner(row, year), axis=1)
elections


# In[26]:


elections['election_democrat_wins'] = sum([elections['election_' + str(year)].str.count('Democrat') for year in election_years])
elections['election_republican_wins'] = sum([elections['election_' + str(year)].str.count('Republican') for year in election_years])
elections = elections[['fips', 'election_democrat_wins', 'election_republican_wins']]
elections


# In[27]:


# Show missing county rows in elections dataset
county[~county['fips'].isin(elections['fips'])]


# Unemployment
# 
# From the Unemployment dataset, we have several useful data points involving geography (rural vs urban continuum code, urban influence code), income (median household, and represented as a percent of median state total) and unemployment rate.

# In[28]:


unemployment = pd.read_csv('Data/Unemployment.csv').pivot(index='fips_txt', columns='Attribute', values='Value').reset_index().rename(columns = {'fips_txt':'fips'})
geography = unemployment[['fips', 'Rural_urban_continuum_code_2013', 'Urban_influence_code_2013']].rename(columns={'Rural_urban_continuum_code_2013': 'rural_urban_code', 'Urban_influence_code_2013': 'urban_influence_code'})
# TODO: convert urban/rural codes into z-scores
geography


# In[29]:


# Show missing county rows in geography dataset
county[~county['fips'].isin(geography['fips'])]


# Income
# 
# To look at the county's economic factors, we keep two columns representing the estimated median household income in 2019 and the county household median income as a percent of the state total median household income. We represent this percent as a decimal.

# In[30]:


income = unemployment[['fips', 'Med_HH_Income_Percent_of_State_Total_2019', 'Median_Household_Income_2019']].rename(columns={'Med_HH_Income_Percent_of_State_Total_2019': 'median_income_percent_state', 'Median_Household_Income_2019': 'median_income'})
income['median_income_percent_state'] = income['median_income_percent_state'].div(100)
income


# In[31]:


# Show missing county rows in income dataset
county[~county['fips'].isin(income['fips'])]


# Unemployment

# In[32]:


unemployment = unemployment[['fips', 'Unemployment_rate_2019']].rename(columns={'Unemployment_rate_2019': 'unemployment'})
unemployment['unemployment'] = unemployment['unemployment'].div(100)
unemployment


# In[33]:


# Show missing county rows in unemployment dataset
county[~county['fips'].isin(unemployment['fips'])]


# Religion

# In[34]:


religion = pd.read_csv('Data/U.S. Religion Census Religious Congregations and Membership Study, 2010 (County File).csv')
religion = religion[['FIPS', 'TOTRATE', 'EVANRATE', 'MPRTRATE']].rename(columns = {'FIPS': 'fips', 'TOTRATE': 'religion_total', 'EVANRATE': 'religion_evangelical', 'MPRTRATE': 'religion_mainline_protestant'})
religion
# TODO: decide to drop black_protestant and/or orthodox columns


# In[35]:


# Show missing county rows in religion dataset
county[~county['fips'].isin(religion['fips'])]


# In[36]:


# Show missing value county rows in religion_total column
county[county['fips'].isin(religion[religion['religion_total'].isnull()]['fips'])]


# In[37]:


# Show missing value county rows in religion_evangelical column
county[county['fips'].isin(religion[religion['religion_evangelical'].isnull()]['fips'])]


# In[38]:


# Show missing value county rows in religion_evangelical column
county[county['fips'].isin(religion[religion['religion_mainline_protestant'].isnull()]['fips'])]


# Data Completeness
# 
# As noted above we are missing a few recurring pattern of certain counties being missing in most of our datasets.
# 
# Oglala Lakota County, South Dakota (FIPS 46102)
# Kalawao County, Hawaii (FIPS 15005)
# Various parts of Alaska, especially in the election dataset
# Upon further investigation into those areas, we learned that Oglala Lakota County does not have a functioning county seat and remains unorganized, which explains the difficulty government surveyers would have with gathering data there. However, this county is entirely on an Indian reservation, which would give us valuable insight on Native American vaccine hesitancy.
# 
# Kalawao County because of its small population does not have many of the functions that a normal county would have.
# 
# While Alaska does administer using county divisions, for elections they use a different geographic boundary of boroughs, which do not conveniently align with counties. This makes a county level political correlation with vaccine hesitancy impossible for us. Similar to the note above, Alaska is home to a lot of Native Americans, 15% of the population, which means our analysis will lose insight into Native American vaccine hesitancy.
# 
# Because of the issues surrounding population size, county organization and governmental issues, we decide to drop those two data points and focus our analysis on the central parts of USA, ignoring Alaska.

# In[39]:


county = county.drop(county[county['fips'] == 46102].index)
county = county.drop(county[county['fips'] == 15005].index)
county = county.drop(county[county['state'] == 'Alaska'].index)
county


# Aggregation
# 
# We set the fips as index for all of our dataframes and then concatenate them along it with an inner join. We note that there's only one row that was lost.

# In[40]:


dfs = [df.set_index('fips') for df in [county, vaccine_hesitancy, social_vulnerability_index, ethnicity, natality, unemployment, geography, income, poverty, education, religion, elections]]
df = pd.concat(dfs, axis=1, join='inner').reset_index()
df


# EXPLORATORY DATA ANALYSIS

# In[44]:


df.to_csv('Cleaned Data/interim_clean_dataset_2021-06-06.csv', index=False)
df.head()


# In[45]:


# TODO: Bar charts (univariate) to explore vaccine hesitancy cluster groups along demographic data
df.info()


# Vaccine Hesitancy
# 
# 
# Vaccine Hesitant

# In[46]:


df['vaccine_hesitant'].describe()


# In[47]:


vaccine_hesitant_histogram = sns.histplot(data=df, x='vaccine_hesitant', kde=True).set_title('Distribution of percent of vaccine hesitant population by county count')


# In[48]:


vaccine_hesitant_boxplot = sns.boxplot(data=df, x='vaccine_hesitant').set_title('Box plot distribution of percent of vaccine hesitant population by county')


# The vaccine hesitant population is distributed normally more or less with a very slight right skew, however that isn't too significant. There are slightly more outliers on the less hesitant side than the very hesitant side.

# In[49]:


df['vaccine_hesitant_strong'].describe()


# In[50]:


vaccine_hesitant_strong_histogram = sns.histplot(data=df, x='vaccine_hesitant_strong', kde=True).set_title('Distribution of percent of strongly vaccine hesitant population by county count')


# In[51]:


vaccine_hesitant_strong_boxplot = sns.boxplot(data=df, x='vaccine_hesitant_strong').set_title('Box plot distribution of percent of strongly vaccine hesitant population by county')


# The strongly vaccine hesitant distribution is also normally distributed, but seemingly less so this time. The distribution has a slight right skew. There are much more outliers on the hesitant side than there are on the less hesitant (the opposite of the last vaccine_hesitant distribution).
# 
# TODO: is this a fat tailed distribution?

# Ethnicity

# In[52]:


df_narrow_ethnicity = df.melt(id_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_vars=['ethnicity_hispanic', 'ethnicity_native', 'ethnicity_asian', 'ethnicity_black', 'ethnicity_hawaiian', 'ethnicity_white'], var_name='ethnicity')
graph = sns.FacetGrid(df_narrow_ethnicity, col='ethnicity')
graph.map_dataframe(sns.scatterplot, x='value', y='vaccine_hesitant', s=scatterplot_marker_size)
graph.fig.subplots_adjust(top=0.8)
graph.fig.suptitle('Percent vaccine hesitant vs Percent ethnicity in county')
_ = graph.set_axis_labels("Ethnicity Percent", "Vaccine Hesitant")
# TODO: Figure out better way to visualize. Group dots into larger points?


# In[53]:


graph = sns.FacetGrid(df_narrow_ethnicity, col='ethnicity')
graph.map_dataframe(sns.scatterplot, x='value', y='vaccine_hesitant_strong', s=scatterplot_marker_size)
graph.fig.subplots_adjust(top=0.8)
graph.fig.suptitle('Percent strongly vaccine hesitant vs Percent ethnicity in county')
_ = graph.set_axis_labels("Ethnicity Percent", "Strongly Vaccine Hesitant")


# Our ethnicity plots don't show a strong relationship between vaccine hesitancy and county ethnic composition. It's difficult to tell in some cases because of the low minority population in many counties, which tend to show a plot clustered around the y-axis.
# 
# The Hispanic plots show an almost 2D bell curve distribution with a normal at 18% vaccine hesitant and 9% strongly vaccine hesitant. Both of these figures are similar to the overall total vaccine hesitant distributions above.
# 
# Counties with higher Native populations seem to be more highly vaccine hesitant. However the vast majority of counties have a very small Native population, which is somewhat normally distributed along vaccine hesitancy.
# 
# Counties with higher Asian populations tend to be less vaccine hesitant and strongly vaccine hesitant.
# 
# Counties with higher Black populations seem to have a higher percent of vaccine hesitant, but not strongly vaccine hesitant populations.
# 
# Counties with a larger Hawaiian composition tend to be evenly distributed along vaccine hesitancy except for a small modal bump at a very low vaccine hesitancy of 10% and strongly vaccine hesitant 5%.
# 
# The strongly vaccine hesitant white show an almost 2D bell curve distribution with a normal at 10%. The vaccine hesitant graph doesn't show a clear relationship

# Geography
# 
# Region

# In[54]:


df_narrow_region = df.melt(id_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_vars=['region'], value_name='Region')
graph = sns.FacetGrid(df_narrow_region, col='Region')
graph.map_dataframe(sns.histplot, bins=20, kde=True)
graph.fig.subplots_adjust(top=0.8)
plt.legend(labels=["Hesitant","Strongly\nHesitant"])
graph.fig.suptitle('Distribution of percent of vaccine hesitant population by county count for each census region')
_ = graph.set_axis_labels("Vaccine hesitant percent", "County count")


# We look at the vaccine hesitant and strongly vaccine hesitant populations by census region. The south and midwest have a very differentiated distribution of the two variables with little overlap. The west and northeast have much more of an overlap. This suggest that opinion on vaccine hesitancy may be more poplarized in the south and midwest.

# Division

# In[56]:


df_narrow_division = df.melt(id_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_vars=['division'], value_name='Division')
graph = sns.FacetGrid(df_narrow_division, col='Division', col_wrap=4)
graph.map_dataframe(sns.histplot, bins=15, kde=True)
graph.fig.subplots_adjust(top=0.8)
plt.legend(labels=["Hesitant","Strongly\nHesitant"])
graph.fig.suptitle('Distribution of percent of vaccine hesitant population by county count for each census division')
_ = graph.set_axis_labels("Vaccine Hesitant Percent", "County Count")


# State

# In[57]:


df_narrow_state = df.melt(id_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_vars=['state'], value_name='State')
graph = sns.FacetGrid(df_narrow_state, col='State', col_wrap=5)
graph.map_dataframe(sns.histplot, kde=True)
graph.set_axis_labels("Vaccine Hesitant Percent", "County Count")
plt.legend(labels=["Hesitant","Strongly\nHesitant"])
graph.fig.subplots_adjust(top=1)
graph.fig.suptitle('Distribution of percent of vaccine hesitant population by county count for each state')
_ = plt.xlim(0, 0.32)

# TODO: check variances of all states, fix title


# Rural Urban Code

# In[58]:


df_narrow_rural_urban = df.melt(id_vars=['rural_urban_code'], value_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_name='vaccine_hesitancy')
rural_urban_code_boxplot = sns.boxplot(data=df_narrow_rural_urban, x='rural_urban_code', y='vaccine_hesitancy', hue='variable').set_title('Box plot distribution of percent of vaccine hesitant population by rural/urban county code')


# We see a slight increase in vaccine hesitant and strongly vaccine hesitant populations with an increase in the rural urban code. This suggests that more rural counties may have higher vaccine hesitant populations.

# Urban Influence Code

# In[61]:


df_narrow_urban_influence = df.melt(id_vars=['urban_influence_code'], value_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_name='vaccine_hesitancy')
urban_influence_code_boxplot = sns.boxplot(data=df_narrow_urban_influence, x='urban_influence_code', y='vaccine_hesitancy', hue='variable').set_title('Box plot distribution of percent of vaccine hesitant population by county urban influence code')


# Population

# In[76]:


df_narrow_population = df.melt(id_vars=['population'], value_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_name='vaccine_hesitancy')
population_scatterplot = sns.scatterplot(data=df_narrow_population, x='population', y='vaccine_hesitancy', hue='variable', s=scatterplot_marker_size).set_title('Percent vaccine hesitant vs county population')


# Birth Rate

# In[64]:


df_narrow_birth_rate = df.melt(id_vars=['birth_rate'], value_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_name='vaccine_hesitancy')
birth_rate_scatterplot = sns.scatterplot(data=df_narrow_birth_rate, x='birth_rate', y='vaccine_hesitancy', hue='variable', s=scatterplot_marker_size).set_title('Percent vaccine hesitant vs county birth rate')
# birth_rate_scatterplot = sns.scatterplot(data=df, x='birth_rate', y='vaccine_hesitant', s=scatterplot_marker_size).set_title('Percent vaccine hesitant vs county birth rate')


# Economic

# In[66]:


# median_income_scatterplot = sns.scatterplot(data=df, x='median_income', y='vaccine_hesitant', s=scatterplot_marker_size).set_title('Percent vaccine hesitant vs county median household income')


# In[67]:


df_narrow_median_income = df.melt(id_vars=['median_income'], value_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_name='vaccine_hesitancy')
median_income_scatterplot = sns.scatterplot(data=df_narrow_median_income, x='median_income', y='vaccine_hesitancy', hue='variable', s=scatterplot_marker_size).set_title('Percent vaccine hesitant vs county median household income')


# We see an inverse relationship between both vaccine hesitant and strongly vaccine hesitant populations and median household income. The relationship seems to be stronger with the strongly vaccine hesitant population.

# In[68]:


# median_income_percent_state_scatterplot = sns.scatterplot(data=df, x='median_income_percent_state', y='vaccine_hesitant', s=scatterplot_marker_size)
# TODO: drop column. superfluous with median_income?


# This scatterplot of median household income as a percent of the state income shows more or less the same thing as the median household income. We will drop this from our dataset.

# In[69]:


df_narrow_poverty = df.melt(id_vars=['poverty'], value_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_name='vaccine_hesitancy')
poverty_scatterplot = sns.scatterplot(data=df_narrow_poverty, x='poverty', y='vaccine_hesitancy', hue='variable', s=scatterplot_marker_size).set_title('Percent vaccine hesitant vs county poverty rate')
# poverty_scatterplot = sns.scatterplot(data=df, x='poverty', y='vaccine_hesitant', s=scatterplot_marker_size).set_title('Percent vaccine hesitant vs county poverty rate')


# Education

# In[70]:


df_narrow_education = pd.melt(df, id_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_vars=['education_high_school_less', 'education_high_school_only', 'education_degree_some', 'education_bachelors_degree'], var_name='education')
# df_narrow_education = df.melt(id_vars=['poverty'], value_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_name='vaccine_hesitancy')
graph = sns.FacetGrid(df_narrow_education, col='education')
graph.map_dataframe(sns.scatterplot, x='value', y='vaccine_hesitant', s=scatterplot_marker_size)
graph.fig.subplots_adjust(top=0.8)
graph.fig.suptitle('Percent of vaccine hesitant population vs educational achievement')
_ = graph.set_axis_labels("Education Percent", "Vaccine Hesitant")


# In[71]:


graph = sns.FacetGrid(df_narrow_education, col='education')
graph.map_dataframe(sns.scatterplot, x='value', y='vaccine_hesitant_strong', s=scatterplot_marker_size)
graph.fig.subplots_adjust(top=0.8)
graph.fig.suptitle('Percent of strongly vaccine hesitant population vs educational achievement')
_ = graph.set_axis_labels("Education Percent", "Vaccine Hesitant")


# For the two plots for less than high school education, we see a slight linear relationship between vaccine hesitancy and increase of percent of less than high school education population.
# 
# This relationship weakens with the more educated high school graduate only plots.
# 
# The some post-secondary degree group does not have a significant relationship.
# 
# The bachelors degree and higher graphs show an inverse relationship between the amount of population with a bachelors degree and vaccine hesitancy.

# Religion

# In[72]:


# df_narrow_religion = df.melt(id_vars=['religion_total', 'religion_evangelical', 'religion_mainline_protestant'], value_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_name='Religion')
df_narrow_religion = pd.melt(df, id_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_vars=['religion_total', 'religion_evangelical', 'religion_mainline_protestant'], var_name='religion')
graph = sns.FacetGrid(df_narrow_religion, col='religion')
graph.map_dataframe(sns.scatterplot, x='value', y='vaccine_hesitant', s=scatterplot_marker_size)
graph.fig.subplots_adjust(top=0.8)
graph.fig.suptitle('Percent of vaccine hesitant population vs religious adherents')
_ = graph.set_axis_labels("Religious adherents per 1000", "Vaccine Hesitant")


# In[73]:


graph = sns.FacetGrid(df_narrow_religion, col='religion')
graph.map_dataframe(sns.scatterplot, x='value', y='vaccine_hesitant_strong', s=scatterplot_marker_size)
graph.fig.subplots_adjust(top=0.8)
graph.fig.suptitle('Percent of vaccine hesitant population vs religious adherents')
_ = graph.set_axis_labels("Religious adherents per 1000", "Vaccine Hesitant")


# Our hypothesis going into this was religiosity had an effect on vaccine hesitancy, however these graphs don't show as strong a relationship as expected.
# 
# Total religious adherents doesn't seem to have a relationship with vaccine hesitancy
# 
# Evangelical adherent numbers have a slight uptake as vaccine hesitancy increases but then does not increase much after the mean.
# 
# Mainline protestant has a slight bell curve distribution with a some concentration on the upper end of vaccine hesitancy, however not very strongly.

# Politics

# In[74]:


df_narrow_democrat_wins = df.melt(id_vars=['election_democrat_wins'], value_vars=['vaccine_hesitant', 'vaccine_hesitant_strong'], value_name='vaccine_hesitancy')
election_democrat_wins_boxplot = sns.boxplot(data=df_narrow_democrat_wins, x='election_democrat_wins', y='vaccine_hesitancy', hue='variable').set_title('Box plot distribution of percent vaccine hesitant population by number of\n Democrat party presidential wins in 2008-2020 elections')

# election_democrat_wins_boxplot = sns.boxplot(data=df, x='election_democrat_wins', y='vaccine_hesitant').set_title('Box plot distribution of percent vaccine hesitant population by number of\n Democrat party presidential wins in 2008-2020 elections')
# TODO: # votes democrat vs vaccine hesitancy


# As the number of Democrat party presidential wins in 2008-2020 elections increase, we see a slight decrease in both vaccine hesitant and strongly vaccine hesitant populations. The last box plot with 4 Democrat wins is a bit of an outlier since vaccine hesitancy increases a bit, however the variance, shown by the whiskers, of the population is much larger.

# Choropleth Visualization

# In[77]:


import folium

df_map = df[['fips', 'vaccine_hesitant']].copy()
df_map['fips'] = df_map['fips'].astype(str)

map = folium.Map(location=[39.8282, -98.5795], zoom_start=5)
folium.Choropleth(
    geo_data='https://raw.githubusercontent.com/python-visualization/folium/master/tests/us-counties.json',
    name="choropleth",
    data=df_map,
    columns=['fips', 'vaccine_hesitant'],
    key_on='feature.id',
    fill_color="YlGn",
    fill_opacity=0.7,
    line_opacity=.1,
    legend_name="Vaccine Hesitancy (%)",
).add_to(map)
folium.LayerControl().add_to(map)
map
# TODO: explore bivariate choropleth maps? hue (vaccine hesitancy) vs opacity (other demographic var)


# In[ ]:




