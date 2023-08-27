'''
1 Data exploration: a complete review and analysis of the dataset 
1.1 Load the 'Bicycle_Thefts.csv' file into a dataframe and descibe data elements (columns),
provide descriptions & types, ranges and values of elements as appropriate.
1.2 Statistical assessments including means, averages, correlations
1.3 Missing data evaluations – use pandas, NumPy and any other python packages
1.4 Graphs and visualizations – use pandas, matplotlib, seaborn, NumPy and any other python packages, you also can use power BI desktop.
'''

import pandas as pd
import os
import numpy as np

path = "C:/Users/User/Desktop/"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
print(fullpath)

data_bicycle = pd.read_csv(fullpath)
data_bicycle.columns.values
data_bicycle.shape
data_bicycle.describe()
data_bicycle.describe
data_bicycle.dtypes
data_bicycle.head(5)

data_bicycle['Division'].describe()
data_bicycle['Division'].unique()
grouped = data_bicycle.groupby('Division')
grouped.groups

data_bicycle['Neighbourhood'].describe()
data_bicycle['Neighbourhood'].unique()

data_bicycle['Premise_Type'].describe()
data_bicycle['Premise_Type'].unique()

data_bicycle['Location_Type'].describe()
data_bicycle['Location_Type'].unique()

data_bicycle['Bike_Make'].describe()
data_bicycle['Bike_Make'].unique()

data_bicycle['Bike_Colour'].describe()
data_bicycle['Bike_Colour'].unique()

data_bicycle['Cost_of_Bike'].describe()
data_bicycle['Cost_of_Bike'].unique()

data_bicycle['Bike_Make'].describe()
data_bicycle['Bike_Make'].unique()

data_bicycle['Bike_Type'].describe()
data_bicycle['Bike_Type'].unique()

data_bicycle['Bike_Speed'].describe()
data_bicycle['Bike_Speed'].describe()

data_bicycle = pd.read_csv(fullpath)

# For Cost_of_Bike, fill missing with median
# Check how many null before fill the missing 
print(data_bicycle['Cost_of_Bike'].isnull().sum()) #1536
print(data_bicycle['Cost_of_Bike'].notnull().sum()) #20048
median = data_bicycle['Cost_of_Bike'].median()
print(median)

# Fill missing value with median
data_bicycle['Cost_of_Bike'].fillna(median, inplace= True)

# Check how many null after fill the missing 
print(data_bicycle['Cost_of_Bike'].isnull().sum()) #0
print(data_bicycle['Cost_of_Bike'].notnull().sum()) #21584

# For Bike_Model, fill missing with "UNKNOWN"
# Check how many null before fill the missing 
print(data_bicycle['Bike_Model'].isnull().sum().sum()) #8140
data_bicycle['Bike_Model'].fillna('UNKNOWN', inplace= True)

# Check how many null after fill the missing 
print(data_bicycle['Bike_Model'].isnull().sum()) #0

# For Bike_Colour, fill missing with "UNKNOWN"
# Check how many null before fill the missing 
print(data_bicycle['Bike_Colour'].isnull().sum().sum()) #1729
data_bicycle['Bike_Colour'].fillna('UNKNOWN', inplace= True)

# Check how many null after fill the missing 
print(data_bicycle['Bike_Colour'].isnull().sum()) #0

from matplotlib import pyplot as plt
# Create a scatterplot
fig_Premise_Cost = data_bicycle.plot(kind='scatter',x='Premise_Type',y='Cost_of_Bike')
fig_Location_Cost = data_bicycle.plot(kind='scatter',x='Location_Type',y='Cost_of_Bike')
fig_Division_Cost = data_bicycle.plot(kind='scatter',x='Division',y='Cost_of_Bike')

# Save the scatter plot
figfilename = "ScatterPlot_Liping.pdf"
figfullpath = os.path.join(path, figfilename)
fig_Premise_Cost.figure.savefig(figfullpath)

# Plot a histogram
import matplotlib.pyplot as plt
hist_year= plt.hist(data_bicycle['Occurrence_Year'],bins=12)
plt.xlabel('Occurrence_Year')
plt.ylabel('and Stolen')
plt.title('Occurrence_Year and Stolen')
plt.show()

import matplotlib.pyplot as plt
hist_month= plt.hist(data_bicycle['Occurrence_Month'],bins=12)
plt.xlabel('Occurrence_Month')
plt.ylabel('and Stolen')
plt.title('Occurrence_Month and Stolen')
plt.show()

# Plot a histogram
import matplotlib.pyplot as plt
hist= plt.hist(data_bicycle['Occurrence_Time'],bins=24)
plt.xlabel('Occurrence_Time')
plt.ylabel('and Stolen')
plt.title('Occurrence_Time and Stolen')
plt.show()

# Plot a histogram
import matplotlib.pyplot as plt
hist_location= plt.hist(data_bicycle['Location_Type'],bins=12)
plt.xlabel('Location_Type')
plt.ylabel('and Stolen')
plt.title('Location_Type and Stolen')
plt.show()

# Plot a histogram
import matplotlib.pyplot as plt
hist_Premise= plt.hist(data_bicycle['Premise_Type'],bins=12)
plt.xlabel('Premise_Type')
plt.ylabel('Stolen')
plt.title('Premise_Type and Stolen')
plt.show()

import matplotlib.pyplot as plt
hist_Division= plt.hist(data_bicycle['Division'],bins=12)
plt.xlabel('Division')
plt.ylabel('Stolen')
plt.title('Division and Stolen')
plt.show()

# Plot a boxplot
import matplotlib.pyplot as plt
plt.boxplot(data_bicycle['Occurrence_Day'])
plt.ylabel('Occurrence_Day')
plt.title('Box Plot of Occurrence_Day')
plt.show()

# Plot a boxplot
import matplotlib.pyplot as plt
plt.boxplot(data_bicycle['Cost_of_Bike'])
plt.ylabel('Cost_of_Bike')
plt.title('Box Plot of Cost_of_Bike')
plt.show()

# Plot a boxplot
import matplotlib.pyplot as plt
plt.boxplot(data_bicycle['X'])
plt.ylabel('X')
plt.title('Box Plot of X')
plt.show()

# Plot a boxplot
import matplotlib.pyplot as plt
plt.boxplot(data_bicycle['Y'])
plt.ylabel('Y')
plt.title('Box Plot of Y')
plt.show()