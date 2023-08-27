## Summary & Overview of solution

Toronto Police and residents are having a hard time because of the cases of bicycle theft in different regions of Toronto.<br>
This project is conducted for public safety and awareness from local bicycle theft crimes.<br>
It helps people to analyze whether a stolen bicycle will be returned or not.<br><br>
Toronto police will be able to further strengthen their solutions to prevent theft in certain areas, and residents will be extra careful and seek preventive measures such as anti-theft locks.<br>
Therefore, this will gradually reduce the number of bicycle theft cases.<br><br>
This analysis is based on the open data provided by [Toronto government and police](https://data.torontopolice.on.ca/pages/open-data).

## Activities

- **Data Exploration**
    - **1.1** Load the 'Bicycle_Thefts.csv' file into a dataframe and descibe data elements (columns),
provide descriptions & types, ranges and values of elements as appropriate.
    - **1.2** Statistical assessments including means, averages, correlations
    - **1.3** Missing data evaluations – use pandas, NumPy and any other python packages.
    - **1.4** Graphs and visualizations – use pandas, matplotlib, seaborn, NumPy and any other python packages.

- **Data Modelling**
    - **2.1** Data transformations – includes handling missing data, categorical data management, data normalization and standardizations as needed.
    - **2.2** Feature selection – use pandas and sci-kit learn.
    - **2.3** Train, Test data splitting – use NumPy, sci-kit learn.
 
- **Predictive model building**
    - **3.1** Use logistic regression and decision trees as a minimum – use scikit learn.
 
- **Model scoring and evaluation**
    - **4.1** Present results as scores, confusion matrices and ROC - use sci-kit learn.
    - **4.2** Select and recommend the best performing model.
 
- **Deploying the model**
    - **5.1** Using flask framework arrange to turn your selected machine-learning model into an API.
    - **5.2** Using pickle module arrange for Serialization & Deserialization of your model.
    - **5.3** Build a client to test your model API service. Use the test data, which was not previously used to train the module.

## Metadata

| Field | Field_Description | ObjectId |
| :-------------: |:-------------:| :-----:|
| Index | Record Unique Identifier | 1 |
| event_unique_id | Event Occurrence Identifier | 2 |
| Primary_Offence | Offence related to the occurrence | 3 |
| Occurrence_Date | Date of occurrence | 4 |
| Occurrence_Year | Occurrence year | 5 |
| Occurrence_Month | Occurrence Month | 6 |
| Occurrence_Day | Occurrence Day | 7 |
| Occurrence_Time | Occurrence Time | 8 |
| Division | Police Division where event occurred | 9 |
| City | City where event occurred | 10 |
| Location_Type | Location Type where event occurred | 11 |
| Premise_Type | Premise Type where event occurred | 12 |
| Bike_Make | Bicycle Make | 13 |
| Bike_Model | Bicycle Model | 14 |
| Bike_Type | Bicycle Type | 15 |
| Bike_Speed | Bicycle Speed | 16 |
| Bike_Colour | Bicycle Colour | 17 |
| Cost_of_Bike | Cost of Bicycle | 18 |
| Status | Status of event | 19 |
| Lat | Longitude of point extracted after offsetting X and &<br> Coordinates to nearest intersection node | 20 |
| Long | Latitude of point extracted after offsetting X and &<br> Coordinates to nearest intersection node | 21 |

## Data Insights & Visualizations
### Pie Chart
<img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/pie_chart_toronto_bike_status.PNG" width="500">

### Map
<img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/map_chart_toronto_bike_status.PNG" width="800">

### Location
<p float="left">
    <img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/Top5_Dangerous_Division.png" width="500">
    <img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/Top5_Safe_Division.png" width="500">
</p>
<p float="left">
    <img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/Top10_Dangerous_Neighborhood.png" width="500">
    <img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/Top10_Safe_Neighborhood.png" width="500">
</p>
<img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/Premise_Type and Stolen.png" width="500">

### Time
<p float="left">
    <img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/Occurrence_Year and Stolen.png" width="500">
    <img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/Occurrence_Month and Stolen.png" width="500">
</p>
<p float="left">
    <img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/Occurrence_Day of Week and Stolen.png" width="500">
    <img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/Occurrence_Hour of a Day and Stolen.png" width="500">
</p>

### Bike
<p float="left">
    <img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/Bike_Make and Stolen.png" width="500">
    <img src="https://github.com/danadayoonlee/toronto-bike-theft-analysis/blob/main/Bike_Colour and Stolen.png" width="500">
</p>
