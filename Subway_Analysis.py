import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #removes sklearn warnings to make console outputs more readable

subway_data = pd.read_excel("ttc-subway-delay-data-2019.xlsx", sheet_name=None)
df = pd.concat(subway_data, ignore_index=True) #places all the excel worksheets into a single dataframe in order of date
df = df.dropna() #Drop null values in the dataframe as this is a sign the train did not run
average = df["Min Delay"].mean()
maximum = df["Min Delay"].max()
count = (df['Min Delay'] != 0).sum()
journeys = len(df.index)
percentage = count / journeys * 100
stations = (df["Station"].unique())
Station_delays = {"Station":[], "Delays":[]} #Creates a dictionary to store the station with the most delays as well as the amount of delays
highest = 0
for i in stations:
    counting = (df['Min Delay'] != 0).sum() and (df["Station"] == i).sum() #The variable stores the number of times there is a delay at each station
    if counting > highest:
        Station_delays = {"Station":i, "Delays":counting} #The station with the most delays gets added to the dictionary
        highest = counting
Total_journeys = (df["Station"] == Station_delays["Station"]).sum()
Top_station_delays_percentage = Total_journeys / Station_delays["Delays"] * 100
non_delay_percentage = 100 - percentage
print("2019 Toronto Subway Delay Statistics:") #Output of summary statistics
print("The average delay time was:", average)
print("The largest delay was:", maximum, "minutes")
print("There was a delay on", count, "journeys out of", journeys, "journeys, Which is", percentage, "% of journeys")
print("The largest number of delays was:", Station_delays["Delays"], "out of", Total_journeys, "at", Station_delays["Station"], "Percentage:", Top_station_delays_percentage)
print("The percentage of journeys without a delay was", non_delay_percentage,"%")
df[['Time', 'Station', 'Bound', 'Line',"Vehicle"]] = df[['Time', 'Station', 'Bound', 'Line', "Vehicle"]].apply(lambda x: pd.factorize(x)[0]) #test and train data must be numbers, this applies a numerical index to each value

x = df.loc[:,("Time", "Station", "Min Gap", "Bound", "Line", "Vehicle")] #Data used to train
y = df.loc[:,("Min Delay")] #Data for the model to predict
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42) #creates a 70/30 split for training and test data

clf=RandomForestClassifier(n_estimators=100) #sets the amount of iterations to 100
clf.fit(x_train,y_train) #fits the training data
y_pred=clf.predict(x_test) #Random forest attempts to predict the length of delay using the test data
Accuracy = metrics.accuracy_score(y_test, y_pred) #Compares the test predictions to the actual test data to determine the accuracy
print("Random Forest Classifier accurately predicted the length of a delay", Accuracy * 100, "% of the time")