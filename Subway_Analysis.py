import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sns.set()

subway_data = pd.read_excel("ttc-subway-delay-data-2019.xlsx", sheet_name=None)
df = pd.concat(subway_data, ignore_index=True)
df = df.dropna() #Drop null values in the dataframe as this is a sign the train did not run
stations = (df["Station"].unique())
Station_delays = {"Station":[], "Delays":[]}
highest = 0
for i in stations:
    counting = (df['Min Delay'] != 0).sum() and (df["Station"] == i).sum()
    if counting > highest:
        Station_delays = {"Station":i, "Delays":counting}
        highest = counting
Total_journeys = (df["Station"] == Station_delays["Station"]).sum()
Top_station_delays_percentage = Total_journeys / Station_delays["Delays"] * 100
average = df["Min Delay"].mean()
print("2019 Toronto Subway Delay Statistics:")
print("The average delay time was:", average)
maximum = df["Min Delay"].max()
print("The largest delay was:", maximum, "minutes")
count = (df['Min Delay'] != 0).sum()
journeys = len(df.index)
percentage = count / journeys * 100
non_delay_percentage = 100 - percentage
print("There was a delay on", count, "journeys out of", journeys, "journeys, Which is", percentage, "% of journeys")
description = df.describe()
df[['Time', 'Station', 'Bound', 'Line',"Vehicle"]] = df[['Time', 'Station', 'Bound', 'Line', "Vehicle"]].apply(lambda x: pd.factorize(x)[0])

print("The largest number of delays was:", Station_delays["Delays"], "at", Station_delays["Station"], "Percentage:", Top_station_delays_percentage)

x = df.loc[:,("Time", "Station", "Min Gap", "Bound", "Line", "Vehicle")]
y = df.loc[:,("Min Delay")]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
Accuracy = metrics.accuracy_score(y_test, y_pred)
print("Random Forest Classifier accurately predicted a delay", Accuracy * 100, "% of the time")