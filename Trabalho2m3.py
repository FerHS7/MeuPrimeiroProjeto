import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")
df.head()
#Task1-FlightNumber vs Launch Site
sns.catplot(x='FlightNumber', y='LaunchSite', hue='Class', data=df, aspect=5)
plt.xlabel("Flight Number", fontsize=15)
plt.ylabel("Launch Site", fontsize=15)
plt.show()
#Task2-Payload Mass vs Launch Site
sns.catplot(x='PayloadMass', y='LaunchSite', hue='Class', data=df, aspect=5)
plt.xlabel("Payload Mass (kg)", fontsize=15)
plt.ylabel("Launch Site", fontsize=15)
plt.show()

#Task3-Success or type the orbit
success_rate = df.groupby('Orbit')['Class'].mean().reset_index()
sns.barplot(x='Orbit', y='Class', data=success_rate)
plt.ylabel('Success Rate')
plt.title('Success Rate by Orbit Type')
plt.show()

#Task4-Flight Number vs Orbit Type
sns.catplot(x='FlightNumber', y='Orbit', hue='Class', data=df, aspect=5)
plt.xlabel("Flight Number", fontsize=15)
plt.ylabel("Orbit Type", fontsize=15)
plt.show()

#Task5-Payload vs orbit
sns.catplot(x='PayloadMass', y='Orbit', hue='Class', data=df, aspect=5)
plt.xlabel("Payload Mass (kg)", fontsize=15)
plt.ylabel("Orbit Type", fontsize=15)
plt.show()

#Task6-successful trend by year
def Extract_year(date):
    return [i.split("-")[0] for i in date]

df['Year'] = Extract_year(df["Date"])

yearly_success = df.groupby('Year')['Class'].mean().reset_index()
sns.lineplot(x='Year', y='Class', data=yearly_success)
plt.title('Launch Success Trend Over Years')
plt.ylabel('Success Rate')
plt.show()

#5. Attribute Engineering
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights',
               'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]

#Task7One-hot encoding(categorical variables)
features_one_hot = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])
features_one_hot.head()

#Task 8 – Convert everything to float64
features_one_hot = features_one_hot.astype('float64')

#6.save for later
features_one_hot.to_csv('dataset_part_3.csv', index=False)

#testing the code

['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights',
 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights',
               'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]

print("Orbit:", df['Orbit'].nunique())
print("LaunchSite:", df['LaunchSite'].nunique())
print("LandingPad:", df['LandingPad'].nunique())
print("Serial:", df['Serial'].nunique())

total_columns = 8 + 11 + 3 + 5 + 53  # 8 numéricas + dummies
print(total_columns)  # Result:
