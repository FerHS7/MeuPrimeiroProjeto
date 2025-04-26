import pandas as pd
import numpy as np

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
print(df.head())
# Check missing values
print("\nMissing values (%):")
print(df.isnull().sum()/len(df)*100)

# Data types
print("\nData types:")
print(df.dtypes)
launch_site_counts = df['LaunchSite'].value_counts()
print("\nLaunches per site:")
print(launch_site_counts)
orbit_counts = df['Orbit'].value_counts()
print("\nLaunch distribution by orbit type:")
print(orbit_counts)
landing_outcomes = df['Outcome'].value_counts()
print("\nLanding outcomes:")
print(landing_outcomes)

# Identify unsuccessful outcomes
bad_outcomes = set(landing_outcomes.index[[1, 3, 5, 6, 7]])
print("\nUnsuccessful landing categories:")
print(bad_outcomes)
df['Class'] = df['Outcome'].apply(lambda x: 0 if x in bad_outcomes else 1)
print("\nSample classified data:")
print(df[['Outcome', 'Class']].head(8))

# Calculate success rate
success_rate = df['Class'].mean()
print(f"\nOverall landing success rate: {success_rate:.2%}")
df.to_csv("dataset_part_2.csv", index=False)
print("\nData successfully saved to dataset_part_2.csv")