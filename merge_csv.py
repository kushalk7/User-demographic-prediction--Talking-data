import pandas as pd

a = pd.read_csv("C:\\Users\\Kushal\\Downloads\\TalkingData\\TalkingData\\events.csv")
b = pd.read_csv("C:\\Users\\Kushal\\Downloads\\TalkingData\\TalkingData\\gender_age_train.csv")
merged = a.merge(b, on='device_id')
merged.to_csv("C:\\Users\\Kushal\\Downloads\\TalkingData\\TalkingData\\output.csv", index=False)
