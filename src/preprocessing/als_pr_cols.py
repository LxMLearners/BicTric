import pandas as pd

data = pd.read_csv("data/als_data_2021/dataWithoutDunnoNIV_90.csv")

ref_i = 0
date1 = None
alsfrs1 = None

ds = list()
als = list()

for index, row in data.iterrows():
    ref = row['REF']
    if ref != ref_i:
        date1 = row['medianDate']
        alsfrs1 = row['ALS-FRS-R']
    ds.append(date1)
    als.append(alsfrs1)
    ref_i = ref

data['Date1'] = ds
data['ALS-FRS-R1'] = als
data.to_csv("data/als_data_2021/dataWithoutDunnoNIV_90.csv", index=False)
