import pandas as pd
import glob

path = "../../../Models/mmnet*/results.csv"

count = 0
for filename in glob.glob(path):
    if count == 0:
        df_metrics = pd.read_csv(filename)
        count += 1
    else:
        df_metrics_temp = pd.read_csv(filename)
        df_metrics = pd.concat([df_metrics, df_metrics_temp])
        
df_metrics.to_csv("../../../Models/mmnet_all_results.csv", index=False)

