import pandas as pd
import ast
import numpy as np
import collections

nums = [str(i).zfill(5) for i in range(2105)]

realdata = []
for i in range(2105):
    with open ("data/sparserep/part-" + nums[i], "r") as myfile:
        data=myfile.readlines()
    realdata.extend(data)

data = []
for i in range(len(realdata)):
    try:
        data.append(eval(realdata[i].replace("List","")))
    except SyntaxError:
        print("error" + str(i))
print("read data")

labels = pd.read_csv("readmission_labels.csv")
labels.head()

df = pd.DataFrame.from_records(data)
df = df.merge(labels, how = "left", left_on = 0, right_on = "HADM_ID")
del df["HADM_ID"]

def svmlight(testvec, label):
    writestring = "0" if label == 0 else "1"
    try:
        sortedtestvec = tuple(sorted(testvec, key=lambda item: item[0]))
        for i in sortedtestvec:
            writestring = writestring + " " + str(i[0])+ ":" + str(i[1])
    except TypeError:
        writestring = writestring + " " + str(testvec[0])+ ":" + str(testvec[1])
    return writestring

df["svmlight"] = np.vectorize(svmlight)(df[1],df["test"])
print("svmlight created")


sortedids = pd.read_csv("subj_hadm_ids_sorted.csv")
print("sorted ids read")

dffinal = sortedids.merge(df, how = "inner", left_on = "HADM_ID", right_on = 0)
del dffinal["HADM_ID"]


dffinal[["svmlight"]].to_csv("data.svmlight", index = False, header = False)
print("done")