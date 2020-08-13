import csv
from matplotlib import pyplot as plt
filename= '../../Us State/AI_HUB/us_daily.csv'
with open(filename) as f:
    reader=csv.reader(f)
    header_row=next(reader)


    positive=[]
    for row in reader:
        positive.append(int(row[17]))
fig=plt.figure(dpi=128,figsize=(10,6))
plt.plot(positive,c='red')
plt.show()

