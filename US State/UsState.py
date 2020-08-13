import csv
from matplotlib import pyplot as plt
filename= 'us_states_daily.csv'
output_filename= 'MA.csv'
res=[]
with open(filename,'r') as f:
    with open(output_filename,'w') as out:

        reader=csv.reader(f)
        writer=csv.writer(out)
        header_row=next(reader)
        writer.writerow(header_row)
        for row in reader:
            if(row[1]=="MA"):
                writer.writerow(row)





