import csv
import numpy as np
name=[]
for i in range(9):
    name.append(0)
print(name)

with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
    with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/creatdata.csv"), 'w', newline='') as f2:
        writer = csv.writer(f2)
        cs = list(csv.reader(f))
        for i, rows in enumerate(cs):
            row = []
            for j, s in enumerate(rows):
                if name[int(rows[10]) - 1] >= 3000:
                    break
                else:
                    name[int(rows[10]) - 1] += 1

                row.append(s)
            if name[int(rows[10]) - 1] >= 3000:
                continue
            writer.writerow(row)
        # print(data)

print(name)