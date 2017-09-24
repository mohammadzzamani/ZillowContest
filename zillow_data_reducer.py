from multiprocessing import Process, Manager

# this file is sorted
zillow_file = 'data/properties_2016.csv'
id_index_z = 0

# this file is not sorted
language_file = 'data/csvX0.csv'
id_index_l = 0

fin_z = open(zillow_file, 'r')
fin_l = open(language_file, 'r')

z_lines = fin_z.readlines()
l_lines = fin_l.readlines()

l_ids = Manager().list()
z_reduced = Manager().list()

for l in l_lines[1:]:
    l_ids.append(l.split(',')[id_index_l])

def filtering(data_num, blank):
    for d in chunks[data_num]:
        if d.split(',')[0] in l_ids:
            z_reduced.append(d)

plist = list()
n = 8  # number of processes
l = len(z_lines)
chunks = Manager().list()
for c in [z_lines[int((l / n) * (x)):int((l / n) * (x + 1))] for x in range(n)]:
    chunks.append(c)
for i, chunk in enumerate(chunks):
    print("proc i -", i)
    plist.append(Process(target=filtering, args=(i, '')))

print("launching processes")
for p in plist:
    p.start()

for p in plist:
    p.join()


fout = open('z_reduced.txt', 'w')
fout.writelines(z_reduced)

