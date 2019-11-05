f = open("train.txt","r")
lines = f.readlines()
for line in lines:
    line
    txt = line[0:6]
    #print(txt)
    f=txt+'.txt'
    file = open(f, 'w')
    #print(f)
    line=line[22:]
    #print(line)
    for db in line.split():
        #print(db)
        file.write(db[:-2])
        file.write('\n')

