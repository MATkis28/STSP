import matplotlib
import matplotlib.pyplot as plot
import numpy as np

naglowki = 1
nazwaIn = "15_02_21_02_41_47" #nazwa pliku z danymi

space = 0

kolumny = 0
proby = 0
plik = open(nazwaIn, 'r')
dane = []
bKolsSet = False

for line in plik:
    if (len(line) == 0):
        continue
    tokenized = []
    gotKol = 0
    for word in line.split():
        tokenized.append(word)
        if (not bKolsSet):
            kolumny += 1
        gotKol += 1
        
    for i in range(gotKol, kolumny):
        tokenized.append("")
    dane.append(tokenized)
    proby += 1
    bKolsSet = True
plik.close()
print(kolumny)
proby -= naglowki

#generowanie wykresow
colors = ['green','blue','aqua','purple','red','orange','black']
ncolor = 0

def wykres(x,y,blady,label,bladx=0):
    global ncolor
    z1 = np.polyfit(x, y, 7)
    p1 = np.poly1d(z1)
    interpolacja = p1([i/100 for i in range(x[len(x)-1]*100, 1000)])
    plot.plot([i/100 for i in range(x[len(x)-1]*100, 1000)], interpolacja,color=colors[ncolor])
    plot.errorbar(x,y,yerr=blady,fmt='o',label=label,color=colors[ncolor])
    ncolor=ncolor+1

for i in range(kolumny):
    print(str(i)+" "+dane[0][i])
wierzcholki = [int(dane[x][0]) for x in range(1,proby+1)]
czasgen = list()
bczasgen= list()
labels = ["Populacja 10 Generacje 20","Populacja 10 Generacje 50","Populacja 20 Generacje 20","Populacja 20 Generacje 50","Populacja 40 Generacje 20","Populacja 40 Generacje 40"]
for i in range(6):
    czasgen.append([float(dane[x][1+i]) for x in range(1,proby+1)])
    bczasgen.append([float(dane[x][18+i]) for x in range(1,proby+1)])
    wykres(wierzcholki,czasgen[i],bczasgen[i],labels[i])
plot.legend()
plot.title("Zależność czasu od liczby wierzchołków")
plot.xlabel("Wierzchołki")
plot.ylabel("Czas")
plot.show()

ncolor = 0
jakgen=list()
bjakgen=list()
for i in range(6):
    jakgen.append([float(dane[x][10+i]) for x in range(1,proby+1)])
    bjakgen.append([float(dane[x][26+i]) for x in range(1,proby+1)])
    wykres(wierzcholki,jakgen[i],bjakgen[i],labels[i])
plot.legend()
plot.title("Zależność jakości od liczby wierzchołków")
plot.xlabel("Wierzchołki")
plot.ylabel("Jakosc")
plot.show()

ncolor = 0
jakgen=list()
bjakgen=list()
czasgen = list()
bczasgen= list()
#gen
jakgen.append([float(dane[x][15]) for x in range(1,proby+1)])
bjakgen.append([float(dane[x][31]) for x in range(1,proby+1)])
czasgen.append([float(dane[x][6]) for x in range(1, proby + 1)])
bczasgen.append([float(dane[x][23]) for x in range(1, proby + 1)])
#los
jakgen.append([float(dane[x][16]) for x in range(1,proby+1)])
bjakgen.append([float(dane[x][32]) for x in range(1,proby+1)])
czasgen.append([float(dane[x][8]) for x in range(1, proby + 1)])
bczasgen.append([float(dane[x][24]) for x in range(1, proby + 1)])
#zach
jakgen.append([float(dane[x][17]) for x in range(1,proby+1)])
bjakgen.append([float(dane[x][25]) for x in range(1,proby+1)])
czasgen.append([float(dane[x][9]) for x in range(1, proby + 1)])
bczasgen.append([float(dane[x][33]) for x in range(1, proby + 1)])

labels=["Genetyczny Populacja 40 Generacje 40","Losowy","Zachłanny"]

ncolor = 0
for i in range(3):
    wykres(wierzcholki,czasgen[i],bczasgen[i],labels[i])
plot.legend()
plot.title("Zależność czasu od liczby wierzchołków")
plot.xlabel("Wierzchołki")
plot.ylabel("Czas")
plot.show()

ncolor = 0
for i in range(3):
    wykres(wierzcholki,jakgen[i],bjakgen[i],labels[i])
plot.legend()
plot.title("Zależność jakości od liczby wierzchołków")
plot.xlabel("Wierzchołki")
plot.ylabel("Jakosc")
plot.show()