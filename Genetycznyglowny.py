import random
import math
import statistics

import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import itertools
import time

n = 9              #liczba wierzcholkow
nmax = 20          # do ilu wierzcholkow liczyc odchylenia
pop = 30           #populacja
gen = 100          #generacje
maxPos = 50         #maksymalne x i y wierzcholka (minimalne to 0)
minSkalar = 1.4     #minimalna wartosc skalara (zmiennoprzecikowa)
maxSkalar = 4.4     #maksymalna wartosc skalara (zmiennoprzecikowa)
skalaWykresu = 8
powtorzenia = 100  #powtorzenia algorytmu genetycznego dla jednego grafu

def generujGraf(n):
    graf = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            graf[i][j] = graf[j][i] = math.sqrt((punkty[j][0] - punkty[i][0])**2 + (punkty[j][1] - punkty[i][1])**2)
            if (graf[i][j] < 2):
                j = 0
                punkty[i] = (random.randint(0,maxPos), random.randint(0,maxPos))
    return graf

def losujSciezke():
    solution = [i for i in range(n)]
    random.shuffle(solution)
    solution.append(solution[0])
    return solution

def ocena(sol):
    return sum([graf[sol[i - 1]][sol[i]] * skalary[i - 1] for i in range(1, n + 1)])

def ocenaSort(sols):
    sols.sort(key=ocena)
    return sols

def napraw(b_sol,krzyzowane,start,koniec):
    sol = b_sol[:]
    for i in range(start): # naprawa przed [start]
        if krzyzowane.count(sol[i]) > 0:
            for j in np.random.permutation([i for i in range(n)]):
                if sol[:n].count(j)==0:
                    sol[i]=j
                    break
    for i in range (koniec+1,n): # naprawa po [koniec]
        if krzyzowane.count(sol[i]) > 0:
            for j in np.random.permutation([i for i in range(n)]):
                if sol[:n].count(j)==0:
                    sol[i]=j
                    break
    sol[n]=sol[0]
    return sol

def napraw2(b_sol,krzyzowane,start,koniec):
    sol = b_sol[:]
    dun = [0 for i in range(n)]
    for i in range(n):
        dun[sol[i]] = dun[sol[i]]+1
    indeksy = [i for i in range(n)]
    random.shuffle(indeksy)
    k = 0
    for i in range(start): # naprawa przed [start]
        if dun[sol[i]]>1:
            while (dun[indeksy[k]]!=0):
                k=k+1
            sol[i] = indeksy[k]
            dun[indeksy[k]] = dun[indeksy[k]]+1
    for i in range (koniec+1,n): # naprawa po [koniec]
        if dun[sol[i]]>1:
            while (dun[indeksy[k]]!=0):
                k=k+1
            sol[i] = indeksy[k]
            dun[indeksy[k]] = dun[indeksy[k]]+1
    sol[n]=sol[0]
    return sol

def krzyzuj(sol1,sol2):
    start = random.randint(0,n-2)
    koniec = random.randint(start,n-2)
    krzyzowane = sol2[start:koniec+1]
    sol = sol1[:start]+krzyzowane+sol1[koniec+1:]
    sol = napraw2(sol,krzyzowane,start,koniec)
    return sol

def mutuj (sol):
    xsol = sol[:]
    w1 = random.randint(0,n-1)
    w2 = random.randint(1, n - 1)
    if w2 <= w1:
        w2 = w2 - 1
    wtemp = xsol[w1]
    xsol[w1]=xsol[w2]
    xsol[w2]=wtemp
    xsol[n]=xsol[0]
    return xsol

def krzyzimutacja(sol1,sol2):
    sol = krzyzuj(sol1,sol2)
    return mutuj(sol)

def genetyka(graf, populacja, generacje):
    sols = list([])
    for i in range(populacja):
        sols.append(losujSciezke())
    sols = ocenaSort(sols)
    sols=sols[:populacja]
    for i in range(generacje):
        solsnew = sols[:]
        for j in range(round(populacja / 2)):
            solsnew.append(krzyzimutacja(sols[j * 2], sols[j * 2 + 1]))
            solsnew.append(krzyzimutacja(sols[j * 2 + 1], sols[j * 2]))
        solsnew = ocenaSort(solsnew)
        sols=solsnew[:populacja]
    return sols

def zachlanny(graf, skalary):
    bestSol = []
    bestSum = sum(max(graf))*max(skalary)
    for p in itertools.permutations([i for i in range(n)]):
        sol = list(p)
        sol.append(sol[0])
        Sum = sum([graf[sol[i-1]][sol[i]]*skalary[i-1] for i in range(1, n+1)])
        if bestSum > Sum:
            bestSol = sol
            bestSum = Sum
    return (bestSum, bestSol)

def rysujKrawedz(i, rozwiazanie, punkty, skalary, skalaWykresu):
    indeks = rozwiazanie[i]
    indeksPrev = rozwiazanie[i-1]
    #Kolor zalezy od wartosci skalara. Minimalna wartosc jest niebieska, maksymalna czerwona, reszta pomiedzy
    if (np.unique(max(skalary))-np.unique(min(skalary)) == 0):
        kolor = 0
    else:
        kolor = (skalary[i-1]-np.unique(min(skalary)))/(np.unique(max(skalary))-np.unique(min(skalary)))
    plot.plot([punkty[indeksPrev][0], punkty[indeks][0]], [punkty[indeksPrev][1], punkty[indeks][1]],
              c = (kolor[0],0,1-kolor[0]), markersize = skalaWykresu/4, zorder=1)
    
def rysujRozwiazanie(rozwiazanie,title ="", skalaWykresu = skalaWykresu):
    plot.scatter([punkty[i][0] for i in range(n)], [punkty[i][1] for i in range(n)], s=skalaWykresu, zorder=2)
    plot.title(title)
    plot.xlabel("X")
    plot.ylabel("Y")
    for i in range(1, len(rozwiazanie)):
            rysujKrawedz(i, rozwiazanie, punkty, skalary, skalaWykresu)
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(0, 1, N)
    vals[:, 1] = np.linspace(0, 0, N)
    vals[:, 2] = np.linspace(1, 0, N)
    newcmp = matplotlib.colors.ListedColormap(vals)
    plot.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min(skalary), vmax=max(skalary)),cmap=newcmp))


print("\n========================\n    Obliczenie błedu:\n")
print("Populacje: ",pop,"\nGeneracje: ",gen,"\nPowtorzenia: ",powtorzenia,"\n")
odchylenia = list()
for i in range(n,nmax+1):
    print("Wierzcholki: ",i)
    nstart = time.time()
    wynikg = list()
    n = i
    skalary = [(minSkalar + random.random() * (maxSkalar - minSkalar)) for _ in range(n)]
    punkty = [(random.randint(0, maxPos), random.randint(0, maxPos)) for _ in range(n)]
    graf = generujGraf(n)
    for j in range(powtorzenia):
        sols = genetyka(graf, pop, gen)
        wynikg.append(ocena(sols[0]))
    odchylenie = math.sqrt(statistics.variance(wynikg) / (len(wynikg) - 2))
    odchylenia.append(odchylenie)
    nkoniec=time.time()
    print("Dlugosc sciezki dla ostatniego powtorzenia: ", ocena(sols[0]))
    print("Odchylenie: ", odchylenie)
    print("Czas: ",nkoniec-nstart,"s\n")
odchylenieodchylenia = math.sqrt(statistics.variance(odchylenia) / (len(odchylenia)*(len(odchylenia) - 1)))
print("Odchylenie odchylenia: ",odchylenieodchylenia)
