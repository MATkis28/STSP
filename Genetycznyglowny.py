import random
import time
import copy
import math
import itertools
import statistics

import matplotlib
import matplotlib.pyplot as plot
import numpy as np

n = 800                                                 #liczba wierzcholkow
populacja = 10                                          #licznosc populacji
wstepnaPopulacja = populacja*4                          #liczba losowan rozwiazan do wygenerowania wstepnej populacji
generacje = 20                                          #liczba generacji
maxPos = 50                                             #maksymalne x i y wierzcholka (musi byc wieksze od 1) w pierwszym sposobie generowania grafu
minDys = 2                                              #minimalny dystans w 1. metodzie generacji algorytmu
maxWaga = 50                                            #maksymalna waga krawedzi w drugim sposobie generowania grafu
minSkalar = 1.4                                         #minimalna wartosc skalara (zmiennoprzecikowa)
maxSkalar = 4.4                                         #maksymalna wartosc skalara (zmiennoprzecikowa)
skalaWykresu = 8                                        #zmienna wprowadzajace kosmetyczne zmiany
powtorzenia = 100                                       #powtorzenia algorytmu genetycznego dla jednego grafu
czas = 2                                                #maksymalny czas w sekundach dzialania algorytmu losowego
#sekcja optymalizacyjna
cache = []
indeksy = [i for i in range(n)] 
kolejnosc = [i for i in range(populacja)]
stop = populacja - (populacja%2)

#Generuje graf, ktory mozna ladnie przedstawic graficznie
def generujGrafNaPlaszczyznie(n, punkty, skalary):
    global cache
    graf = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            graf[i][j] = graf[j][i] = math.sqrt((punkty[j][0] - punkty[i][0])**2 + (punkty[j][1] - punkty[i][1])**2)
            if (graf[i][j] < minDys):
                j = 0
                punkty[i] = (random.randint(0, maxPos), random.randint(0, maxPos))
    cache = [np.array(graf)*skalary[i] for i in range(n)]
    return graf

#Bardziej ogolna metoda generacji grafu
def generujGraf(n, skalary):
    global cache
    graf = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            graf[i][j] = graf[j][i] = random.random() * maxWaga
    cache = [np.array(graf)*skalary[i] for i in range(n)]
    return graf

#Generuje rozwiazania, z ktorych powstaje wstepna populacja
def losujSciezke():
    global indeksy
    solution = copy.deepcopy(indeksy)
    random.shuffle(solution)
    solution.append(solution[0])
    return solution

#Funkcja oceny
def ocena(sol):
    global cache
    global n
    suma = 0
    for i in range(n):
        #if (sol[i] == sol[i+1]):
            #print(sol)                                     #nie powinno sie zdarzyc i sie nie zdarza. Odkomentowac dla testow
        suma += cache[i][sol[i]][sol[i+1]]                  #cache to tablica 3 wymiarowa, ktora zawiera wszystkie przeskalowane
                                                            #wagi krawedzi, zeby nie trzeba bylo tego robic za kazdym razem
    return suma

def napraw(sol, start, koniec):
    global n, indeksy
    wolne = [1] * n                                         #wolne[i] posiada wartosc 1, jezeli wierzcholek o indeksie i nie zostal wykorzystany
    for i in range(n):
        wolne[sol[i]] = 0
    random.shuffle(indeksy)                                 #indeksy[i] to losowy wierzcholek, po to by naprawa wrzucala losowy wierzcholek w miejsce nieprawidlowego
    k = 0
    uzyte = [0] * n                                         #uzyte sluzy do znalezienia powtarzajacych sie wierzcholkow
    for i in range(start, koniec):                          #nie zmieniamy wierzcholkow z fragmentu, ktory zostal wrzucony z krzyzowania, dlatego te wszystkie wierzcholki 
        uzyte[sol[i]] = 1                                   #sa od razu w uzyciu i to one wymuszaja poprawe wierzcholka, gdy taki sam wystepuje w otoczeniu tego fragmentu
    for i in range(n):
        if i >= start and i < koniec:
            continue
        if uzyte[sol[i]]:
            while (not wolne[indeksy[k]]):
                k += 1
            sol[i] = indeksy[k]
            wolne[indeksy[k]] = 0
        uzyte[sol[i]] = 1
    sol[n]=sol[0]

#Wybranie z populacji "najlepszych" osobnikow
def operatorSelekcji(sols):
    global populacja
    newsols = sols 
    [newsols.append(x) for x in sols if x not in newsols]
    sols = newsols
    sols.sort(key=ocena)
    sols=sols[:populacja]

#Drobne losowe zmiany
def operatorMutacji(sol):
    global n
    w1 = random.randint(0, n - 1)                               #losowanie dwoch wierzcholkow na sciezce
    w2 = random.randint(1, n - 1)
    if w2 <= w1:
        w2 = w2 - 1                                             #dzieki temu zabiegowi mamy gwarancje, ze w1 != w2
    sol[w1], sol[w2] = sol[w2], sol[w1]                         #zamiana miejscami dwoch wybranych wierzcholkow
    sol[n]=sol[0]                                               #dla pewnosci, jezeli w1 albo w2 bylo rowne 0, trzeba poprawic rozwiazanie   

#Stworzenie dwoch nowych rozwiazan na podstawie dwoch podanych
def krzyzuj(sols, indeks1, indeks2):
    sol1 = sols[indeks1]
    sol2 = sols[indeks2]
    start = random.randint(0,n-2)                           
    koniec = random.randint(start,n-2)
    newSol1 = sol1[:start]+sol2[start:koniec]+sol1[koniec:]     #utworzenie nowego osobnika nakladajac wylosowany fragment na pierwszego osobnika
    napraw(newSol1, start, koniec)                              #doprowadzenie nowego osobnika do stanu w ktorym przestawia on cykl Hamiltona
    operatorMutacji(newSol1)                                    #mozna zamienic kolejnosc krzyzowania z mutacja, ale z naszych testow wynika, ze nie ma to duzego wplywu
    newSol2 = sol2[:start]+sol1[start:koniec]+sol2[koniec:]
    napraw(newSol2, start, koniec)
    operatorMutacji(newSol2)
    return (newSol1,newSol2)

#Zmieszanie dwoch rozwiazan w celu poszukiwania nowych
def operatorKrzyzowania(sols):                  
    global kolejnosc, populacja, stop
    random.shuffle(kolejnosc)
    j = 0
    while (j != stop):
        indeks1 = kolejnosc[j]
        j += 1
        indeks2 = kolejnosc[j]
        j += 1
        (newSol1, newSol2) = krzyzuj(sols, indeks1, indeks2)
        sols.append(newSol1)
        sols.append(newSol2)

#Wykonuje algorytm genetyczny
def genetyczny(graf):
    global wstepnaPopulacja, populacja, generacje
    sols = [losujSciezke() for _ in range(wstepnaPopulacja)]    #wylosowanie startowej populacji

    sols.sort(key=ocena)               
    operatorSelekcji(sols)
    for i in range(generacje):
        operatorKrzyzowania(sols) 
        operatorSelekcji(sols)
    return sols[0]

#Ustawia nowe parametry algorytmu i generuje struktury pomocnicze (optymalizacyjne)
def kalibrujAlgorytm(newN, newPopulacja = populacja, newGeneracje = generacje):
    global n, populacja, generacje, indeksy, indeksy, stop
    n = newN                                                   #liczba wierzcholkow
    populacja = newPopulacja
    generacje = newGeneracje
    #sekcja optymalizacyjna
    indeksy = [i for i in range(n)] 
    kolejnosc = [i for i in range(populacja)]
    stop = populacja - (populacja%2)

#Algorytm wyczerpujacy dla naszego problemu
def wyczerpujacy(graf, skalary):
    bestSol = []
    bestSum = sum(max(graf))*max(skalary)
    for p in itertools.permutations([i for i in range(n)]):
        sol = list(p)
        sol.append(sol[0])
        Sum = ocena(sol)
        if bestSum > Sum:
            bestSol = sol
            bestSum = Sum
    return (bestSum, bestSol)

def zachlanny(graf):
    dowybrania = [i for i in range(1,n)]
    sol = [0]
    wybrane=0
    for i in range(0,n-1):
        best = dowybrania[0]
        for w in dowybrania:
            if cache[wybrane][sol[wybrane]][best]>cache[wybrane][sol[wybrane]][w]:
                best = w
        dowybrania.remove(best)
        sol.append(best)
        wybrane=wybrane+1
    sol.append(sol[0])
    return sol

def losowy(graf, czas):
    global czas
    start = time.time()
    best = losujSciezke()
    bestsum = ocena(best)
    while time.time()-start<czas:
        sciezka = losujSciezke()
        sum = ocena(sciezka)
        if sum<bestsum:
            best = sciezka
            bestsum = sum
    return sciezka

def testuj(algorytm):
    global maxSkalar, minSkalar, maxPos, n
    odchylenia = []
    nstart = time.time()
                                                                                           #generacja nowego grafu
    skalary = [(minSkalar + random.random() * (maxSkalar - minSkalar)) for _ in range(n)]
    #punkty = [(random.randint(0, maxPos), random.randint(0, maxPos)) for _ in range(n)]
    graf = generujGraf(n, skalary)
    wyniki = []
    odchylenia = []
    print("0")
    wyniki.append(ocena(algorytm(graf)))
    for i in range(1, 10):
        print(i)
        wyniki.append(ocena(algorytm(graf)))
        odchylenia.append(statistics.stdev(wyniki))

    while (statistics.stdev(odchylenia)/math.sqrt(len(odchylenia)) > statistics.fmean(odchylenia)/20):
        print(statistics.stdev(odchylenia)/math.sqrt(len(odchylenia))/statistics.fmean(odchylenia) * 100)
        wyniki.append(ocena(algorytm(graf)))
        odchylenia.append(statistics.stdev(wyniki))
                                                                                        
    print("Srednia ocena jakosi cyklu: ", statistics.fmean(wyniki)) #To trzeba zapisac i zwrocic z tej funkcji
    print("Odchylenie odchylen: ", statistics.stdev(odchylenia)/math.sqrt(len(odchylenia)))
    print("Srednie odchylenie: ", statistics.fmean(odchylenia))
    bladWOsiY = 15 * statistics.fmean(odchylenia)/4               

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
    
def rysujRozwiazanie(rozwiazanie, title = "", skalaWykresu = skalaWykresu):
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

dane = [10] + [50] + #[i for (i in range(100, 1000))] na razie bez tych wiekszych danych
for i in range(len(dane)-1, -1):    #zaczynam od konca, zeby nie sprawdzic czy algorytm trwa za dlugo
    start = time.time()
    jakiesdane = testuj(genetyczny)
    kalibrujAlgorytm(dane[i], 10, 50)
    jakiesdane = testuj(genetyczny)
    kalibrujAlgorytm(dane[i], 20, 20)
    jakiesdane = testuj(genetyczny)
    kalibrujAlgorytm(dane[i], 20, 50)
    jakiesdane = testuj(genetyczny)
    kalibrujAlgorytm(dane[i], 40, 20)
    jakiesdane = testuj(genetyczny)
    kalibrujAlgorytm(dane[i], 40, 40)
    jakiesdane = testuj(genetyczny)
    #w jakis danych powinien byc tez sredni czas algoytmu
    #wez srednia z tych czasow i zrob
    czas = sredniczas
    jakiesdane = testuj(losowy)
    #tak zeby losowy mial jakis sensowny czas wykonania
    jakiesdane = testuj(zachlanny)
    #zapisz jakiesdane, zeby potem mozna uzyc w wykresiee
    print(time.time() - start)
#Wykresy dla dan
