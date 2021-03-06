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
    newsols.sort(key=ocena)
    return newsols[:populacja]

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
    sols = [losujSciezke() for _ in range(wstepnaPopulacja)]
    sols.append(zachlanny(graf))

    sols = operatorSelekcji(sols)
    for i in range(generacje):
        operatorKrzyzowania(sols)
        sols = operatorSelekcji(sols)
    return sols[0]

#Ustawia nowe parametry algorytmu i generuje struktury pomocnicze (optymalizacyjne)
def kalibrujAlgorytm(newN, newPopulacja = populacja, newGeneracje = generacje):
    global n, populacja, generacje, indeksy, kolejnosc, wstepnaPopulacja, stop
    n = newN                                                   #liczba wierzcholkow
    populacja = newPopulacja
    generacje = newGeneracje
    wstepnaPopulacja = populacja*4
    #sekcja optymalizacyjna
    indeksy = [i for i in range(n)] 
    kolejnosc = [i for i in range(populacja)]
    stop = populacja - (populacja%2)

#Algorytm wyczerpujacy dla naszego problemu
def wyczerpujacy(graf, skalary):
    bestSol = []
    bestSum = n * max(max(graf))*max(skalary)
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

def losowy(graf):
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

def testuj(algorytm, graf):
    global maxSkalar, minSkalar, maxPos, n
    wyniki = []
    czasy = []
    odchyleniaWynik = []
    odchyleniaCzas = []
    #print("0")
    start = time.time()
    wyniki.append(ocena(algorytm(graf)))
    czasy.append(time.time() - start)
    for i in range(1, 10):
    #    print(i)
        start = time.time()
        wyniki.append(ocena(algorytm(graf)))
        czasy.append(time.time() - start)
        odchyleniaWynik.append(statistics.stdev(wyniki))
        odchyleniaCzas.append(statistics.stdev(czasy))

    if (statistics.fmean(odchyleniaWynik) != 0):
        while (statistics.stdev(odchyleniaWynik) / math.sqrt(len(odchyleniaWynik)) > statistics.fmean(
                odchyleniaWynik) / 20 or
               (statistics.fmean(odchyleniaCzas) != 0 and statistics.stdev(odchyleniaCzas) / math.sqrt(
                   len(odchyleniaCzas)) > statistics.fmean(odchyleniaCzas) / 20)):
            #print("ooj: ", statistics.stdev(odchyleniaWynik) / math.sqrt(len(odchyleniaWynik)) / statistics.fmean(
            #    odchyleniaWynik) * 100)
            #if (statistics.fmean(odchyleniaCzas) != 0):
            #    print("oocz: ", statistics.stdev(odchyleniaCzas) / math.sqrt(len(odchyleniaCzas)) / statistics.fmean(
            #        odchyleniaCzas) * 100)
            start = time.time()
            wyniki.append(ocena(algorytm(graf)))
            czasy.append(time.time() - start)
            odchyleniaWynik.append(statistics.stdev(wyniki))
            odchyleniaCzas.append(statistics.stdev(czasy))
        
    #print("Srednia ocena jakosci cyklu: ", statistics.fmean(wyniki))
    #print("Odchylenie odchylen jakosci: ", statistics.stdev(odchyleniaWynik)/math.sqrt(len(odchyleniaWynik)))
    #print("Srednie odchylenie jakosci: ", statistics.fmean(odchyleniaWynik))
    #print("Srednia ocena czasu cyklu: ", statistics.fmean(czasy))
    #print("Odchylenie odchylen czasu: ", statistics.stdev(odchyleniaCzas)/math.sqrt(len(odchyleniaCzas)))
    #print("Srednie odchylenie czasu: ", statistics.fmean(odchyleniaCzas))
    bladWOsiYJakosc = 15 * statistics.fmean(odchyleniaWynik)/4
    bladWOsiYCzas = 15 * statistics.fmean(odchyleniaCzas)/4
    return (statistics.fmean(wyniki), statistics.fmean(czasy), bladWOsiYJakosc, bladWOsiYCzas)

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

def appendfile(filename,tekst):
    f = open(filename,"at")
    f.write(tekst)
    f.close

pre_file_name =time.strftime("%d_%m_%y_%H_%M_%S")
naglowki = ["n","czas1020","czas1050","czas2020","czas2050","czas4020","czas4040","czassredni","czaslosowy","czaszachlanny",
            "jak1020","jak1050","jak2020","jak2050","jak4020","jak4040","jaklosowy","jakzachlanny",
            "bczas1020","bczas1050","bczas2020","bczas2050","bczas4020","bczas4040","bczaslosowy","bczaszachlanny",
            "bjak1020","bjak1050","bjak2020","bjak2050","bjak4020","bjak4040","bjaklosowy","bjakzachlanny"]

appendfile(pre_file_name,' '.join(naglowki)+"\n")
dane = [1,50]+[i for i in range(100,600,100)]
#dane = [i for i in range(11)]
#dane = [i for i in range(3, 11)] #na razie bez tych wiekszych danych
tablicaJakosci = [[] for _ in range(len(dane))]
tablicaCzasu = [[] for _ in range(len(dane))]
tablicaBleduJakosci = [[] for _ in range(len(dane))]
tablicaBleduCzasu = [[] for _ in range(len(dane))]


for i in range(len(dane)-1, -1,-1):    #zaczynam od konca, zeby nie sprawdzic czy algorytm trwa za dlugo
    n = dane[i]
                                                                                               #generacja nowego grafu
    skalary = [(minSkalar + random.random() * (maxSkalar - minSkalar)) for _ in range(n)]
    #punkty = [(random.randint(0, maxPos), random.randint(0, maxPos)) for _ in range(n)]
    graf = generujGraf(n, skalary)
    start = time.time()


    #t = time.time()
    #(sum,a)=wyczerpujacy(graf,skalary)
    #tk = time.time()
    #print(n," ",sum," ",tk-t)

    kalibrujAlgorytm(n, 10, 20)
    (jakosc, czas, bladWOsiYJakosc, bladWOsiYCzas) = testuj(genetyczny, graf)
    tablicaJakosci[i].append(jakosc)
    tablicaCzasu[i].append(czas)
    tablicaBleduJakosci[i].append(bladWOsiYJakosc)
    tablicaBleduCzasu[i].append(bladWOsiYCzas)
    kalibrujAlgorytm(n, 10, 50)
    (jakosc, czas, bladWOsiYJakosc, bladWOsiYCzas) =  testuj(genetyczny, graf)
    tablicaJakosci[i].append(jakosc)
    tablicaCzasu[i].append(czas)
    tablicaBleduJakosci[i].append(bladWOsiYJakosc)
    tablicaBleduCzasu[i].append(bladWOsiYCzas)
    kalibrujAlgorytm(n, 20, 20)
    (jakosc, czas, bladWOsiYJakosc, bladWOsiYCzas) =  testuj(genetyczny, graf)
    tablicaJakosci[i].append(jakosc)
    tablicaCzasu[i].append(czas)
    tablicaBleduJakosci[i].append(bladWOsiYJakosc)
    tablicaBleduCzasu[i].append(bladWOsiYCzas)
    kalibrujAlgorytm(n, 20, 50)
    (jakosc, czas, bladWOsiYJakosc, bladWOsiYCzas) =  testuj(genetyczny, graf)
    tablicaJakosci[i].append(jakosc)
    tablicaCzasu[i].append(czas)
    tablicaBleduJakosci[i].append(bladWOsiYJakosc)
    tablicaBleduCzasu[i].append(bladWOsiYCzas)
    kalibrujAlgorytm(n, 40, 20)
    (jakosc, czas, bladWOsiYJakosc, bladWOsiYCzas) =  testuj(genetyczny, graf)
    tablicaJakosci[i].append(jakosc)
    tablicaCzasu[i].append(czas)
    tablicaBleduJakosci[i].append(bladWOsiYJakosc)
    tablicaBleduCzasu[i].append(bladWOsiYCzas)
    kalibrujAlgorytm(n, 40, 40)
    (jakosc, czas, bladWOsiYJakosc, bladWOsiYCzas) = testuj(genetyczny, graf)
    tablicaJakosci[i].append(jakosc)
    tablicaCzasu[i].append(czas)
    tablicaBleduJakosci[i].append(bladWOsiYJakosc)
    tablicaBleduCzasu[i].append(bladWOsiYCzas)

    #czas=sum(tablicaCzasu[i])/len(tablicaCzasu[i])
    tablicaCzasu[i].append(czas)

    (jakosc, czas, bladWOsiYJakosc, bladWOsiYCzas) = testuj(losowy, graf)
    tablicaJakosci[i].append(jakosc)
    tablicaCzasu[i].append(czas)
    czas1=czas
    tablicaBleduJakosci[i].append(bladWOsiYJakosc)
    tablicaBleduCzasu[i].append(bladWOsiYCzas)
    #tak zeby losowy mial jakis sensowny czas wykonania
    (jakosc, czas, bladWOsiYJakosc, bladWOsiYCzas) = testuj(zachlanny, graf)
    tablicaJakosci[i].append(jakosc)
    tablicaCzasu[i].append(czas)
    tablicaBleduJakosci[i].append(bladWOsiYJakosc)
    tablicaBleduCzasu[i].append(bladWOsiYCzas)
    appendfile(pre_file_name,str(n)+" "+' '.join([str(n) for n in tablicaCzasu[i]])+" "+' '.join([str(n) for n in tablicaJakosci[i]])+" "+' '.join([str(n) for n in tablicaBleduCzasu[i]])+" "+' '.join([str(n) for n in tablicaBleduJakosci[i]])+"\n")
    #zapisz dane, zeby potem mozna uzyc w wykresie
    #Bledy zapisz po prostu w tablicy i przekaz tablice potem do yerr
    print(time.time() - start)
#Zapisz wygenerowane dane!!!
#Szkoda je stracic, jakbysmy musieli cos poprawic w wykresach na szybko
#Wykresy dla
#Jakos od czasu
#Pamietaj o bledach w osi x = 0, w osy y = bladWOsiYJakosc lub bladWOsiYCzas
