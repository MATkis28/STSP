naglowki = 1
nazwaIn = "15_02_21_00_05_25"
nazwaOut = "wynik1.tex"
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

plik = open(nazwaOut, "w", encoding="utf-8")
plik.write("\\begin{tabular}{")
if (space != 0):
    for j in range(kolumny):
        #plik.write("|>{\\centering\\arraybackslash}p{")
        plik.write("|p{")
        plik.write(str(space/kolumny))
        plik.write("\\linewidth} ")
    plik.write("|}\n\\hline\n")
else:
    for j in range(kolumny):
        #plik.write("|>{\\centering\\arraybackslash}p{")
        plik.write("|r")
    plik.write("|}\n\\hline\n")
    
for i in range(naglowki):
    for j in range(kolumny):
        if (j != 0):
            plik.write("\\multicolumn{1}{c|}{\\textbf{")
        else:
            plik.write("\\multicolumn{1}{|c|}{\\textbf{")
        plik.write(dane[i][j])
        if (j != kolumny - 1):
            plik.write("}} & ")
        else:
            plik.write("}}\\\\\\hline\n")

for i in range(proby):
    i += naglowki
    for p in range(kolumny):
        plik.write(dane[i][p])
        if (p != kolumny-1):
            plik.write(" & ")
    plik.write("\\\\\\hline\n")
plik.write("\\end{tabular}")
plik.close()
