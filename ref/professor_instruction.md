1) Progettare un database per i dati.

2) Scaricare i dati da Yahoo Finance. 
Direi tutte le azioni dal NASDAQ. Frequenza giornaliera con i dati OCLHV (Open, Close, Low, High, Volume)
Direi di considerare i dati dall'inizio del titolo fino ad oggi.

3)  Usiamo come training set i dati fino al 2015 e come test set i dati dal 2015 ad oggi

4)  Definire inputs ed output per la rete neurale e salvalri nel database
Siano:
MC(x) il valore medio del valore di close negli ultimi x giorni escluso il giorno corrente;
SL(x) il valore dello slope negli ultimi x giorni escluso il giorno corrente;
O il valore di opening del giorno corrente;
C il valore di close del giorno corrente
Y = (C - O)/C

L'input della rete è:
MC(5)/O, MC(10)/O, MC(20)/O, MC(50)/O, MC(100)/O, MC(200)/O, 
SL(5)/O, SL(10)/O, SL(20)/O, SL(50)/O, SL(100)/O, SL(200)/O.

Il valore di ouptut della rete è Y (yield).

5) Definiano una rete neurale feed forward che predice lo yield giornaliero.
Scegliere arhicettura della rete.

6) FAre il training della rete su tutti i titoli fino al 2015.
Nota bene, usiamo la stessa rete per tutti i titoli.

7) Valutare la performance della rete su tutti i titoli dal 2015 ad oggi.
Valutare l'errore medio e la sua deviazione standard.

8) Ripetere i punti  5, 6, 7 per diverse architetture con  l'obiettivo di minimizzare il valore medio dell'errore.

9) Costruire la frontiera di Pareto (errore medio, standard deviation) per le architetture di cui sopra.
