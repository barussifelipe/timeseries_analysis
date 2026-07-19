- Daily Data 
- NASDAQ and NHSE
- Timeseries per tutti i titoli di questi. Database 
1. Rede Neurale che guarda lo storico. Parte alla matina e voglio prevedere il valore del close. 
2. LSTM, decidere la architectura della storia per x numeri di giorni. 
- Explore nel tirocinio una struttura mista, valori brutto per pochi giorni e per varie giorni valori come slope. 
- Opening, High, Low, Close, Adj. Close, Volume. 
- Prendiamo il rapporto. Open(oggi)/Close(ieri) oppure Open(oggi)/Close(oggi). Close normale. 
- Se io sono al inizio della giornata, il mio obiettivo è conoscere il rapporto tra oggi. Se io sono al fine giornata, opening(domani)/close(oggi). 
- LSTM con finestre diverse. Prevedere tempo t+1. 
- 10 anni per training. 10 anni per testing. 
- Ogni row a ticket. 


