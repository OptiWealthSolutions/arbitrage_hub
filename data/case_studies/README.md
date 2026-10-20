# Données des études de cas

Clôtures quotidiennes (prix « Last », séances régulières) téléchargées via l'API Interactive Brokers
le 20 octobre 2026 : 500 séances du 21 octobre 2024 au 19 octobre 2026, calendrier NYSE commun aux
quatre titres. Prix non ajustés des dividendes.

| Fichier | Titre | Contrat IBKR |
|---|---|---|
| GLD_close.txt | SPDR Gold Shares (ARCA) | 51529211 |
| GLDM_close.txt | SPDR Gold MiniShares (ARCA) | 543859629 |
| KO_close.txt | Coca-Cola (NYSE) | 8894 |
| PEP_close.txt | PepsiCo (NASDAQ) | 11017 |

Format : une ligne de 500 valeurs séparées par des virgules, dans l'ordre chronologique.
Chargement : `np.loadtxt(path, delimiter=",")`.
