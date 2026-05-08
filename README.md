# Motion Tracker — Fastest Object Tracking

Questo progetto è un sistema di visione artificiale in tempo reale progettato per identificare tutti gli oggetti in movimento in una scena, focalizzando il tracciamento grafico esclusivamente sull'oggetto più veloce.

## Panoramica
A differenza dei comuni sistemi di rilevamento, questo software calcola dinamicamente la velocità di ogni corpo rilevato. Ignora il rumore di fondo e i movimenti lenti per evidenziare l'azione principale, rendendolo ideale per analisi balistiche, monitoraggio sportivo o esperimenti di fisica.

## Caratteristiche principali
* **Tracking Selettivo:** Rileva più oggetti simultaneamente ma traccia la traiettoria solo del target con la velocità maggiore.
* **Scia Cromatica (Trail):** Una scia fluida con gradiente di colore mostra la traiettoria recente, facilitando l'analisi del percorso.
* **Calcolo Velocità:** Visualizza la velocità istantanea in px/f (pixel per frame) direttamente sopra l'oggetto attivo.
* **HUD Interattivo:** Pannello di controllo in sovrimpressione per monitorare FPS, numero di oggetti e sensibilità del sensore.
* **Filtrazione Avanzata:** Utilizza la sottrazione del background (MOG2) e filtri morfologici per garantire un segnale pulito anche con webcam standard.

## Analisi del video
Nel filmato dimostrativo, il sistema aggancia la parabola di una pallina lanciata. Mentre gli altri piccoli movimenti vengono ignorati o semplicemente segnalati, l'oggetto più veloce viene racchiuso in un riquadro di selezione attivo e la sua scia viene disegnata in tempo reale senza latenza percepibile.

## Controlli da tastiera
| Tasto | Azione |
| :--- | :--- |
| SPAZIO | Mette in pausa o riprende l'analisi video |
| + / - | Aumenta o diminuisce la sensibilità del sensore (varThreshold) |
| C | Cancella la traiettoria corrente dal buffer |
| S | Salva uno screenshot del frame corrente con la scia inclusa |
| Q | Chiude l'applicazione e rilascia la webcam |

## Requisiti:
 1.opencv-python 
 2.numpy
