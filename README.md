### FSWS18 Uebung 3 VIZ

This project is an edited version of the [ESIM](https://github.com/coetaur0/ESIM/) github repo to visualize exemplary entailment pairs.
Snippets of the original readme are referred to and left at the bottom of this readme.

### Setup

After cloning into this repository, refer to the **"Install the package"**, **"Fetch the data to train and test the model"** and **"Preprocess the data"** steps in the [ESIM](https://github.com/coetaur0/ESIM/) readme (using the fetch_data.py and preprocess_data.py from THIS repo. It is not necessary to download anything from the ESIM repo.) . Training is **not** necessary as we will be using a checkpoint.

### Uebung 3 Aufgabe 1
Formale und Computationelle Semantik WS2018

In dieser Aufgabe visualisierst du mithilfe von matplotlib oder einem anderen Visualisierungstool bzw. Modul deiner Wahl Attentionwerte von ESIM für beispielhafte Entailment-Paare.

Interessant ist für uns lediglich die Datei *ESIM/scripts/fsws18uebung03_viz.py*. Keine andere Datei sollte editiert werden
und es sollte auch nur diese Datei (samt eventuellen weiteren Dateien für Antworten zu anderen Teilaufgaben) abgegeben werden. 

Für die vollständige Bearbeitung dieser Teilaufgabe ist lediglich die Funktion
				
				viz_entailment_pair()
in fsws2018uebung03_viz.py zu implementieren. Sie wird von viz_input() aufgerufen, welche in der letzten Zeile von main() aufgerufen wird.In dieser Datei finden sich auch weitere Instruktionen zur Bearbeitung.
Für das Format deines Plots findest du im Aufgabenblatt ein unverbindliches Beispiel zur Orientierung.

Nach korrekter Implementation sollte dieses script über die Kommandozeile wie folgt aufgerufen werden können und einen Graphen plotten:
```
$ python3 fsws18uebung03_viz.py --prem 'A cat is on a mat .' --hyp 'An animal is outside .'
```
-------------
Dependencies:

python3.5+

torch

matplotlib.pyplot or other

wget

and many more. Refer to the Installation in the [ESIM](https://github.com/coetaur0/ESIM/) readme.

The below is an excerpt from the original readme and can be ignored for the exercise.

# ESIM - Enhanced Sequential Inference Model
Implementation of the ESIM model for natural language inference with PyTorch

This repository contains an implementation with PyTorch of the sequential model presented in the paper 
["Enhanced LSTM for Natural Language Inference"](https://arxiv.org/pdf/1609.06038.pdf) by Chen et al. in 2016.
