import vamp
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from math import ceil

gamme_maj = [1, 0, 0, 1, 1, 0, "dim"] # 1 signifie un accord Majeur, un 0 un accord Mineur, "dim" un accord diminué (sera ignoré car pas un accord dans la base de données)

profile_maj = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88] # Selon l'algorithme de Krumhansl-Schmuckl, le profile d'une gamme Majeur
profile_min = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17] # Selon l'algorithme de Krumhansl-Schmuckl, le profile d'une gamme Mineur
def trouve_gamme(audio): # Fonction pour estimer la gamme d'une chanson
    """
    Cette fonction estime la gamme d'une chanson
    Entrée: liste NumPy contenant le fichier audio
    Sortie: liste contentant 2 estimations de gammes
    """
    assert librosa.get_duration(y=audio) > 0, "Le fichier peut être corrompu." # Pour empêcher bug qui se produit parfois lorsque un fichier musique a une longueur 0
    chromagram = librosa.feature.chroma_cqt(y=audio) # Calcule le Chroma CQT du fichier audio, soit un type de représentation de la musique qui permet de caractériser la tonalité et l'harmonie d'une pièce musicale.
    presence_notes = [] # Initialisation de la liste presence_notes, qui stocke la somme de la présence de chaque note musicale dans le fichier audio
    for i in chromagram:
        presence_notes.append(np.sum(i)) # Calcule la présence de chaque note, en prenant la somme de la présence des notes à travers le fichier
    correlation_maj, correlation_min = [], [] # Initialisation des listes correlation_maj et correlation_min, qui stockeront le coeffcient de correlation entre chaque gamme, et les profiles de Krumhansl-Schmuckl. Le coefficient le plus élevé va corresponde à la gamme à estimer.
    for i in range(12):
        gamme_test = [presence_notes[(i+j)%12] for j in range(12)]  # Crée une liste contenant chaque note (sous forme numérique, où 0 --> Do et 11 --> Si), mais à chaque itération, l'ordre est différent (ex: itération 1 gamme_test = [0, 1, ..., 10, 11], itération 5 gamme_test = [4, 5, ..., 2, 3])
        correlation_maj.append(np.corrcoef(profile_maj, gamme_test)[0][1]) # Detérmine la corrélation ce la gamme testée avec le profile d'une gamme majeur
        correlation_min.append(np.corrcoef(profile_min, gamme_test)[0][1]) # Detérmine la corrélation ce la gamme testée avec le profile d'une gamme mineur
    correlation = correlation_maj + correlation_min # Combination des 2 listes correlation_maj et correlation_min
    gamme = [] # Liste qui contient l'information sur les gammes estimées
    for i in range(2): # range(2) car on fait 2 estimations
        gamme_estime = correlation.index(max(correlation)) # La première gamme qu'on estime sous forme numérique (la note fondamentale a pour index gamme_estime%12 avec 0 --> Do et 11 --> Si), et si gamme_estime > 11, la gamme est mineur, sinon la gamme est majeur 
        gamme.append(gamme_estime) # Ajoute la gamme estimée à la liste gamme
        if gamme_estime > 11: # Si le mode de la gamme estimée est Mineur
            gamme_relatif = (gamme_estime + 3)%12 # Détermine la gamme relatif de la première gamme estimée
        else: # Si le mode de la gamme estimée est Majeur
            gamme_relatif = (gamme_estime + 9)%12+12 # Détermine la gamme relatif de la première gamme estimée

        gamme.append(gamme_relatif) # Ajoute la gamme relatif à la liste gamme
        gamme.append(correlation[gamme_estime]) # Ajoute le coefficient de correlation entre la gamme estimé et le profile majeur/mineur
        correlation[gamme_relatif] = -50 # Change la valeur de la correlation de la gamme estimée pour pouvoir faire une deuxième estimation
        correlation[gamme_estime] = -50 # Change la valeur de la correlation de la gamme relatif à celle estimée pour pouvoir faire une deuxième estimation
    return gamme # Renvoie la liste gamme

def LoadModel(k_voisins=1): # Fonction pour charger le model des K Plus Proches Voisins qui estimera les accords d'une chanson
    """
    Cette fonction prépare le model des K Plus Proches Voisins, qui va estimer les accords d'une chanson.
    Entrée facultatif: nombre entier correspondant aux nombres de voisins à utiliser pour estimers les accords
    Sortie: Le model des K Plus Proches Voisins, et sa précision
    """
    assert isinstance(k_voisins, int), "L'entrée k_voisins doit être un nombre entier"
    # Chargement des données des accords pour entrainer le model
    X = np.load("donnees/x_train.npy")
    Y = np.load("donnees/y_train.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) # Séparation des données. Une partie pour entrainer le model, l'autre pour le tester.
    knn = KNeighborsClassifier(n_neighbors=k_voisins) # Initialisation du model
    knn.fit(X_train, y_train) # Entrainement du model

    # Détermination de la précision du model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return knn, accuracy # Renvoie le model et sa précision

def trouve_accords(audio, sr, knn, diatonique=False, cle_maj=None):
    """
    Cette fonction estime les accords d'une chanson.
    Entrée:
    audio = liste NumPy contenant le fichier audio
    sr = taux d'échantillonnage du fichier audio
    knn = model des K Plus Proches Voisins à utiliser pour déterminer les accords
    diatonique = Si True, les accords estimés seront dans la gamme de cle_maj
    cle_maj = Mode majeur de la gamme à utiliser pour rendre les accords diatonique
    Sortie: Une liste de listes contenant les temps de chaque accord
    """
    assert isinstance(diatonique, bool), "Le paramètre diatonique doit être un booléen"
    assert isinstance(cle_maj, int) and 0 <= cle_maj < 12, "Le paramètre cle_maj doit être un nombre entier compris entre 0 et 11"
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr) # Renvoie le tempo de la chanson, ainsi qu'une liste stockant les temps des battements
    beats_time = librosa.frames_to_time(beats, sr=sr) # Stocke les temps de battements en seconde
    beats_samples = librosa.frames_to_samples(beats) # # Stocke les temps de battements en "sample", soit la plus petite unité, qui compose la chanson
    chromas = [] # Initialisation de la liste qui va stocker les "chromas" de chaque morceau découpé de la chanson. Un chroma est similiaire à ce qu'on obtient avec librosa.feature.chroma_cqt.
    temps_accord = [] # Initialisation de la liste qui stockera les temps des morceaux, et les accords qui jouent pendant ces morceaux
    hop = ceil(tempo/60) # Détermine la longueur des "hops", pour découper la chanson afin d'avoir des morceaux de environ 1.5 secondes
    for i in range(0, len(beats)-hop, hop): # Pour chaque morceau coupé
        y = audio[beats_samples[i]:beats_samples[i+hop]+1] # Prend la partie coupée de l'audio
        temps_accord.append([float(beats_time[i]), float(beats_time[i+hop])]) # Ajoute les temps correspondant au début et à la fin de cette partie à la liste temps_accord
        chroma = vamp.collect(y, sr, "nnls-chroma:nnls-chroma", output="bothchroma", parameters={"rollon":0.01})["matrix"][1] # Obtient le chromagram de cette partie de la chanson
        average_chroma = np.mean(chroma, axis=0) # Détermine la moyenne de la présence de chaque note d'après le chroma
        chromas.append(average_chroma) # Ajoute le chroma des la présence moyenne des notes à la liste chromas
    index_accords = knn.predict(chromas) # Pour chaque morceaux, détermine l'accord qui joue
    for i in range(len(index_accords)): 
        temps_accord[i].append(int(index_accords[i])) # Ajoute les accords à leurs temps correspondant dans la liste temps_accord
    
    if diatonique and cle_maj != None: # Si l'option diationique est vrai, est si une gamme majeur à été donnée
        index_base = cle_maj # Stocke l'indice de la note fondamentale de la gamme
        notes_gamme = [index_base, (index_base + 2)%12, (index_base + 4)%12, (index_base + 5)%12,
                     (index_base + 7)%12, (index_base + 9)%12, (index_base + 11)%12] # Initialisation de la liste stockant les notes de la gammes
        for i in range(len(temps_accord)):
            if i < len(temps_accord):
                if (temps_accord[i][2]%12 not in notes_gamme) or temps_accord[i][2] == 24 or (gamme_maj[notes_gamme.index(temps_accord[i][2]%12)] == "dim"): # Si la note fondamentale de l'accord ne fait pas partie de la gamme, où si dans la gamme, l'accord associé à cette note fondamentale est "diminué" (très rare en musique), ou si le model KNN n'a pas détécter d'accord dans le morceau (si temps_accord[i][2] == 24)
                    try:
                        temps_accord[i][2] = temps_accord[i-1][2] # Fusionne cette accord avec l'accord précédent, qui est forcément un accord de la gamme
                    except: # Si cet accord non-diatonique est au début de la liste
                        temps_accord.remove(temps_accord[i]) # On supprime cet accord de la liste
                elif (temps_accord[i][2] > 11 and gamme_maj[notes_gamme.index(temps_accord[i][2]%12)] == 1): # Si l'accord est mineur, mais dans la gamme il devrait être majeur
                    temps_accord[i][2] -= 12 # Transforme l'accord mineur en accord majeur
                elif (temps_accord[i][2] < 12 and gamme_maj[notes_gamme.index(temps_accord[i][2]%12)] == 0): # Si l'accord est majeur, mais dans la gamme il devrait être mineur
                    temps_accord[i][2] += 12 # Transforme l'accord majeur en accord mineur
    return temps_accord # Renvoie la liste stockant tous les temps des accords

def fetch_tempo(audio, sr): # Fonction qui renvoie le tempo de la chanson
    """
    Cette fonction renvoie le tempo d'une chanson
    Entrée:
    audio = liste NumPy contenant le fichier audio
    sr = taux d'échantillonnage du fichier audio
    Sortie: Tempo de la chanson
    """
    return librosa.beat.beat_track(y=audio, sr=sr)[0] # Renvoie le tempo de la chanson