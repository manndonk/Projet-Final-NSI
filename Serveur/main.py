from flask import Flask, render_template, jsonify, request, send_file
from music import *
from pytube import Search, YouTube
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import glob

# Efface tous les fichiers audio
for f in glob.glob("chanson_passe/*"):
    os.remove(f)
for f in glob.glob("*.mp3"):
    os.remove(f)
for f in glob.glob("*.mp4"):
    os.remove(f)

chanson_passe = [] # Liste stockant les chansons passées
app = Flask(__name__) # Initialisation du serveur
CORS(app) # Ligne de code nécessaire pour empêcher l'erreur de Cross-Origin
knn, precision = LoadModel() # Stocke le model dans la variable knn
print(f"Le model a une précision de {precision*100}%") # Énonce la précision du model. Envirion 90%.

limiter = Limiter(get_remote_address, app=app) # Initiation du limiteur qui permet de limiter le nombre de requête fait par un utilisateur (car parfois la fonction fetch en JavaScript fait 2 requêtes)
i = -1 # initiation de la variable i, qui permettra de donner un nom unique au 5 dernières chanson lorsqu'elle sont installées (si 5 requêtes sont envoyées rapidement, la commande ffmpeg n'aurait donc pas de bug)
@app.route("/upload", methods=["GET"]) # La fonction analyse_chanson est éxecutée lorsque une requête est fait au enpoint "/upload"
@limiter.limit("1/minute") # Définit la limite de une requête par minute
def analyse_chanson():
    global i
    # On obtient le fichier audio
    artiste = request.args.get("artiste") # Stocke le nom de l'artiste dans la variable artiste
    titre = request.args.get("titre") # Stocke le nom de la chanson dans la variable titre
    search = Search(f"{titre} {artiste}") # Cherche la chanson sur YouTube
    audio = YouTube(search.results[0].watch_url, use_oauth=True).streams.filter(only_audio=True).first() # Séléctionne la première vidéo qui apparait
    i = (i + 1)%5 # i peut prendre les valeurs 0, 1, 2, 3, ou 4
    audio.download(filename=f"{i}.mp4") # Installation du fichier audio

    command = f"ffmpeg -loglevel quiet -i {i}.mp4 {i}.mp3" # Commande pour changer le fichier audio pour pouvoir le traiter
    os.system(command) # Exécution de la commande

    os.remove(f"{i}.mp4") # Supprime le fichier chanson.mp4 qui n'est plus utile
    chanson_passe.append(f"{artiste} {titre}.mp3") # Ajoute le fichier à liste stockant les chanson passées
    if len(chanson_passe) > 5: # Le maximum qu'on peut stocke dans cette liste est 5
        fichier_exces = chanson_passe.pop(0)
        if os.path.isfile(f"chanson_passe/{fichier_exces}"):
            os.remove("chanson_passe/" + fichier_exces) # Lorsque la limite de 5 chansons est dépassée, on supprime la plus ancienne

    info = {} # Initialisation du dictionnaire qui va stocker les informations à transmettre à l'utilisateur
    y, sr = librosa.load(f"{i}.mp3") # Chargement de la chanson
    os.rename(f"{i}.mp3", f"chanson_passe/{artiste} {titre}.mp3") # On change le nom du fichier, puis on le déplace dans le dossier chanson_passe, qui stocke les fichiers des 5 dernières chansons
    y_harmonique = librosa.effects.harmonic(y) # On extrait la partie "harmonique" de la chanson, plus intéressante pour cette application
    gamme = trouve_gamme(y_harmonique) # Détermine la clé de la chanson
    # Trouve les notes de la gamme majeur ayant les notes de la première estimation
    gamme_maj = gamme[0]
    if gamme_maj > 11:
        gamme_maj = (gamme_maj + 3)%12
    temps_accords = trouve_accords(y_harmonique, sr, knn, diatonique=True, cle_maj=gamme_maj) # Détermine les accords de la chanson
    info["cle"] = gamme # Met l'information sur la gamme de la chanson dans le dictionnaire info
    info["tempo"] = fetch_tempo(y, sr) # Met l'information sur le tempo de la chanson dans le dictionnaire info
    info["temps_accord"] = temps_accords # Met l'information sur les accords de la chanson dans le dictionnaire info
    return jsonify(info) # Renvoie les informations de la chanson à l'utilisateur

@app.route("/chanson") # La fonction telecharge_chanson est éxecutée lorsque une requête est fait au endpoint "/chanson"
def telecharge_chanson():
    artiste = request.args.get("artiste") # Stocke le nom de l'artiste dans la variable artiste
    titre = request.args.get("titre") # Stocke le nom de la chanson dans la variable titre
    if os.path.isfile(f"chanson_passe/{artiste} {titre}.mp3"): # Si le fichier existe
        return send_file(f"chanson_passe/{artiste} {titre}.mp3", as_attachment=True) # Renvoie le fichier de la chanson à l'utilisateur
    return "Ce fichier n'existe pas" # Si le fichier n'a pas été trouvé, on renvoie ce message

app.run(host='0.0.0.0', port=80) # Commence le serveur