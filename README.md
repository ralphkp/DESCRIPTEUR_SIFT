Détecteur de points d'intérêt SIFT et descripteur SIFT
# Détecteur de points d'intérêt SIFT et descripteur SIFT




## Demarrage


### Cloner le project

Tout d'abord, vous devez cloner le projet sur votre ordinateur avec :

```
https://github.com/ralphkp/DESCRIPTEUR_SIFT.git
```

Vous pouvez maintenant vous déplacer dans le dossier nouvellement créé :
```
cd DESCRIPTEUR_SIFT
```

### Create a virtual environment


1. Create a new virtual environment
```
python3 -m venv venv
```

2. Activate the virtual environment
```
source env/bin/activate
```

Assurez-vous que l'environnement virtuel est activé chaque fois que vous voulez lancer le projet.

### Install all requirements

Les dépendances du projet sont stockées dans le fichier requirements.txt. Vous pouvez les installer avec la commande suivante :

**WARNING** :  Vérifiez que votre environnement virtuel est activé, sinon vous installerez les paquets à l'échelle du système.
```
pip install -r requirements.txt
```


### Start Project


```
python3 Detecteur_SIFT.py
```

**Note:** le projet utiliser Python3.


Vous pouvez maintenant visiter le rep "data" pour voir les resultats.

**WARNING** : nous avons 2 repertoires 'Objets et ImagesRecherche', une seule image à été choisie dans le repertoire Objets pour chercher sa correspondance dans l'ensemble de contenu de l'autre repertoire.

![Uploading matches_thaprua2.jpg…]()


![matches_20221206_003418_small](https://github.com/ralphkp/DESCRIPTEUR_SIFT/assets/83407803/ffa1bf95-518b-42a2-bba5-fb9702211b0d)

