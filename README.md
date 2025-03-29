# SafeSentry - Système de Détection de Somnolence avec Raspberry Pi et Arduino (Projet IoT)

## Aperçu

**SafeSentry** est un système de détection de somnolence conçu pour améliorer la sécurité routière en surveillant l'état d'alerte d'un conducteur en temps réel. Ce projet, réalisé dans le cadre de la matière **Internet des Objets (IoT)**, utilise un Raspberry Pi pour le traitement des données et un Arduino avec un buzzer pour émettre des alertes sonores. Le système détecte les signes de somnolence (comme la fermeture des yeux ou les bâillements) grâce à l'analyse des caractéristiques faciales et émet une alerte pour éviter les accidents potentiels.

Ce projet combine des techniques de vision par ordinateur et d'apprentissage automatique pour analyser les flux vidéo provenant d'une webcam, offrant une solution économique et fiable pour la détection de somnolence en temps réel.

## Sujet du Projet

L'objectif principal de ce projet IoT est de **détecter la somnolence et alerter le conducteur en temps réel**, garantissant ainsi des conditions de conduite plus sûres. Le système intègre un Raspberry Pi pour le traitement vidéo et un Arduino pour contrôler un buzzer qui émet des alertes sonores, offrant une solution efficace et abordable pour prévenir les accidents causés par la fatigue du conducteur.

## Fonctionnalités

- **Détection en temps réel** : Surveille et détecte la somnolence en temps réel à l'aide d'une webcam connectée à un Raspberry Pi.
- **Approche à double modèle** : Utilise deux modèles YOLOv8 — un pour la détection de l'état des yeux (ouverts/fermés) et un autre pour la détection des bâillements.
- **Points de repère faciaux** : Analyse la fermeture des yeux et la fréquence des bâillements à l'aide de MediaPipe pour une reconnaissance faciale précise.
- **Alertes sonores via Arduino** : Active un buzzer via un Arduino lorsqu'une somnolence est détectée.
- **Interface utilisateur** : Une interface conviviale construite avec PyQt5 pour visualiser les résultats de détection et les statistiques.
- **Enregistrement des données** : Capture et enregistre les données de détection pour une analyse ultérieure.
- **Seuils personnalisables** : Permet d'ajuster les seuils de détection pour une sensibilité adaptée.

### Fichiers Clés

- **`DrowsinessDetector.py`** : Logique principale de détection, incluant les points de repère faciaux, le système d'alerte et l'intégration avec Arduino.
- **`AutoLabelling.py`** : Script pour étiqueter automatiquement les données pour l'entraînement.
- **`CaptureData.py`** : Capture les données vidéo pour la détection de somnolence.
- **`LoadData.ipynb`** : Notebook pour charger et prétraiter les données.
- **`RedirectData.ipynb`** : Redirige et gère les données capturées.
- **`train.ipynb`** : Notebook Jupyter pour entraîner le modèle de détection.
- **`arduino_buzzer.ino`** : Code Arduino pour contrôler le buzzer en fonction des commandes du Raspberry Pi.

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/votre-username/Real_time_drowsy_driving_detection.git
cd Real_time_drowsy_driving_detection
```

2. Créer un environnement virtuel et l'activer :
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
venv\Scripts\activate     # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Fonctionnement du Programme

Le programme de détection de somnolence fonctionne en temps réel avec les caractéristiques suivantes :

### Détection en Temps Réel
- Capture vidéo continue via la webcam
- Traitement des images à 640x640 pixels
- Performance optimisée :
  - Prétraitement : ~7-8ms
  - Inférence : ~190-200ms
  - Post-traitement : ~1-2ms

### Fonctionnalités Principales
1. **Détection des Yeux**
   - État des yeux (ouvert/fermé)
   - Comptage des clignements
   - Détection des micro-sommeils

2. **Détection des Bâillements**
   - Identification des bâillements
   - Comptage des bâillements
   - Alertes sonores pour les bâillements prolongés

3. **Interface Utilisateur**
   - Affichage vidéo en direct
   - Barre de progression de la fatigue
   - Indicateurs d'état en temps réel
   - Statistiques de vigilance
   - Boutons de contrôle (Démarrer/Arrêter/Réinitialiser)

4. **Alertes et Notifications**
   - Alertes visuelles pour la somnolence
   - Alertes sonores pour les bâillements
   - Indicateurs de niveau de fatigue
   - Historique des détections

### Utilisation
1. Lancer le programme :
```bash
python DrowsinessDetector.py
```

2. Positionner votre visage devant la caméra
3. Le programme détectera automatiquement :
   - Les yeux ouverts/fermés
   - Les bâillements
   - Les signes de fatigue

4. Les alertes se déclencheront en cas de :
   - Yeux fermés prolongés
   - Bâillements fréquents
   - Niveau de fatigue élevé

5. Les statistiques sont sauvegardées dans "vigilance_stats.txt" à la fermeture

## Structure du Projet

1. **Cloner le dépôt :**
    ```bash
    git clone https://github.com/ZouhairSA/Somnolence-detection-IOT.git
    cd Somnolence-detection-IOT