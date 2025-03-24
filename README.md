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

1. **Cloner le dépôt :**
    ```bash
    git clone https://github.com/ZouhairSA/Somnolence-detection-IOT.git
    cd Somnolence-detection-IOT