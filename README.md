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
git clone https://github.com/ZouhairSA/Somnolence-detection-IOT.git
cd Somnolence-detection-IOT
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

### Système de Détection Amélioré
1. **Détection des Yeux (Système Avancé)**
   - Analyse du ratio d'ouverture des yeux (EAR - Eye Aspect Ratio)
   - Détection des micro-sommeils (fermeture rapide des yeux)
   - Suivi de la fréquence des clignements
   - Seuils personnalisables pour la sensibilité
   - Compensation de la luminosité ambiante
   - Filtrage des faux positifs

2. **Détection des Bâillements (Système Avancé)**
   - Analyse du ratio d'ouverture de la bouche (MAR - Mouth Aspect Ratio)
   - Détection de la durée des bâillements
   - Suivi de la fréquence des bâillements
   - Distinction entre bâillements et parole
   - Seuils adaptatifs selon l'heure de la journée

3. **Système de Scoring de Fatigue**
   - Score de fatigue en temps réel (0-100)
   - Combinaison de multiples facteurs :
     - Durée des yeux fermés
     - Fréquence des bâillements
     - Mouvements de la tête
     - Temps de réaction
   - Historique des scores sur la dernière heure
   - Prédiction de la fatigue à venir

### Interface Utilisateur Améliorée
1. **Design Moderne et Intuitif**
   - Thème sombre/clair personnalisable
   - Interface responsive et adaptative
   - Animations fluides pour les transitions
   - Icônes intuitives et tooltips informatifs

2. **Tableau de Bord Principal**
   - Vue en direct de la webcam avec overlay des détections
   - Graphique de score de fatigue en temps réel
   - Indicateurs visuels pour :
     - État des yeux (vert/rouge)
     - Niveau de bâillements
     - Score de fatigue global
   - Boutons de contrôle rapide

3. **Panneau de Statistiques**
   - Graphiques historiques de :
     - Fréquence des clignements
     - Nombre de bâillements
     - Score de fatigue
   - Export des données au format CSV
   - Filtres temporels (heure/jour/semaine)

4. **Paramètres Avancés**
   - Calibration de la caméra
   - Ajustement des seuils de détection
   - Configuration des alertes
   - Personnalisation des sons d'alerte
   - Sauvegarde des préférences

5. **Système d'Alertes Intégré**
   - Alertes visuelles personnalisables
   - Sons d'alerte progressifs
   - Notifications système
   - Historique des alertes
   - Mode silencieux disponible

### Communication avec Arduino
1. **Configuration du Port Série**
   - Détection automatique du port COM
   - Configuration de la vitesse de communication (baud rate)
   - Gestion des erreurs de connexion

2. **Protocole de Communication**
   - Envoi de commandes en temps réel
   - Format des messages : "EYE_CLOSED", "YAWN_DETECTED", "ALERT"
   - Gestion des délais et des timeouts

3. **Intégration du Buzzer**
   - Contrôle du buzzer via Arduino
   - Différents types d'alertes sonores
   - Configuration de la fréquence et de la durée des alertes

4. **Sécurité et Robustesse**
   - Vérification de la connexion Arduino
   - Gestion des déconnexions
   - Logs de communication

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
    ```