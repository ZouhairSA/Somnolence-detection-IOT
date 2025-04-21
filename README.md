# Système de Détection de Somnolence en Temps Réel

Ce projet est un système avancé de détection de somnolence en temps réel, utilisant la vision par ordinateur et l'apprentissage automatique pour surveiller l'état de vigilance d'un conducteur.

## Fonctionnalités Principales

- Détection en temps réel des clignements d'yeux
- Détection des bâillements
- Identification des micro-sommeils
- Calcul du niveau de fatigue
- Stockage des données dans Cassandra
- Interface utilisateur moderne et intuitive

## Prérequis

- Python 3.8+
- OpenCV
- MediaPipe
- PyQt5
- Docker (pour Cassandra)
- Docker Compose

## Installation

1. Cloner le dépôt :
```bash
git clone [URL_DU_REPO]
cd Real_time_drowsy_driving_detection
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Configuration de Cassandra avec Docker :

a. Créer un fichier `docker-compose.yml` :
```yaml
version: '3'

services:
  cassandra:
    image: cassandra:latest
    ports:
      - "9042:9042"
    environment:
      - CASSANDRA_CLUSTER_NAME=vigilance_cluster
      - CASSANDRA_DC=dc1
      - CASSANDRA_RACK=rack1
    volumes:
      - cassandra_data:/var/lib/cassandra

volumes:
  cassandra_data:
```

b. Lancer Cassandra :
```bash
docker-compose up -d
```

c. Attendre que Cassandra soit prêt (environ 1-2 minutes) :
```bash
docker-compose logs -f cassandra
```

d. Créer le keyspace et les tables :
```bash
docker exec -it real_time_drowsy_driving_detection-cassandra-1 cqlsh -e "
CREATE KEYSPACE IF NOT EXISTS vigilance_db 
WITH REPLICATION = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE vigilance_db;

CREATE TABLE IF NOT EXISTS fatigue_events (
    event_id UUID PRIMARY KEY,
    timestamp TIMESTAMP,
    event_type TEXT,
    confidence FLOAT,
    details TEXT,
    fatigue_level INT,
    device_id TEXT,
    session_id TEXT
);

CREATE TABLE IF NOT EXISTS session_stats (
    session_id TEXT PRIMARY KEY,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_blinks INT,
    total_yawns INT,
    total_microsleeps INT,
    max_fatigue_level INT,
    avg_fatigue_level FLOAT,
    device_id TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id UUID PRIMARY KEY,
    timestamp TIMESTAMP,
    alert_type TEXT,
    severity TEXT,
    message TEXT,
    device_id TEXT,
    session_id TEXT
);"
```

## Structure du Projet

```
Real_time_drowsy_driving_detection/
├── DrowsinessDetector.py    # Code principal
├── cassandra_manager.py     # Gestion de la base de données
├── docker-compose.yml       # Configuration Docker
├── requirements.txt         # Dépendances
└── README.md               # Documentation
```

## Utilisation

1. Lancer l'application :
```bash
python DrowsinessDetector.py
```

2. L'interface s'ouvre avec :
- Affichage vidéo en direct
- Barre de niveau de fatigue
- Statistiques en temps réel
- Alertes visuelles

## Fonctionnalités Avancées

### Détection des Yeux
- Utilisation de MediaPipe pour la détection des points d'intérêt
- Calcul du ratio d'aspect des yeux (EAR)
- Détection des clignements et micro-sommeils

### Détection des Bâillements
- Analyse du ratio d'aspect de la bouche (MAR)
- Détection des bâillements prolongés

### Stockage des Données
- Enregistrement des événements de fatigue dans Cassandra
- Suivi des statistiques de session
- Historique des alertes

## Configuration

### Paramètres de Détection
- `EYE_AR_THRESH`: Seuil pour la détection des yeux fermés
- `EYE_AR_CONSEC_FRAMES`: Nombre de frames pour confirmer un clignement
- `YAWN_THRESH`: Seuil pour la détection des bâillements
- `YAWN_CONSEC_FRAMES`: Nombre de frames pour confirmer un bâillement

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Auteurs

- [Votre Nom]
- [Autres contributeurs]

## Remerciements

- MediaPipe pour la détection faciale
- OpenCV pour le traitement d'image
- Cassandra pour le stockage des données
- La communauté open source