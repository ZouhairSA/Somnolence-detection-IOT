from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import WhiteListRoundRobinPolicy, DowngradingConsistencyRetryPolicy
from cassandra.auth import PlainTextAuthProvider
from datetime import datetime
import json

class CassandraManager:
    def __init__(self, contact_points=['localhost'], port=9042, keyspace='vigilance_db'):
        """Initialise la connexion à Cassandra"""
        self.contact_points = contact_points
        self.port = port
        self.keyspace = keyspace
        self.session = None
        self.connect()

    def connect(self):
        """Établit la connexion à Cassandra avec une configuration simplifiée"""
        try:
            # Configuration de base sans event loop
            profile = ExecutionProfile(
                load_balancing_policy=WhiteListRoundRobinPolicy(self.contact_points),
                retry_policy=DowngradingConsistencyRetryPolicy(),
                request_timeout=10
            )

            auth_provider = PlainTextAuthProvider(username='cassandra', password='cassandra')
            
            cluster = Cluster(
                contact_points=self.contact_points,
                port=self.port,
                auth_provider=auth_provider,
                execution_profiles={EXEC_PROFILE_DEFAULT: profile},
                protocol_version=4  # Utilisation d'une version de protocole plus stable
            )

            self.session = cluster.connect()
            
            # Création du keyspace si nécessaire
            self.session.execute(f"""
                CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
                WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
            """)
            
            self.session.set_keyspace(self.keyspace)
            self._create_tables()
            print("Connexion à Cassandra établie avec succès")
            
        except Exception as e:
            print(f"Erreur de connexion à Cassandra: {str(e)}")
            raise

    def _create_tables(self):
        """Crée les tables nécessaires dans Cassandra"""
        try:
            # Table pour les événements de fatigue
            self.session.execute("""
                CREATE TABLE IF NOT EXISTS fatigue_events (
                    event_id UUID PRIMARY KEY,
                    timestamp TIMESTAMP,
                    event_type TEXT,
                    confidence FLOAT,
                    details TEXT,
                    fatigue_level INT,
                    device_id TEXT,
                    session_id TEXT
                )
            """)

            # Table pour les statistiques de session
            self.session.execute("""
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
                )
            """)

            # Table pour les alertes
            self.session.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id UUID PRIMARY KEY,
                    timestamp TIMESTAMP,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    device_id TEXT,
                    session_id TEXT
                )
            """)
            print("Tables créées avec succès")
        except Exception as e:
            print(f"Erreur lors de la création des tables: {str(e)}")
            raise

    def log_fatigue_event(self, event_type, confidence, details, fatigue_level, device_id, session_id):
        """Enregistre un événement de fatigue"""
        try:
            query = """
                INSERT INTO fatigue_events (
                    event_id, timestamp, event_type, confidence, 
                    details, fatigue_level, device_id, session_id
                ) VALUES (uuid(), %s, %s, %s, %s, %s, %s, %s)
            """
            self.session.execute(query, (
                datetime.now(), event_type, confidence, 
                json.dumps(details), fatigue_level, device_id, session_id
            ))
        except Exception as e:
            print(f"Erreur lors de l'enregistrement de l'événement: {str(e)}")

    def log_alert(self, alert_type, severity, message, device_id, session_id):
        """Enregistre une alerte"""
        try:
            query = """
                INSERT INTO alerts (
                    alert_id, timestamp, alert_type, severity, 
                    message, device_id, session_id
                ) VALUES (uuid(), %s, %s, %s, %s, %s, %s)
            """
            self.session.execute(query, (
                datetime.now(), alert_type, severity, 
                message, device_id, session_id
            ))
        except Exception as e:
            print(f"Erreur lors de l'enregistrement de l'alerte: {str(e)}")

    def update_session_stats(self, session_id, stats):
        """Met à jour les statistiques de la session"""
        try:
            query = """
                INSERT INTO session_stats (
                    session_id, start_time, end_time, total_blinks,
                    total_yawns, total_microsleeps, max_fatigue_level,
                    avg_fatigue_level, device_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.session.execute(query, (
                session_id, stats['start_time'], stats['end_time'],
                stats['total_blinks'], stats['total_yawns'],
                stats['total_microsleeps'], stats['max_fatigue_level'],
                stats['avg_fatigue_level'], stats['device_id']
            ))
        except Exception as e:
            print(f"Erreur lors de la mise à jour des statistiques: {str(e)}")

    def get_session_stats(self, session_id):
        """Récupère les statistiques d'une session"""
        try:
            query = "SELECT * FROM session_stats WHERE session_id = %s"
            result = self.session.execute(query, (session_id,))
            return result.one()
        except Exception as e:
            print(f"Erreur lors de la récupération des statistiques: {str(e)}")
            return None

    def close(self):
        """Ferme la connexion à Cassandra"""
        if self.session:
            self.session.cluster.shutdown()
            self.session.shutdown() 