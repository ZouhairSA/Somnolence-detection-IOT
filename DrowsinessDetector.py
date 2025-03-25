import queue
import threading
import time
import pygame
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
import mysql.connector
import logging
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QProgressBar, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, QObject

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vigilance_core.log"),
        logging.StreamHandler()
    ]
)

class VigilanceSignals(QObject):
    """Classe pour émettre des signaux afin de mettre à jour l'interface graphique dans le thread principal."""
    update_ui_signal = pyqtSignal(str, dict)
    display_frame_signal = pyqtSignal(str, QImage, float)
    update_stats_signal = pyqtSignal(str, dict)

class VigilanceCore(QMainWindow):
    def __init__(self, cameras):
        super().__init__()

        # Initialisation de pygame pour le son
        pygame.mixer.init()
        try:
            self.alert_sound = pygame.mixer.Sound("runs\sunflower-street-drumloop-85bpm-163900.mp3")  # Remplacez par le chemin de votre fichier son
        except pygame.error as e:
            logging.error(f"Erreur de chargement du son : {e}")
            self.alert_sound = None

        # Signaux pour les mises à jour de l'interface
        self.signals = VigilanceSignals()
        self.signals.update_ui_signal.connect(self.update_ui)
        self.signals.display_frame_signal.connect(self.display_frame)
        self.signals.update_stats_signal.connect(self.update_stats_ui)

        # Liste des caméras
        self.cameras = cameras
        self.camera_objects = {}

        # Initialisation des états pour chaque caméra
        self.states = {}
        for camera in self.cameras:
            self.states[camera["name"]] = {
                "yawn_state": "",
                "left_eye_state": "",
                "right_eye_state": "",
                "alert_text": "",
                "alert_type": "",
                "fatigue_level": 0,
                "blinks": 0,
                "microsleeps": 0,
                "yawns": 0,
                "yawn_duration": 0,
                "fps": 0,
                "frame_count": 0,
                "start_time": time.time(),
                "alert_count": 0,
                "last_alert_time": 0,
                "left_eye_still_closed": False,
                "right_eye_still_closed": False,
                "yawn_in_progress": False,
                "frame_queue": queue.Queue(maxsize=2),
                "stop_event": threading.Event(),
                "cap": None,
                "video_label": None,
                "name_label": None,
                "fps_label": None,
                "highlight_timer": None,
                "status_icon": None,
                "connection_status": "Déconnecté",
                "recent_alerts": [],
                "last_detection_time": time.time(),  # Pour gérer les cas où aucun visage n'est détecté
                "eye_closed_frames": 0,  # Compteur pour lisser les détections d'yeux fermés
                "yawn_frames": 0,  # Compteur pour lisser les détections de bâillements
                "frame_interval": 0.045  # Intervalle moyen entre frames (45ms)
            }

        # Initialisation de MediaPipe FaceMesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.points_ids = [187, 411, 152, 68, 174, 399, 298]  # Points pour la bouche et les yeux

        # Connexion à la base de données MySQL
        try:
            self.db_connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",  # Remplacez par votre mot de passe MySQL
                database="vigilance_db"
            )
            self.db_cursor = self.db_connection.cursor()
            logging.info("Connexion à la base de données réussie")
        except mysql.connector.Error as err:
            logging.error(f"Erreur de connexion à la base de données : {err}")
            sys.exit(1)

        # Créer les tables si elles n'existent pas
        self.create_db_tables()

        # Compter les alertes existantes pour chaque caméra
        for camera in self.cameras:
            camera_name = camera["name"]
            self.db_cursor.execute("SELECT COUNT(*) FROM alertes WHERE camera_name = %s", (camera_name,))
            self.states[camera_name]["alert_count"] = self.db_cursor.fetchone()[0]

        # Configuration de la fenêtre principale
        self.setWindowTitle("Vigilance Core - Drowsiness Detection")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1F2A44, stop:1 #3E4C75);")

        # Widget central et layout principal
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)

        # Header sophistiqué
        self.header_widget = QWidget()
        self.header_layout = QHBoxLayout(self.header_widget)
        self.header_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4682B4, stop:1 #5A9BD4);
            border-radius: 12px;
            border: 1px solid #FFFFFF;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
            padding: 10px;
        """)

        # Logo école (gauche)
        self.school_logo = QLabel(self)
        school_pixmap = QPixmap("runs/img/hestim.png").scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.school_logo.setPixmap(school_pixmap)
        self.school_logo.setStyleSheet("""
            border-radius: 40px;
            background-color: #FFFFFF;
            padding: 5px;
            border: 1px solid #4682B4;
        """)
        self.header_layout.addWidget(self.school_logo)

        # Titre
        self.header_title = QLabel("Vigilance Core")
        self.header_title.setFont(QFont("Lato", 28, QFont.Bold))
        self.header_title.setStyleSheet("""
            color: #FFFFFF;
            text-align: center;
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
            background: none;
        """)
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.header_title)
        self.header_layout.addStretch()

        # Logo matière (droite)
        self.subject_logo = QLabel(self)
        subject_pixmap = QPixmap("runs/img/iot.jpg").scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.subject_logo.setPixmap(subject_pixmap)
        self.subject_logo.setStyleSheet("""
            border-radius: 40px;
            background-color: #FFFFFF;
            padding: 5px;
            border: 1px solid #4682B4;
        """)
        self.header_layout.addWidget(self.subject_logo)

        self.main_layout.addWidget(self.header_widget, stretch=1)

        # Contenu principal
        self.content_widget = QWidget()
        self.content_layout = QHBoxLayout(self.content_widget)
        self.content_layout.setSpacing(20)
        self.main_layout.addWidget(self.content_widget, stretch=8)

        # Zone d'affichage des vidéos (grille)
        self.video_widget = QWidget()
        self.video_layout = QGridLayout(self.video_widget)
        self.video_layout.setSpacing(10)

        num_cameras = len(self.cameras)
        cols = 2
        rows = (num_cameras + cols - 1) // cols
        for idx, camera in enumerate(self.cameras):
            camera_name = camera["name"]
            video_container = QWidget()
            video_container_layout = QVBoxLayout(video_container)
            video_container_layout.setContentsMargins(0, 0, 0, 0)
            video_container_layout.setSpacing(5)

            video_label = QLabel(self)
            video_label.setStyleSheet("""
                border-radius: 12px;
                background-color: #1F2A44;
                border: 2px solid #4682B4;
                box-shadow: 0 0 15px rgba(70, 130, 180, 0.3);
            """)
            video_label.setMinimumSize(320, 240)
            video_label.setScaledContents(True)

            info_widget = QWidget()
            info_layout = QHBoxLayout(info_widget)
            info_layout.setContentsMargins(0, 0, 0, 0)

            name_label = QLabel(camera_name)
            name_label.setFont(QFont("Lato", 12, QFont.Bold))
            name_label.setStyleSheet("color: #FFFFFF; background: none;")

            status_icon = QLabel("🔴")
            status_icon.setFont(QFont("Lato", 12))
            status_icon.setStyleSheet("background: none;")
            self.states[camera_name]["status_icon"] = status_icon

            fps_label = QLabel("FPS: 0")
            fps_label.setFont(QFont("Lato", 10))
            fps_label.setStyleSheet("color: #A9A9A9; background: none;")

            info_layout.addWidget(name_label)
            info_layout.addWidget(status_icon)
            info_layout.addStretch()
            info_layout.addWidget(fps_label)

            video_container_layout.addWidget(video_label)
            video_container_layout.addWidget(info_widget)

            row = idx // cols
            col = idx % cols
            self.video_layout.addWidget(video_container, row, col)
            self.states[camera_name]["video_label"] = video_label
            self.states[camera_name]["name_label"] = name_label
            self.states[camera_name]["fps_label"] = fps_label

        self.content_layout.addWidget(self.video_widget, stretch=3)

        # Panneau de contrôle
        self.control_widget = QWidget()
        self.control_layout = QVBoxLayout(self.control_widget)
        self.control_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4682B4, stop:1 #5A9BD4);
            border-radius: 12px;
            border: 1px solid #FFFFFF;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
            padding: 15px;
        """)
        self.content_layout.addWidget(self.control_widget, stretch=1)

        # Barre de fatigue, état et alertes récentes (par caméra)
        self.fatigue_bars = {}
        self.status_labels = {}
        self.alert_labels = {}
        self.recent_alerts_labels = {}
        for camera in self.cameras:
            camera_name = camera["name"]

            fatigue_bar = QProgressBar()
            fatigue_bar.setRange(0, 100)
            fatigue_bar.setValue(0)
            fatigue_bar.setTextVisible(True)
            fatigue_bar.setFormat(f"{camera_name} - Fatigue: %p%")
            fatigue_bar.setStyleSheet("""
                QProgressBar {
                    border: none;
                    border-radius: 8px;
                    background-color: #1F2A44;
                    text-align: center;
                    color: #FFFFFF;
                    font-family: 'Lato';
                    font-size: 14px;
                    box-shadow: 0 0 5px rgba(255, 255, 255, 0.1);
                }
                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4682B4, stop:1 #87CEEB);
                    border-radius: 8px;
                }
            """)
            fatigue_bar.setToolTip(f"Niveau de fatigue pour {camera_name}")
            self.fatigue_bars[camera_name] = fatigue_bar
            self.control_layout.addWidget(fatigue_bar)

            status_label = QLabel(f"{camera_name} - État: Optimal")
            status_label.setFont(QFont("Lato", 14, QFont.Bold))
            status_label.setStyleSheet("""
                color: #FFFFFF;
                text-align: center;
                padding: 8px;
                background-color: rgba(70, 130, 180, 0.2);
                border-radius: 8px;
                box-shadow: 0 0 5px rgba(255, 255, 255, 0.1);
            """)
            status_label.setFixedHeight(40)
            self.status_labels[camera_name] = status_label
            self.control_layout.addWidget(status_label)

            alert_label = QLabel("")
            alert_label.setFont(QFont("Lato", 12, QFont.Bold))
            alert_label.setStyleSheet("""
                color: #B22222;
                text-align: center;
                padding: 8px;
                background-color: rgba(178, 34, 34, 0.2);
                border-radius: 8px;
                box-shadow: 0 0 5px rgba(178, 34, 34, 0.3);
            """)
            alert_label.setFixedHeight(40)
            self.alert_labels[camera_name] = alert_label
            self.control_layout.addWidget(alert_label)

            recent_alerts_label = QLabel("Alertes récentes:\nAucune alerte")
            recent_alerts_label.setFont(QFont("Lato", 10))
            recent_alerts_label.setStyleSheet("""
                color: #FFFFFF;
                padding: 8px;
                background-color: rgba(70, 130, 180, 0.1);
                border-radius: 8px;
            """)
            recent_alerts_label.setFixedHeight(80)
            self.recent_alerts_labels[camera_name] = recent_alerts_label
            self.control_layout.addWidget(recent_alerts_label)

        self.control_layout.addStretch()

        # Boutons
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(10)

        self.capture_button = QPushButton("Capturer Image")
        self.capture_button.setFont(QFont("Lato", 12, QFont.Bold))
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #32CD32;
                color: #FFFFFF;
                padding: 8px;
                border-radius: 8px;
                border: none;
                box-shadow: 0 0 5px rgba(255, 255, 255, 0.2);
            }
            QPushButton:hover {
                background-color: #3CB371;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
            }
        """)
        self.capture_button.clicked.connect(self.capture_image)
        self.button_layout.addWidget(self.capture_button)

        self.refresh_button = QPushButton("Rafraîchir")
        self.refresh_button.setFont(QFont("Lato", 12, QFont.Bold))
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #FFA500;
                color: #FFFFFF;
                padding: 8px;
                border-radius: 8px;
                border: none;
                box-shadow: 0 0 5px rgba(255, 255, 255, 0.2);
            }
            QPushButton:hover {
                background-color: #FFB347;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
            }
        """)
        self.refresh_button.clicked.connect(self.refresh_stats)
        self.button_layout.addWidget(self.refresh_button)

        self.reset_button = QPushButton("Réinitialiser")
        self.reset_button.setFont(QFont("Lato", 12, QFont.Bold))
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #4682B4;
                color: #FFFFFF;
                padding: 8px;
                border-radius: 8px;
                border: none;
                box-shadow: 0 0 5px rgba(255, 255, 255, 0.2);
            }
            QPushButton:hover {
                background-color: #5A9BD4;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
            }
        """)
        self.reset_button.clicked.connect(self.reset_stats)
        self.button_layout.addWidget(self.reset_button)

        self.quit_button = QPushButton("Arrêt")
        self.quit_button.setFont(QFont("Lato", 12, QFont.Bold))
        self.quit_button.setStyleSheet("""
            QPushButton {
                background-color: #B22222;
                color: #FFFFFF;
                padding: 8px;
                border-radius: 8px;
                border: none;
                box-shadow: 0 0 5px rgba(255, 255, 255, 0.2);
            }
            QPushButton:hover {
                background-color: #DC143C;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
            }
        """)
        self.quit_button.clicked.connect(self.close)
        self.button_layout.addWidget(self.quit_button)

        self.control_layout.addLayout(self.button_layout)

        # Tableau des détections (en bas)
        self.detections_table = QTableWidget()
        self.detections_table.setRowCount(len(self.cameras))
        self.detections_table.setColumnCount(7)
        self.detections_table.setHorizontalHeaderLabels(["Caméra", "Nombre d'alertes", "Type de cas", "Dernier cas détecté", "Clignements", "Micro-sommeils", "Action"])
        self.detections_table.setStyleSheet("""
            QTableWidget {
                background-color: #1F2A44;
                color: #FFFFFF;
                border-radius: 8px;
                border: 1px solid #4682B4;
                font-family: 'Lato';
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
                border: none;
            }
            QTableWidget::item:hover {
                background-color: rgba(70, 130, 180, 0.3);
            }
            QHeaderView::section {
                background-color: #4682B4;
                color: #FFFFFF;
                padding: 8px;
                border: none;
                font-family: 'Lato';
                font-size: 12px;
                font-weight: bold;
            }
        """)
        self.detections_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.detections_table.setFixedHeight(120)
        self.detections_table.setSelectionMode(QTableWidget.NoSelection)
        self.detections_table.setEditTriggers(QTableWidget.NoEditTriggers)

        for row, camera in enumerate(self.cameras):
            camera_name = camera["name"]
            self.detections_table.setItem(row, 0, QTableWidgetItem(camera_name))
            self.detections_table.setItem(row, 1, QTableWidgetItem(str(self.states[camera_name]["alert_count"])))
            self.detections_table.setItem(row, 2, QTableWidgetItem("Aucun"))
            self.detections_table.setItem(row, 3, QTableWidgetItem("Aucun cas détecté"))
            self.detections_table.setItem(row, 4, QTableWidgetItem(str(self.states[camera_name]["blinks"])))
            self.detections_table.setItem(row, 5, QTableWidgetItem(str(round(self.states[camera_name]["microsleeps"], 2))))

            delete_button = QPushButton("Delete")
            delete_button.setStyleSheet("""
                QPushButton {
                    background-color: #B22222;
                    color: #FFFFFF;
                    padding: 5px;
                    border-radius: 5px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #DC143C;
                }
            """)
            delete_button.clicked.connect(lambda checked, name=camera_name: self.delete_camera(name))
            self.detections_table.setCellWidget(row, 6, delete_button)

        self.main_layout.addWidget(self.detections_table, stretch=1)

        # Initialisation des modèles YOLO
        try:
            self.detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
            self.detecteye = YOLO("runs/detecteye/train/weights/best.pt")
        except FileNotFoundError:
            logging.error("Erreur : Les fichiers de modèle YOLO (best.pt) sont introuvables. Utilisation du modèle par défaut.")
            self.detectyawn = YOLO("yolov8n.pt")
            self.detecteye = YOLO("yolov8n.pt")

        # Initialisation des threads pour chaque caméra
        for camera in self.cameras:
            camera_name = camera["name"]
            logging.info(f"Tentative de connexion à la caméra {camera_name} ({camera['source']})...")
            cap = cv2.VideoCapture(camera["source"])
            if not cap.isOpened():
                logging.error(f"Erreur : Impossible de se connecter à la caméra {camera_name} ({camera['source']})")
                self.states[camera_name]["status_icon"].setText("🔴")
                continue
            logging.info(f"Connexion réussie à la caméra {camera_name}")
            self.states[camera_name]["cap"] = cap
            self.states[camera_name]["connection_status"] = "Connecté"
            self.states[camera_name]["status_icon"].setText("🟢")
            capture_thread = threading.Thread(target=self.capture_frames, args=(camera_name,), daemon=True)
            process_thread = threading.Thread(target=self.process_frames, args=(camera_name,), daemon=True)
            capture_thread.start()
            process_thread.start()
            self.camera_objects[camera_name] = {"capture_thread": capture_thread, "process_thread": process_thread}

        # Timer pour les animations
        self.alert_timers = {}
        self.alert_blink_states = {}
        for camera in self.cameras:
            camera_name = camera["name"]
            self.alert_blink_states[camera_name] = False
            alert_timer = QTimer(self)
            alert_timer.timeout.connect(lambda cam=camera_name: self.toggle_alert_glow(cam))
            self.alert_timers[camera_name] = alert_timer

            highlight_timer = QTimer(self)
            highlight_timer.setSingleShot(True)
            highlight_timer.timeout.connect(lambda cam=camera_name: self.clear_highlight(cam))
            self.states[camera_name]["highlight_timer"] = highlight_timer

        # Animation pour la barre de fatigue
        self.fatigue_animations = {}
        for camera in self.cameras:
            camera_name = camera["name"]
            animation = QPropertyAnimation(self.fatigue_bars[camera_name], b"value")
            animation.setEasingCurve(QEasingCurve.InOutQuad)
            self.fatigue_animations[camera_name] = animation

        # Timer pour la mise à jour en temps réel des stats
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_all_stats)
        self.stats_timer.start(5000)

    def create_db_tables(self):
        """Crée les tables nécessaires dans la base de données si elles n'existent pas."""
        try:
            self.db_cursor.execute("""
                CREATE TABLE IF NOT EXISTS alertes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    camera_name VARCHAR(255) NOT NULL,
                    detection_time DATETIME NOT NULL,
                    image_name VARCHAR(255) NOT NULL,
                    image_data LONGBLOB NOT NULL,
                    INDEX idx_camera_name (camera_name),
                    INDEX idx_detection_time (detection_time)
                )
            """)
            self.db_cursor.execute("""
                CREATE TABLE IF NOT EXISTS camera_stats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    camera_name VARCHAR(255) NOT NULL UNIQUE,
                    blinks INT DEFAULT 0,
                    microsleeps FLOAT DEFAULT 0,
                    yawns INT DEFAULT 0,
                    yawn_duration FLOAT DEFAULT 0,
                    fatigue_level INT DEFAULT 0,
                    alert_count INT DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_camera_name (camera_name)
                )
            """)
            self.db_connection.commit()
            logging.info("Tables de la base de données créées ou vérifiées avec succès")
        except mysql.connector.Error as err:
            logging.error(f"Erreur lors de la création des tables : {err}")
            sys.exit(1)

    def reconnect_db(self):
        """Tente de rétablir la connexion à la base de données MySQL."""
        try:
            if self.db_connection.is_connected():
                return
        except mysql.connector.Error:
            pass

        try:
            self.db_connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",  # Remplacez par votre mot de passe MySQL
                database="vigilance_db"
            )
            self.db_cursor = self.db_connection.cursor()
            logging.info("Reconnexion à la base de données réussie")
        except mysql.connector.Error as err:
            logging.error(f"Erreur lors de la reconnexion à la base de données : {err}")

    def update_all_stats(self):
        """Met à jour les statistiques pour toutes les caméras et les envoie à la base de données."""
        for camera_name in self.states:
            self.update_stats(camera_name)
            self.update_stats_db(camera_name)

    def update_stats_db(self, camera_name):
        """Met à jour les statistiques dans la base de données en temps réel."""
        state = self.states[camera_name]
        try:
            self.db_cursor.execute("SELECT COUNT(*) FROM camera_stats WHERE camera_name = %s", (camera_name,))
            count = self.db_cursor.fetchone()[0]

            if count > 0:
                query = """
                    UPDATE camera_stats 
                    SET blinks = %s, microsleeps = %s, yawns = %s, yawn_duration = %s, 
                        fatigue_level = %s, alert_count = %s, timestamp = CURRENT_TIMESTAMP
                    WHERE camera_name = %s
                """
                values = (
                    state["blinks"],
                    round(state["microsleeps"], 2),
                    state["yawns"],
                    round(state["yawn_duration"], 2),
                    state["fatigue_level"],
                    state["alert_count"],
                    camera_name
                )
            else:
                query = """
                    INSERT INTO camera_stats (
                        camera_name, blinks, microsleeps, yawns, yawn_duration, fatigue_level, alert_count
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    camera_name,
                    state["blinks"],
                    round(state["microsleeps"], 2),
                    state["yawns"],
                    round(state["yawn_duration"], 2),
                    state["fatigue_level"],
                    state["alert_count"]
                )

            self.db_cursor.execute(query, values)
            self.db_connection.commit()
            logging.info(f"Statistiques mises à jour dans la base de données pour {camera_name}")
        except mysql.connector.Error as err:
            logging.error(f"Erreur lors de la mise à jour des statistiques dans la base de données : {err}")
            self.reconnect_db()
            try:
                self.db_cursor.execute(query, values)
                self.db_connection.commit()
                logging.info(f"Statistiques mises à jour après reconnexion pour {camera_name}")
            except mysql.connector.Error as err:
                logging.error(f"Échec de la mise à jour après reconnexion : {err}")

    def update_stats_ui(self, camera_name, stats):
        """Met à jour l'interface avec les statistiques en temps réel."""
        for row, camera in enumerate(self.cameras):
            if camera["name"] == camera_name:
                self.detections_table.setItem(row, 4, QTableWidgetItem(str(stats["blinks"])))
                self.detections_table.setItem(row, 5, QTableWidgetItem(str(stats["microsleeps"])))
                break

    def capture_image(self):
        """Capture une image manuellement pour toutes les caméras actives."""
        for camera_name in self.states:
            state = self.states[camera_name]
            try:
                frame = state["frame_queue"].get(timeout=0.1)
                current_time = time.time()
                detection_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
                image_name = f"{camera_name}_manual_{detection_time.replace(':', '-')}.jpg"
                _, buffer = cv2.imencode('.jpg', frame)
                image_data = buffer.tobytes()
                threading.Thread(target=self.save_alert_to_db, args=(camera_name, detection_time, image_name, image_data)).start()
                logging.info(f"Image capturée manuellement pour {camera_name}: {image_name}")
            except queue.Empty:
                logging.warning(f"Aucune frame disponible pour la capture manuelle ({camera_name})")

    def save_camera_stats_to_db(self, camera_name):
        """Enregistre les statistiques de la caméra dans la table camera_stats avant suppression."""
        state = self.states[camera_name]
        try:
            query = """
                INSERT INTO camera_stats (
                    camera_name, blinks, microsleeps, yawns, yawn_duration, fatigue_level, alert_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    blinks = VALUES(blinks),
                    microsleeps = VALUES(microsleeps),
                    yawns = VALUES(yawns),
                    yawn_duration = VALUES(yawn_duration),
                    fatigue_level = VALUES(fatigue_level),
                    alert_count = VALUES(alert_count),
                    timestamp = CURRENT_TIMESTAMP
            """
            values = (
                camera_name,
                state["blinks"],
                round(state["microsleeps"], 2),
                state["yawns"],
                round(state["yawn_duration"], 2),
                state["fatigue_level"],
                state["alert_count"]
            )
            self.db_cursor.execute(query, values)
            self.db_connection.commit()
            logging.info(f"Statistiques de la caméra {camera_name} enregistrées dans camera_stats")
        except mysql.connector.Error as err:
            logging.error(f"Erreur lors de l'enregistrement des statistiques dans camera_stats : {err}")
            self.reconnect_db()
            try:
                self.db_cursor.execute(query, values)
                self.db_connection.commit()
                logging.info(f"Statistiques enregistrées après reconnexion pour {camera_name}")
            except mysql.connector.Error as err:
                logging.error(f"Échec de l'enregistrement après reconnexion : {err}")

    def delete_camera(self, camera_name):
        """Supprime une caméra de l'interface avec une pop-up de confirmation et enregistre les données."""
        reply = QMessageBox.question(
            self,
            "Confirmation de suppression",
            f"Êtes-vous sûr de vouloir supprimer la caméra {camera_name} ?\n"
            f"Les statistiques seront enregistrées dans la base de données.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        self.save_camera_stats_to_db(camera_name)

        if camera_name in self.states:
            video_label = self.states[camera_name]["video_label"]
            name_label = self.states[camera_name]["name_label"]
            fps_label = self.states[camera_name]["fps_label"]
            fade_out = QPropertyAnimation(video_label, b"windowOpacity")
            fade_out.setDuration(500)
            fade_out.setStartValue(1.0)
            fade_out.setEndValue(0.0)
            fade_out.start()

            fade_out_name = QPropertyAnimation(name_label, b"windowOpacity")
            fade_out_name.setDuration(500)
            fade_out_name.setStartValue(1.0)
            fade_out_name.setEndValue(0.0)
            fade_out_name.start()

            fade_out_fps = QPropertyAnimation(fps_label, b"windowOpacity")
            fade_out_fps.setDuration(500)
            fade_out_fps.setStartValue(1.0)
            fade_out_fps.setEndValue(0.0)
            fade_out_fps.start()

            QTimer.singleShot(500, lambda: self._perform_camera_deletion(camera_name))

    def _perform_camera_deletion(self, camera_name):
        """Effectue la suppression de la caméra après l'animation."""
        if camera_name in self.camera_objects:
            state = self.states[camera_name]
            state["stop_event"].set()
            if state["cap"]:
                state["cap"].release()
            self.camera_objects[camera_name]["capture_thread"].join()
            self.camera_objects[camera_name]["process_thread"].join()
            del self.camera_objects[camera_name]

        if camera_name in self.states:
            self.video_layout.removeWidget(self.states[camera_name]["video_label"].parent())
            self.states[camera_name]["video_label"].parent().deleteLater()

            self.control_layout.removeWidget(self.fatigue_bars[camera_name])
            self.control_layout.removeWidget(self.status_labels[camera_name])
            self.control_layout.removeWidget(self.alert_labels[camera_name])
            self.control_layout.removeWidget(self.recent_alerts_labels[camera_name])
            self.fatigue_bars[camera_name].deleteLater()
            self.status_labels[camera_name].deleteLater()
            self.alert_labels[camera_name].deleteLater()
            self.recent_alerts_labels[camera_name].deleteLater()

            del self.fatigue_bars[camera_name]
            del self.status_labels[camera_name]
            del self.alert_labels[camera_name]
            del self.recent_alerts_labels[camera_name]
            del self.fatigue_animations[camera_name]
            del self.alert_timers[camera_name]
            del self.alert_blink_states[camera_name]
            del self.states[camera_name]["highlight_timer"]

        self.cameras = [cam for cam in self.cameras if cam["name"] != camera_name]
        del self.states[camera_name]

        self.detections_table.setRowCount(len(self.cameras))
        for row, camera in enumerate(self.cameras):
            camera_name = camera["name"]
            self.detections_table.setItem(row, 0, QTableWidgetItem(camera_name))
            self.detections_table.setItem(row, 1, QTableWidgetItem(str(self.states[camera_name]["alert_count"])))
            alert_type = self.states[camera_name]["alert_type"]
            if alert_type == "Bâillement":
                alert_type = "😴 Bâillement"
            elif alert_type == "Micro-sommeil":
                alert_type = "💤 Micro-sommeil"
            else:
                alert_type = "Aucun"
            self.detections_table.setItem(row, 2, QTableWidgetItem(alert_type))
            self.detections_table.setItem(row, 3, QTableWidgetItem(self.states[camera_name]["alert_text"] if self.states[camera_name]["alert_text"] else "Aucun cas détecté"))
            self.detections_table.setItem(row, 4, QTableWidgetItem(str(self.states[camera_name]["blinks"])))
            self.detections_table.setItem(row, 5, QTableWidgetItem(str(round(self.states[camera_name]["microsleeps"], 2))))

            delete_button = QPushButton("Delete")
            delete_button.setStyleSheet("""
                QPushButton {
                    background-color: #B22222;
                    color: #FFFFFF;
                    padding: 5px;
                    border-radius: 5px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #DC143C;
                }
            """)
            delete_button.clicked.connect(lambda checked, name=camera_name: self.delete_camera(name))
            self.detections_table.setCellWidget(row, 6, delete_button)

        for i in reversed(range(self.video_layout.count())):
            self.video_layout.itemAt(i).widget().setParent(None)
        num_cameras = len(self.cameras)
        cols = 2
        rows = (num_cameras + cols - 1) // cols
        for idx, camera in enumerate(self.cameras):
            camera_name = camera["name"]
            row = idx // cols
            col = idx % cols
            self.video_layout.addWidget(self.states[camera_name]["video_label"].parent(), row, col)

    def highlight_row(self, camera_name):
        """Surligne la ligne du tableau pour une caméra lorsqu'une alerte est détectée."""
        for row, camera in enumerate(self.cameras):
            if camera["name"] == camera_name:
                for col in range(self.detections_table.columnCount()):
                    item = self.detections_table.item(row, col)
                    if item:
                        item.setBackground(QColor(178, 34, 34, 100))
                self.states[camera_name]["highlight_timer"].start(2000)
                break

    def clear_highlight(self, camera_name):
        """Supprime le surlignage de la ligne du tableau."""
        for row, camera in enumerate(self.cameras):
            if camera["name"] == camera_name:
                for col in range(self.detections_table.columnCount()):
                    item = self.detections_table.item(row, col)
                    if item:
                        item.setBackground(QColor(0, 0, 0, 0))
                break

    def update_detections_table(self):
        """Met à jour le tableau des détections pour toutes les caméras."""
        for row, camera in enumerate(self.cameras):
            camera_name = camera["name"]
            try:
                self.db_cursor.execute("SELECT COUNT(*) FROM alertes WHERE camera_name = %s", (camera_name,))
                self.states[camera_name]["alert_count"] = self.db_cursor.fetchone()[0]
            except mysql.connector.Error as err:
                logging.error(f"Erreur lors de la récupération du nombre d'alertes pour {camera_name} : {err}")
                self.reconnect_db()
                try:
                    self.db_cursor.execute("SELECT COUNT(*) FROM alertes WHERE camera_name = %s", (camera_name,))
                    self.states[camera_name]["alert_count"] = self.db_cursor.fetchone()[0]
                except mysql.connector.Error as err:
                    logging.error(f"Échec de la récupération après reconnexion pour {camera_name} : {err}")
                    self.states[camera_name]["alert_count"] = 0

            alert_count = str(self.states[camera_name]["alert_count"])
            alert_type = self.states[camera_name]["alert_type"]
            if alert_type == "Bâillement":
                alert_type = "😴 Bâillement"
            elif alert_type == "Micro-sommeil":
                alert_type = "💤 Micro-sommeil"
            else:
                alert_type = "Aucun"
            alert_text = self.states[camera_name]["alert_text"] if self.states[camera_name]["alert_text"] else "Aucun cas détecté"
            self.detections_table.setItem(row, 1, QTableWidgetItem(alert_count))
            self.detections_table.setItem(row, 2, QTableWidgetItem(alert_type))
            self.detections_table.setItem(row, 3, QTableWidgetItem(alert_text))

    def save_alert_to_db(self, camera_name, detection_time, image_name, image_data):
        """Enregistre une alerte dans la base de données."""
        try:
            query = "INSERT INTO alertes (camera_name, detection_time, image_name, image_data) VALUES (%s, %s, %s, %s)"
            values = (camera_name, detection_time, image_name, image_data)
            self.db_cursor.execute(query, values)
            self.db_connection.commit()
            self.db_cursor.execute("SELECT COUNT(*) FROM alertes WHERE camera_name = %s", (camera_name,))
            self.states[camera_name]["alert_count"] = self.db_cursor.fetchone()[0]
            logging.info(f"Alerte enregistrée dans la base de données pour la caméra {camera_name} avec le nom d'image {image_name}")
        except mysql.connector.Error as err:
            logging.error(f"Erreur lors de l'enregistrement dans la base de données : {err}")
            self.reconnect_db()
            try:
                self.db_cursor.execute(query, values)
                self.db_connection.commit()
                self.db_cursor.execute("SELECT COUNT(*) FROM alertes WHERE camera_name = %s", (camera_name,))
                self.states[camera_name]["alert_count"] = self.db_cursor.fetchone()[0]
                logging.info(f"Alerte enregistrée après reconnexion pour la caméra {camera_name}")
            except mysql.connector.Error as err:
                logging.error(f"Échec de l'enregistrement après reconnexion : {err}")

    def update_stats(self, camera_name):
        """Met à jour les statistiques de détection pour une caméra donnée."""
        if camera_name not in self.states:
            return

        state = self.states[camera_name]
        new_fatigue_level = min(100, int((state["microsleeps"] + state["yawn_duration"]) * 10))
        if new_fatigue_level != state["fatigue_level"]:
            self.signals.update_ui_signal.emit(camera_name, {
                "fatigue_level": new_fatigue_level,
                "action": "update_fatigue"
            })
        state["fatigue_level"] = new_fatigue_level

        current_time = time.time()
        detection_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        image_name = f"{camera_name}_alert_{detection_time.replace(':', '-')}.jpg"

        if round(state["yawn_duration"], 2) > 0.2:
            state["alert_text"] = "⚠ Bâillement prolongé"
            state["alert_type"] = "Bâillement"
            state["recent_alerts"].append(f"{detection_time}: Bâillement")
            if len(state["recent_alerts"]) > 3:
                state["recent_alerts"].pop(0)
            self.signals.update_ui_signal.emit(camera_name, {
                "status_text": f"{camera_name} - État: Attention",
                "status_style": """
                    color: #B22222;
                    text-align: center;
                    padding: 8px;
                    background-color: rgba(178, 34, 34, 0.2);
                    border-radius: 8px;
                    box-shadow: 0 0 5px rgba(178, 34, 34, 0.3);
                """,
                "alert_text": state["alert_text"],
                "action": "update_alert",
                "start_alert_timer": True,
                "recent_alerts": "\n".join(state["recent_alerts"])
            })
            self.states[camera_name]["status_icon"].setText("🟠")
            self.play_sound_in_thread()
            self.signals.update_ui_signal.emit(camera_name, {
                "action": "highlight_row"
            })

            try:
                frame = state["frame_queue"].get(timeout=0.1)
                _, buffer = cv2.imencode('.jpg', frame)
                image_data = buffer.tobytes()
                threading.Thread(target=self.save_alert_to_db, args=(camera_name, detection_time, image_name, image_data)).start()
                state["last_alert_time"] = current_time
            except queue.Empty:
                logging.warning(f"Aucune frame disponible pour la capture ({camera_name})")

        elif round(state["microsleeps"], 2) > 0.5:
            state["alert_text"] = "⚠ Micro-sommeil détecté"
            state["alert_type"] = "Micro-sommeil"
            state["recent_alerts"].append(f"{detection_time}: Micro-sommeil")
            if len(state["recent_alerts"]) > 3:
                state["recent_alerts"].pop(0)
            self.signals.update_ui_signal.emit(camera_name, {
                "status_text": f"{camera_name} - État: Critique",
                "status_style": """
                    color: #B22222;
                    text-align: center;
                    padding: 8px;
                    background-color: rgba(178, 34, 34, 0.2);
                    border-radius: 8px;
                    box-shadow: 0 0 5px rgba(178, 34, 34, 0.3);
                """,
                "alert_text": state["alert_text"],
                "action": "update_alert",
                "start_alert_timer": True,
                "recent_alerts": "\n".join(state["recent_alerts"])
            })
            self.states[camera_name]["status_icon"].setText("🔴")
            self.play_sound_in_thread()
            self.signals.update_ui_signal.emit(camera_name, {
                "action": "highlight_row"
            })

            try:
                frame = state["frame_queue"].get(timeout=0.1)
                _, buffer = cv2.imencode('.jpg', frame)
                image_data = buffer.tobytes()
                threading.Thread(target=self.save_alert_to_db, args=(camera_name, detection_time, image_name, image_data)).start()
                state["last_alert_time"] = current_time
            except queue.Empty:
                logging.warning(f"Aucune frame disponible pour la capture ({camera_name})")

        else:
            state["alert_text"] = ""
            state["alert_type"] = ""
            self.signals.update_ui_signal.emit(camera_name, {
                "status_text": f"{camera_name} - État: Optimal",
                "status_style": """
                    color: #FFFFFF;
                    text-align: center;
                    padding: 8px;
                    background-color: rgba(70, 130, 180, 0.2);
                    border-radius: 8px;
                    box-shadow: 0 0 5px rgba(255, 255, 255, 0.1);
                """,
                "alert_text": state["alert_text"],
                "action": "update_alert",
                "stop_alert_timer": True,
                "recent_alerts": "\n".join(state["recent_alerts"]) if state["recent_alerts"] else "Aucune alerte"
            })
            self.states[camera_name]["status_icon"].setText("🟢")

        self.signals.update_stats_signal.emit(camera_name, {
            "blinks": state["blinks"],
            "microsleeps": round(state["microsleeps"], 2),
            "yawns": state["yawns"],
            "yawn_duration": round(state["yawn_duration"], 2),
            "fatigue_level": state["fatigue_level"],
            "alert_count": state["alert_count"]
        })

        self.update_detections_table()

        state["frame_count"] += 1
        elapsed_time = time.time() - state["start_time"]
        if elapsed_time > 1:
            state["fps"] = state["frame_count"] / elapsed_time
            state["frame_count"] = 0
            state["start_time"] = time.time()

    def update_ui(self, camera_name, data):
        """Met à jour l'interface graphique dans le thread principal."""
        if camera_name not in self.states:
            return

        action = data.get("action")
        if action == "update_fatigue":
            new_fatigue_level = data["fatigue_level"]
            self.fatigue_animations[camera_name].setStartValue(self.states[camera_name]["fatigue_level"])
            self.fatigue_animations[camera_name].setEndValue(new_fatigue_level)
            self.fatigue_animations[camera_name].setDuration(500)
            self.fatigue_animations[camera_name].start()

        elif action == "update_alert":
            self.status_labels[camera_name].setText(data["status_text"])
            self.status_labels[camera_name].setStyleSheet(data["status_style"])
            self.alert_labels[camera_name].setText(data["alert_text"])
            self.recent_alerts_labels[camera_name].setText(f"Alertes récentes:\n{data['recent_alerts']}")
            if data.get("start_alert_timer"):
                if not self.alert_timers[camera_name].isActive():
                    self.alert_timers[camera_name].start(400)
            elif data.get("stop_alert_timer"):
                self.alert_timers[camera_name].stop()

        elif action == "highlight_row":
            self.highlight_row(camera_name)

    def toggle_alert_glow(self, camera_name):
        """Alterne l'effet de glow sur l'étiquette d'alerte."""
        if camera_name not in self.alert_blink_states:
            return
        self.alert_blink_states[camera_name] = not self.alert_blink_states[camera_name]
        glow = "0 0 15px rgba(178, 34, 34, 0.5)" if self.alert_blink_states[camera_name] else "0 0 5px rgba(178, 34, 34, 0.3)"
        self.alert_labels[camera_name].setStyleSheet(f"""
            color: #B22222;
            text-align: center;
            padding: 8px;
            background-color: rgba(178, 34, 34, 0.2);
            border-radius: 8px;
            box-shadow: {glow};
        """)

    def reset_stats(self):
        """Réinitialise les statistiques pour toutes les caméras."""
        for camera_name in list(self.states.keys()):
            state = self.states[camera_name]
            state["blinks"] = 0
            state["microsleeps"] = 0
            state["yawns"] = 0
            state["yawn_duration"] = 0
            state["fatigue_level"] = 0
            state["eye_closed_frames"] = 0
            state["yawn_frames"] = 0
            self.update_stats(camera_name)

    def refresh_stats(self):
        """Rafraîchit manuellement les statistiques."""
        for camera_name in self.states:
            self.update_stats(camera_name)
            self.update_stats_db(camera_name)

    def predict_eye(self, eye_frame, eye_state, camera_name):
        """Prédit l'état de l'œil (ouvert/fermé) à l'aide du modèle YOLO."""
        try:
            results_eye = self.detecteye.predict(eye_frame, conf=0.3)
            boxes = results_eye[0].boxes
            if len(boxes) == 0:
                return eye_state

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_confidence_index = np.argmax(confidences)
            class_id = int(class_ids[max_confidence_index])

            if class_id == 1:
                eye_state = "Close Eye"
            elif class_id == 0 and confidences[max_confidence_index] > 0.30:
                eye_state = "Open Eye"
            return eye_state
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction des yeux pour {camera_name} : {e}")
            return eye_state

    def predict_yawn(self, yawn_frame, current_yawn_state, camera_name):
        """Prédit l'état de bâillement à l'aide du modèle YOLO."""
        try:
            results_yawn = self.detectyawn.predict(yawn_frame, conf=0.5)
            boxes = results_yawn[0].boxes

            if len(boxes) == 0:
                return current_yawn_state

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_confidence_index = np.argmax(confidences)
            class_id = int(class_ids[max_confidence_index])

            if class_id == 0:
                return "Yawn"
            elif class_id == 1 and confidences[max_confidence_index] > 0.50:
                return "No Yawn"
            return current_yawn_state
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction du bâillement pour {camera_name} : {e}")
            return current_yawn_state

    def capture_frames(self, camera_name):
        """Capture les frames de la caméra et les met dans une file d'attente."""
        state = self.states[camera_name]
        cap = state["cap"]
        retry_attempts = 5
        while not state["stop_event"].is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Perte de connexion avec la caméra {camera_name}. Tentative de reconnexion...")
                    cap.release()
                    for attempt in range(retry_attempts):
                        cap = cv2.VideoCapture(self.cameras[[cam["name"] for cam in self.cameras].index(camera_name)]["source"])
                        state["cap"] = cap
                        if cap.isOpened():
                            logging.info(f"Reconnexion réussie à la caméra {camera_name}")
                            self.states[camera_name]["status_icon"].setText("🟢")
                            break
                        logging.warning(f"Tentative {attempt + 1}/{retry_attempts} échouée pour {camera_name}")
                        time.sleep(1)
                    if not cap.isOpened():
                        logging.error(f"Impossible de se reconnecter à la caméra {camera_name} après {retry_attempts} tentatives")
                        self.states[camera_name]["status_icon"].setText("🔴")
                        time.sleep(5)
                        continue
                    continue
                if state["frame_queue"].qsize() < 2:
                    state["frame_queue"].put(frame)
            except Exception as e:
                logging.error(f"Erreur dans capture_frames pour {camera_name} : {e}")
                time.sleep(1)
            time.sleep(0.01)

    def process_frames(self, camera_name):
        """Traite les frames pour détecter la somnolence avec une détection optimisée."""
        state = self.states[camera_name]
        frame_skip = 3  # Traiter une frame sur 3 pour réduire la charge
        frame_counter = 0
        min_frames_for_detection = 3  # Nombre minimum de frames consécutifs pour confirmer une détection

        while not state["stop_event"].is_set():
            try:
                frame = state["frame_queue"].get(timeout=1)
                if frame is None:
                    logging.warning(f"Frame vide pour {camera_name}")
                    continue
                frame_counter += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Afficher la frame même si on ne la traite pas
                h, w, ch = image_rgb.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.signals.display_frame_signal.emit(camera_name, convert_to_Qt_format, state["fps"])

                if frame_counter % frame_skip != 0:
                    continue

                results = self.face_mesh.process(image_rgb)
                current_time = time.time()

                if results.multi_face_landmarks:
                    state["last_detection_time"] = current_time
                    for face_landmarks in results.multi_face_landmarks:
                        ih, iw, _ = frame.shape
                        points = []

                        for point_id in self.points_ids:
                            lm = face_landmarks.landmark[point_id]
                            x, y = int(lm.x * iw), int(lm.y * ih)
                            points.append((x, y))

                        if len(points) != 0:
                            x1, y1 = points[0]
                            x2, _ = points[1]
                            _, y3 = points[2]
                            x4, y4 = points[3]
                            x5, y5 = points[4]
                            x6, y6 = points[5]
                            x7, y7 = points[6]

                            x6, x7 = min(x6, x7), max(x6, x7)
                            y6, y7 = min(y6, y7), max(y6, y7)

                            # Extraction des ROI avec validation
                            mouth_roi = frame[y1:y3, x1:x2]
                            right_eye_roi = frame[y4:y5, x4:x5]
                            left_eye_roi = frame[y6:y7, x6:x7]

                            if mouth_roi.size == 0 or right_eye_roi.size == 0 or left_eye_roi.size == 0:
                                logging.warning(f"ROI vide détecté pour {camera_name}: mouth={mouth_roi.shape}, right_eye={right_eye_roi.shape}, left_eye={left_eye_roi.shape}")
                                continue

                            # Prédiction des yeux et du bâillement
                            state["left_eye_state"] = self.predict_eye(left_eye_roi, state["left_eye_state"], camera_name)
                            state["right_eye_state"] = self.predict_eye(right_eye_roi, state["right_eye_state"], camera_name)
                            state["yawn_state"] = self.predict_yawn(mouth_roi, state["yawn_state"], camera_name)

                            # Détection des yeux fermés avec lissage
                            if state["left_eye_state"] == "Close Eye" and state["right_eye_state"] == "Close Eye":
                                state["eye_closed_frames"] += 1
                                if state["eye_closed_frames"] >= min_frames_for_detection:
                                    if not state["left_eye_still_closed"] and not state["right_eye_still_closed"]:
                                        state["left_eye_still_closed"], state["right_eye_still_closed"] = True, True
                                        state["blinks"] += 1
                                    state["microsleeps"] += state["frame_interval"]
                            else:
                                if state["eye_closed_frames"] >= min_frames_for_detection and state["left_eye_still_closed"] and state["right_eye_still_closed"]:
                                    state["left_eye_still_closed"], state["right_eye_still_closed"] = False, False
                                state["eye_closed_frames"] = 0
                                state["microsleeps"] = 0

                            # Détection des bâillements avec lissage
                            if state["yawn_state"] == "Yawn":
                                state["yawn_frames"] += 1
                                if state["yawn_frames"] >= min_frames_for_detection:
                                    if not state["yawn_in_progress"]:
                                        state["yawn_in_progress"] = True
                                        state["yawns"] += 1
                                    state["yawn_duration"] += state["frame_interval"]
                            else:
                                if state["yawn_frames"] >= min_frames_for_detection and state["yawn_in_progress"]:
                                    state["yawn_in_progress"] = False
                                    state["yawn_duration"] = 0
                                state["yawn_frames"] = 0

                            self.update_stats(camera_name)
                else:
                    # Si aucun visage n'est détecté pendant plus de 5 secondes, réinitialiser les compteurs

                    continue

                    # if current_time - state["last_detection_time"] > 5:
                    #     state["eye_closed_frames"] = 0
                    #     state["yawn_frames"] = 0
                    #     state["microsleeps"] = 0
                    #     state["yawn_duration"] = 0
                    #     state["left_eye_still_closed"] = False
                    #     state["right_eye_still_closed"] = False
                    #     state["yawn_in_progress"] = False
                    #     self.update_stats(camera_name)
                    # logging.info(f"Aucun visage détecté pour {camera_name}")
                    # cv2.putText(frame, "Aucun visage detecte", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # h, w, ch = image_rgb.shape
                    # bytes_per_line = ch * w
                    # convert_to_Qt_format = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    # self.signals.display_frame_signal.emit(camera_name, convert_to_Qt_format, state["fps"])

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Erreur dans process_frames pour {camera_name} : {e}")
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                state["stop_event"].set()

    def display_frame(self, camera_name, image, fps):
        """Affiche une frame dans l'interface graphique."""
        if camera_name not in self.states:
            return
        try:
            if image.isNull():
                logging.warning(f"Image nulle pour {camera_name}, impossible d'afficher")
                return
            p = image.scaled(self.states[camera_name]["video_label"].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.states[camera_name]["video_label"].setPixmap(QPixmap.fromImage(p))
            self.states[camera_name]["fps_label"].setText(f"FPS: {int(fps)}")
        except Exception as e:
            logging.error(f"Erreur lors de l'affichage de la frame pour {camera_name} : {e}")

    def play_alert_sound(self):
        """Joue un son d'alerte."""
        if self.alert_sound:
            self.alert_sound.play()

    def play_sound_in_thread(self):
        """Joue le son d'alerte dans un thread séparé."""
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()

    def resizeEvent(self, event):
        """Gère le redimensionnement de la fenêtre."""
        for camera_name in list(self.states.keys()):
            if not self.states[camera_name]["frame_queue"].empty():
                frame = self.states[camera_name]["frame_queue"].get()
                if frame is not None:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = image_rgb.shape
                    bytes_per_line = ch * w
                    convert_to_Qt_format = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.display_frame(camera_name, convert_to_Qt_format, self.states[camera_name]["fps"])
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Gère la fermeture de l'application."""
        for camera_name in list(self.states.keys()):
            state = self.states[camera_name]
            self.update_stats_db(camera_name)
            with open(f"vigilance_stats_{camera_name}.txt", "w") as f:
                f.write(f"Clignements: {state['blinks']}\n")
                f.write(f"Micro-sommeils: {round(state['microsleeps'], 2)} s\n")
                f.write(f"Bâillements: {state['yawns']}\n")
                f.write(f"Durée bâillements: {round(state['yawn_duration'], 2)} s\n")
                f.write(f"Niveau de fatigue: {state['fatigue_level']}%\n")
                f.write(f"Alertes enregistrées pour {camera_name}: {state['alert_count']}\n")
            state["stop_event"].set()
            if state["cap"]:
                state["cap"].release()
        self.db_cursor.close()
        self.db_connection.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(31, 42, 68))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)

    # Liste des caméras
    cameras = [
        {"name": "Caméra 1", "source": 0}
        # {"name": "Caméra 2", "source": "http://192.168.0.110:8080/video"}
        # {"name": "Caméra 3", "source": "http://192.168.0.102:8080/video"}
    ]

    connected_cameras = []
    for camera in cameras:
        logging.info(f"Tentative de connexion à {camera['name']} ({camera['source']})...")
        cap = cv2.VideoCapture(camera["source"])
        if cap.isOpened():
            logging.info(f"Connexion réussie à {camera['name']}")
            ret, frame = cap.read()
            if ret:
                logging.info(f"Frame capturée avec succès pour {camera['name']}. Dimensions: {frame.shape}")
                connected_cameras.append(camera)
            else:
                logging.warning(f"Échec de la capture d'une frame pour {camera['name']}")
            cap.release()
        else:
            logging.error(f"Échec de la connexion à {camera['name']} ({camera['source']})")

    if connected_cameras:
        window = VigilanceCore(connected_cameras)
        window.show()
        sys.exit(app.exec_())
    else:
        logging.error("Aucune caméra connectée. Arrêt du programme.")
        sys.exit(1)