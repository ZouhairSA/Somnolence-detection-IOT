import queue
import threading
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
import mysql.connector
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QProgressBar, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, QObject
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.block import C2f

class VigilanceSignals(QObject):
    """Classe pour émettre des signaux afin de mettre à jour l'interface graphique dans le thread principal."""
    update_ui_signal = pyqtSignal(str, dict)
    display_frame_signal = pyqtSignal(str, QImage, float)
    update_stats_signal = pyqtSignal(str, dict)

class VigilanceCore(QMainWindow):
    def __init__(self, cameras):
        super().__init__()

        # Ajout des globals autorisés pour éviter les erreurs avec Ultralytics
        torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv, C2f])

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
                "last_image_name": "Aucune image capturée",
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
                "recent_alerts": []
            }

        # Initialisation de MediaPipe FaceMesh avec des seuils moins stricts
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.points_ids = [187, 411, 152, 68, 174, 399, 298]

        # Connexion à la base de données MySQL
        try:
            self.db_connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",  # Remplacez par votre mot de passe MySQL si nécessaire
                database="vigilance_db"
            )
            self.db_cursor = self.db_connection.cursor()
            print("Connexion à la base de données réussie")
        except mysql.connector.Error as err:
            print(f"Erreur de connexion à la base de données : {err}")
            sys.exit(1)

        # Compter les alertes existantes pour chaque caméra
        for camera in self.cameras:
            camera_name = camera["name"]
            self.db_cursor.execute("SELECT COUNT(*) FROM alertes WHERE camera_name = %s", (camera_name,))
            self.states[camera_name]["alert_count"] = self.db_cursor.fetchone()[0]

        # Configuration de la fenêtre principale
        self.setWindowTitle("Vigilance Core - Drowsiness Detection")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background: #1E1E2F;")

        # Widget central et layout principal
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(15)

        # En-tête
        self.header_widget = QWidget()
        self.header_layout = QHBoxLayout(self.header_widget)
        self.header_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3B82F6, stop:1 #60A5FA);
            border-radius: 12px;
            padding: 10px;
        """)

        self.header_title = QLabel("Vigilance Core")
        self.header_title.setFont(QFont("Montserrat", 24, QFont.Bold))
        self.header_title.setStyleSheet("color: #FFFFFF; background: none;")
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.header_title)
        self.header_layout.addStretch()

        self.main_layout.addWidget(self.header_widget, stretch=1)

        # Contenu principal
        self.content_widget = QWidget()
        self.content_layout = QHBoxLayout(self.content_widget)
        self.content_layout.setSpacing(15)
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
                background-color: #2D2D44;
                border: 2px solid #3B82F6;
            """)
            video_label.setMinimumSize(320, 240)
            video_label.setScaledContents(True)

            info_widget = QWidget()
            info_layout = QHBoxLayout(info_widget)
            info_layout.setContentsMargins(0, 0, 0, 0)

            name_label = QLabel(camera_name)
            name_label.setFont(QFont("Montserrat", 12, QFont.Bold))
            name_label.setStyleSheet("color: #E5E7EB; background: none;")

            status_icon = QLabel("🔴")
            status_icon.setFont(QFont("Montserrat", 12))
            status_icon.setStyleSheet("background: none;")
            self.states[camera_name]["status_icon"] = status_icon

            fps_label = QLabel("FPS: 0")
            fps_label.setFont(QFont("Montserrat", 10))
            fps_label.setStyleSheet("color: #9CA3AF; background: none;")

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
            background: #2D2D44;
            border-radius: 12px;
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
                    background-color: #1E1E2F;
                    text-align: center;
                    color: #E5E7EB;
                    font-family: 'Montserrat';
                    font-size: 12px;
                }
                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3B82F6, stop:1 #60A5FA);
                    border-radius: 8px;
                }
            """)
            fatigue_bar.setToolTip(f"Niveau de fatigue pour {camera_name}")
            self.fatigue_bars[camera_name] = fatigue_bar
            self.control_layout.addWidget(fatigue_bar)

            status_label = QLabel(f"{camera_name} - État: Optimal")
            status_label.setFont(QFont("Montserrat", 12, QFont.Bold))
            status_label.setStyleSheet("""
                color: #E5E7EB;
                text-align: center;
                padding: 8px;
                background-color: rgba(59, 130, 246, 0.2);
                border-radius: 8px;
            """)
            status_label.setFixedHeight(40)
            self.status_labels[camera_name] = status_label
            self.control_layout.addWidget(status_label)

            alert_label = QLabel("")
            alert_label.setFont(QFont("Montserrat", 12, QFont.Bold))
            alert_label.setStyleSheet("""
                color: #EF4444;
                text-align: center;
                padding: 8px;
                background-color: rgba(239, 68, 68, 0.2);
                border-radius: 8px;
            """)
            alert_label.setFixedHeight(40)
            self.alert_labels[camera_name] = alert_label
            self.control_layout.addWidget(alert_label)

            recent_alerts_label = QLabel("Alertes récentes:\nAucune alerte")
            recent_alerts_label.setFont(QFont("Montserrat", 10))
            recent_alerts_label.setStyleSheet("""
                color: #E5E7EB;
                padding: 8px;
                background-color: rgba(59, 130, 246, 0.1);
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
        self.capture_button.setFont(QFont("Montserrat", 12, QFont.Bold))
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: #FFFFFF;
                padding: 8px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #34D399;
            }
        """)
        self.capture_button.clicked.connect(self.capture_image)
        self.button_layout.addWidget(self.capture_button)

        self.refresh_button = QPushButton("Rafraîchir")
        self.refresh_button.setFont(QFont("Montserrat", 12, QFont.Bold))
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #F59E0B;
                color: #FFFFFF;
                padding: 8px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #FBBF24;
            }
        """)
        self.refresh_button.clicked.connect(self.refresh_stats)
        self.button_layout.addWidget(self.refresh_button)

        self.reset_button = QPushButton("Réinitialiser")
        self.reset_button.setFont(QFont("Montserrat", 12, QFont.Bold))
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: #FFFFFF;
                padding: 8px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #60A5FA;
            }
        """)
        self.reset_button.clicked.connect(self.reset_stats)
        self.button_layout.addWidget(self.reset_button)

        self.quit_button = QPushButton("Arrêt")
        self.quit_button.setFont(QFont("Montserrat", 12, QFont.Bold))
        self.quit_button.setStyleSheet("""
            QPushButton {
                background-color: #EF4444;
                color: #FFFFFF;
                padding: 8px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #F87171;
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
                background-color: #2D2D44;
                color: #E5E7EB;
                border-radius: 8px;
                border: 1px solid #3B82F6;
                font-family: 'Montserrat';
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
                border: none;
            }
            QTableWidget::item:hover {
                background-color: rgba(59, 130, 246, 0.3);
            }
            QHeaderView::section {
                background-color: #3B82F6;
                color: #FFFFFF;
                padding: 8px;
                border: none;
                font-family: 'Montserrat';
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
                    background-color: #EF4444;
                    color: #FFFFFF;
                    padding: 5px;
                    border-radius: 5px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #F87171;
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
            print("Erreur : Les fichiers de modèle YOLO (best.pt) sont introuvables. Utilisation du modèle par défaut.")
            self.detectyawn = YOLO("yolov8n.pt")
            self.detecteye = YOLO("yolov8n.pt")

        # Initialisation des threads pour chaque caméra
        for camera in self.cameras:
            camera_name = camera["name"]
            print(f"Tentative de connexion à la caméra {camera_name} ({camera['source']})...")
            cap = cv2.VideoCapture(camera["source"])
            if not cap.isOpened():
                print(f"Erreur : Impossible de se connecter à la caméra {camera_name} ({camera['source']})")
                self.states[camera_name]["status_icon"].setText("🔴")
                continue
            print(f"Connexion réussie à la caméra {camera_name}")
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
                password="",  # Remplacez par votre mot de passe MySQL si nécessaire
                database="vigilance_db"
            )
            self.db_cursor = self.db_connection.cursor()
            print("Reconnexion à la base de données réussie")
        except mysql.connector.Error as err:
            print(f"Erreur lors de la reconnexion à la base de données : {err}")

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
            print(f"Statistiques mises à jour dans la base de données pour {camera_name}")
        except mysql.connector.Error as err:
            print(f"Erreur lors de la mise à jour des statistiques dans la base de données : {err}")
            self.reconnect_db()
            # Réessayer une fois après reconnexion
            try:
                self.db_cursor.execute(query, values)
                self.db_connection.commit()
                print(f"Statistiques mises à jour après reconnexion pour {camera_name}")
            except mysql.connector.Error as err:
                print(f"Échec de la mise à jour après reconnexion : {err}")

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
                state["last_image_name"] = image_name
                print(f"Image capturée manuellement pour {camera_name}: {image_name}")
            except queue.Empty:
                print(f"Aucune frame disponible pour la capture manuelle ({camera_name})")

    def save_camera_stats_to_db(self, camera_name):
        """Enregistre les statistiques de la caméra dans la table camera_stats avant suppression."""
        state = self.states[camera_name]
        try:
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
            print(f"Statistiques de la caméra {camera_name} enregistrées dans camera_stats")
        except mysql.connector.Error as err:
            print(f"Erreur lors de l'enregistrement des statistiques dans camera_stats : {err}")
            self.reconnect_db()
            try:
                self.db_cursor.execute(query, values)
                self.db_connection.commit()
                print(f"Statistiques enregistrées après reconnexion pour {camera_name}")
            except mysql.connector.Error as err:
                print(f"Échec de l'enregistrement après reconnexion : {err}")

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
                    background-color: #EF4444;
                    color: #FFFFFF;
                    padding: 5px;
                    border-radius: 5px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #F87171;
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
                        item.setBackground(QColor(239, 68, 68, 100))
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
                print(f"Erreur lors de la récupération du nombre d'alertes pour {camera_name} : {err}")
                self.reconnect_db()
                try:
                    self.db_cursor.execute("SELECT COUNT(*) FROM alertes WHERE camera_name = %s", (camera_name,))
                    self.states[camera_name]["alert_count"] = self.db_cursor.fetchone()[0]
                except mysql.connector.Error as err:
                    print(f"Échec de la récupération après reconnexion pour {camera_name} : {err}")
                    self.states[camera_name]["alert_count"] = 0  # Valeur par défaut en cas d'échec

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
            print(f"Alerte enregistrée dans la base de données pour la caméra {camera_name} avec le nom d'image {image_name}")
        except mysql.connector.Error as err:
            print(f"Erreur lors de l'enregistrement dans la base de données : {err}")
            self.reconnect_db()
            # Réessayer une fois après reconnexion
            try:
                self.db_cursor.execute(query, values)
                self.db_connection.commit()
                self.db_cursor.execute("SELECT COUNT(*) FROM alertes WHERE camera_name = %s", (camera_name,))
                self.states[camera_name]["alert_count"] = self.db_cursor.fetchone()[0]
                print(f"Alerte enregistrée après reconnexion pour la caméra {camera_name}")
            except mysql.connector.Error as err:
                print(f"Échec de l'enregistrement après reconnexion : {err}")

    def update_stats(self, camera_name):
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
                    color: #EF4444;
                    text-align: center;
                    padding: 8px;
                    background-color: rgba(239, 68, 68, 0.2);
                    border-radius: 8px;
                    font-size: 12px;
                    font-weight: bold;
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
                print(f"Aucune frame disponible pour la capture ({camera_name})")

        elif round(state["microsleeps"], 2) > 0.2:
            state["alert_text"] = "⚠ Micro-sommeil détecté"
            state["alert_type"] = "Micro-sommeil"
            state["recent_alerts"].append(f"{detection_time}: Micro-sommeil")
            if len(state["recent_alerts"]) > 3:
                state["recent_alerts"].pop(0)
            self.signals.update_ui_signal.emit(camera_name, {
                "status_text": f"{camera_name} - État: Critique",
                "status_style": """
                    color: #EF4444;
                    text-align: center;
                    padding: 8px;
                    background-color: rgba(239, 68, 68, 0.2);
                    border-radius: 8px;
                    font-size: 12px;
                    font-weight: bold;
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
                print(f"Aucune frame disponible pour la capture ({camera_name})")

        else:
            state["alert_text"] = ""
            state["alert_type"] = ""
            self.signals.update_ui_signal.emit(camera_name, {
                "status_text": f"{camera_name} - État: Optimal",
                "status_style": """
                    color: #E5E7EB;
                    text-align: center;
                    padding: 8px;
                    background-color: rgba(59, 130, 246, 0.2);
                    border-radius: 8px;
                    font-size: 12px;
                    font-weight: bold;
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
        if camera_name not in self.alert_blink_states:
            return
        self.alert_blink_states[camera_name] = not self.alert_blink_states[camera_name]
        glow = "border: 2px solid rgba(239, 68, 68, 0.8)" if self.alert_blink_states[camera_name] else "border: 1px solid rgba(239, 68, 68, 0.3)"
        self.alert_labels[camera_name].setStyleSheet(f"""
            color: #EF4444;
            text-align: center;
            padding: 8px;
            background-color: rgba(239, 68, 68, 0.2);
            border-radius: 8px;
            font-size: 12px;
            font-weight: bold;
            {glow};
        """)

    def reset_stats(self):
        for camera_name in list(self.states.keys()):
            state = self.states[camera_name]
            state["blinks"] = 0
            state["microsleeps"] = 0
            state["yawns"] = 0
            state["yawn_duration"] = 0
            state["fatigue_level"] = 0
            self.update_stats(camera_name)

    def refresh_stats(self):
        """Rafraîchit manuellement les statistiques."""
        for camera_name in self.states:
            self.update_stats(camera_name)
            self.update_stats_db(camera_name)

    def predict_eye(self, eye_frame, eye_state):
        try:
            results_eye = self.detecteye.predict(eye_frame)
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
            print(f"Erreur lors de la prédiction des yeux : {e}")
            return eye_state

    def predict_yawn(self, yawn_frame, current_yawn_state):
        try:
            results_yawn = self.detectyawn.predict(yawn_frame)
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
            print(f"Erreur lors de la prédiction du bâillement : {e}")
            return current_yawn_state

    def capture_frames(self, camera_name):
        state = self.states[camera_name]
        cap = state["cap"]
        retry_attempts = 5
        while not state["stop_event"].is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    print(f"Erreur : Perte de connexion avec la caméra {camera_name}. Tentative de reconnexion...")
                    cap.release()
                    for attempt in range(retry_attempts):
                        cap = cv2.VideoCapture(self.cameras[[cam["name"] for cam in self.cameras].index(camera_name)]["source"])
                        state["cap"] = cap
                        if cap.isOpened():
                            print(f"Reconnexion réussie à la caméra {camera_name}")
                            self.states[camera_name]["status_icon"].setText("🟢")
                            break
                        print(f"Tentative {attempt + 1}/{retry_attempts} échouée pour {camera_name}")
                        time.sleep(1)
                    if not cap.isOpened():
                        print(f"Erreur : Impossible de se reconnecter à la caméra {camera_name} après {retry_attempts} tentatives")
                        self.states[camera_name]["status_icon"].setText("🔴")
                        time.sleep(5)
                        continue
                    continue
                if state["frame_queue"].qsize() < 2:
                    state["frame_queue"].put(frame)
            except Exception as e:
                print(f"Erreur dans capture_frames pour {camera_name} : {e}")
                time.sleep(1)
            time.sleep(0.01)

    def process_frames(self, camera_name):
        state = self.states[camera_name]
        timestamp = 0
        frame_skip = 2  # Traiter une frame sur 2 pour réduire la charge
        frame_counter = 0
        while not state["stop_event"].is_set():
            try:
                frame = state["frame_queue"].get(timeout=1)
                if frame is None:
                    print(f"Frame vide pour {camera_name}")
                    continue
                frame_counter += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame_counter % frame_skip != 0:
                    h, w, ch = image_rgb.shape
                    bytes_per_line = ch * w
                    convert_to_Qt_format = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.signals.display_frame_signal.emit(camera_name, convert_to_Qt_format, state["fps"])
                    continue
                timestamp += 1
                results = self.face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
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
                            y6, y7 = min(y6, y7), max(y7, y7)

                            mouth_roi = frame[y1:y3, x1:x2]
                            right_eye_roi = frame[y4:y5, x4:x5]
                            left_eye_roi = frame[y6:y7, x6:x7]

                            if mouth_roi.size == 0 or right_eye_roi.size == 0 or left_eye_roi.size == 0:
                                print(f"ROI vide détecté pour {camera_name}: mouth={mouth_roi.shape}, right_eye={right_eye_roi.shape}, left_eye={left_eye_roi.shape}")
                                continue

                            try:
                                state["left_eye_state"] = self.predict_eye(left_eye_roi, state["left_eye_state"])
                                state["right_eye_state"] = self.predict_eye(right_eye_roi, state["right_eye_state"])
                                state["yawn_state"] = self.predict_yawn(mouth_roi, state["yawn_state"])
                            except Exception as e:
                                print(f"Erreur de prédiction pour {camera_name} : {e}")

                            if state["left_eye_state"] == "Close Eye" and state["right_eye_state"] == "Close Eye":
                                if not state["left_eye_still_closed"] and not state["right_eye_still_closed"]:
                                    state["left_eye_still_closed"], state["right_eye_still_closed"] = True, True
                                    state["blinks"] += 1
                                state["microsleeps"] += 45 / 1000
                            else:
                                if state["left_eye_still_closed"] and state["right_eye_still_closed"]:
                                    state["left_eye_still_closed"], state["right_eye_still_closed"] = False, False
                                state["microsleeps"] = 0

                            if state["yawn_state"] == "Yawn":
                                if not state["yawn_in_progress"]:
                                    state["yawn_in_progress"] = True
                                    state["yawns"] += 1
                                state["yawn_duration"] += 45 / 1000
                            else:
                                if state["yawn_in_progress"]:
                                    state["yawn_in_progress"] = False
                                    state["yawn_duration"] = 0

                            self.update_stats(camera_name)

                            h, w, ch = image_rgb.shape
                            bytes_per_line = ch * w
                            convert_to_Qt_format = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                            self.signals.display_frame_signal.emit(camera_name, convert_to_Qt_format, state["fps"])
                else:
                    print(f"Aucun visage détecté pour {camera_name} à timestamp {timestamp}")
                    cv2.putText(frame, "Aucun visage detecte", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = image_rgb.shape
                    bytes_per_line = ch * w
                    convert_to_Qt_format = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.signals.display_frame_signal.emit(camera_name, convert_to_Qt_format, state["fps"])

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erreur dans process_frames pour {camera_name} : {e}")
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                state["stop_event"].set()

    def display_frame(self, camera_name, image, fps):
        if camera_name not in self.states:
            return
        try:
            if image.isNull():
                print(f"Image nulle pour {camera_name}, impossible d'afficher")
                return
            p = image.scaled(self.states[camera_name]["video_label"].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.states[camera_name]["video_label"].setPixmap(QPixmap.fromImage(p))
            self.states[camera_name]["fps_label"].setText(f"FPS: {int(fps)}")
        except Exception as e:
            print(f"Erreur lors de l'affichage de la frame pour {camera_name} : {e}")

    def play_alert_sound(self):
        frequency = 2200
        duration = 200
        winsound.Beep(frequency, duration)

    def play_sound_in_thread(self):
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()

    def resizeEvent(self, event):
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
    palette.setColor(QPalette.Window, QColor(30, 30, 47))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)

    # Liste des caméras
    cameras = [
        {"name": "SabyoudZOhair_lab", "source": 0}  # Webcam locale
        # {"name": "ZOHAIR_TL", "source": "http://172.16.1.100:8080/video"},  # Caméra IP via IP Webcam sur votre téléphone
        # {"name": "JABRI_TL", "source": "http://192.168.0.102:8080/video"}  # Commenté pour l'instant
    ]

    connected_cameras = []
    for camera in cameras:
        print(f"Tentative de connexion à {camera['name']} ({camera['source']})...")
        cap = cv2.VideoCapture(camera["source"])
        if cap.isOpened():
            print(f"Connexion réussie à {camera['name']}")
            # Test de capture d'une frame pour confirmer que la caméra fonctionne
            ret, frame = cap.read()
            if ret:
                print(f"Frame capturée avec succès pour {camera['name']}. Dimensions: {frame.shape}")
                connected_cameras.append(camera)
            else:
                print(f"Échec de la capture d'une frame pour {camera['name']}")
            cap.release()
        else:
            print(f"Échec de la connexion à {camera['name']} ({camera['source']})")

    if connected_cameras:
        window = VigilanceCore(connected_cameras)
        window.show()
        sys.exit(app.exec_())
    else:
        print("Aucune caméra connectée. Arrêt du programme.")
        sys.exit(1)