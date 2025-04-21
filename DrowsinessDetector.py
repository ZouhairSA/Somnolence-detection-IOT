import queue
import threading
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QProgressBar, QGridLayout
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, pyqtSignal, QObject
import serial
import uuid
from datetime import datetime
import json

try:
    from cassandra_manager import CassandraManager
    CASSANDRA_AVAILABLE = True
except Exception as e:
    print(f"Cassandra non disponible: {str(e)}")
    print("L'application fonctionnera sans stockage Cassandra")
    CASSANDRA_AVAILABLE = False

class SignalEmitter(QObject):
    update_ui = pyqtSignal(dict)
    update_frame = pyqtSignal(np.ndarray)

class VigilanceCore(QMainWindow):
    def __init__(self, use_arduino=False, arduino=None):
        super().__init__()

        # Initialisation de Cassandra si disponible
        self.cassandra = None
        if CASSANDRA_AVAILABLE:
            try:
                self.cassandra = CassandraManager()
                self.session_id = str(uuid.uuid4())
                self.device_id = "camera_device_01"  # Identifiant unique du dispositif
                self.session_start_time = datetime.now()
                print(f"Session ID: {self.session_id}")
                print(f"Démarrage session: {self.session_start_time}")
            except Exception as e:
                print(f"Erreur lors de l'initialisation de Cassandra: {str(e)}")
                self.cassandra = None

        # Ajout des paramètres Arduino
        self.use_arduino = use_arduino
        self.arduino = arduino
        self.buzzer_pin = 9  # Pin du buzzer sur l'Arduino

        # Initialisation des états et compteurs
        self.yawn_state = ''
        self.left_eye_state = ''
        self.right_eye_state = ''
        self.alert_text = ''
        self.fatigue_level = 0

        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_duration = 0
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Seuils de détection améliorés
        self.EYE_AR_THRESH = 0.23  # Seuil plus sensible pour la détection des yeux fermés
        self.EYE_AR_CONSEC_FRAMES = 2  # Réduction du nombre de frames pour une détection plus rapide
        self.YAWN_THRESH = 0.45  # Seuil ajusté pour la détection des bâillements
        self.YAWN_CONSEC_FRAMES = 2  # Réduction du nombre de frames pour les bâillements

        # Nouveaux paramètres de calibration
        self.calibration_mode = False
        self.calibration_samples = []
        self.eye_ar_thresholds = {'left': 0.25, 'right': 0.25}
        self.yawn_thresholds = {'normal': 0.5, 'talking': 0.7}
        self.adaptive_thresholds = True

        # Système de validation des détections
        self.validation_window = 5  # Nombre de frames pour valider une détection
        self.detection_history = {
            'left_eye': [],
            'right_eye': [],
            'yawn': []
        }

        # Amélioration de la détection des micro-sommeils
        self.microsleep_threshold = 0.8  # Seuil pour les micro-sommeils
        self.microsleep_duration = 0.5  # Durée minimale en secondes
        self.microsleep_history = [time.time()]  # Initialisation avec le temps actuel

        # Compensation de la luminosité
        self.brightness_compensation = True
        self.brightness_history = []
        self.brightness_window = 10

        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False

        # Compteurs pour la détection consécutive
        self.left_eye_counter = 0
        self.right_eye_counter = 0
        self.yawn_counter = 0

        # Initialisation de MediaPipe FaceMesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Points d'intérêt pour les yeux et la bouche (plus précis)
        self.points_ids = {
            'left_eye': [33, 160, 158, 133, 153, 144],
            'right_eye': [362, 385, 387, 263, 373, 380],
            'mouth': [61, 291, 0, 17, 269, 405]
        }

        # Initialisation des modèles YOLO avec des seuils optimisés
        self.detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
        self.detecteye = YOLO("runs/detecteye/train/weights/best.pt")
        self.yolo_object = YOLO("yolov8n.pt")

        # Paramètres avancés de détection
        self.detection_params = {
            'eye_conf_threshold': 0.35,      # Augmentation du seuil de confiance pour les yeux
            'yawn_conf_threshold': 0.40,     # Augmentation du seuil de confiance pour les bâillements
            'object_conf_threshold': 0.45,    # Seuil de confiance pour la détection d'objets
            'iou_threshold': 0.45,           # Seuil IOU pour le NMS
            'min_face_size': 50,             # Taille minimale du visage en pixels
            'max_face_size': 400,            # Taille maximale du visage en pixels
            'eye_ar_threshold': 0.23,        # Seuil EAR ajusté
            'yawn_ar_threshold': 0.45,       # Seuil MAR ajusté
            'debug_mode': True               # Mode débogage activé
        }

        # Paramètres avancés de fatigue
        self.fatigue_params = {
            'blink_threshold': 0.25,         # Réduction du seuil pour les clignements
            'microsleep_threshold': 0.6,      # Réduction du seuil pour les micro-sommeils
            'yawn_duration_threshold': 1.2,   # Réduction du seuil pour les bâillements
            'fatigue_window': 30,            # Réduction de la fenêtre d'analyse
            'blink_frequency_threshold': 25,  # Ajustement du seuil de fréquence des clignements
            'head_movement_threshold': 0.15   # Réduction du seuil de mouvement de la tête
        }

        # Historique pour l'analyse de la fatigue
        self.fatigue_history = {
            'blinks': [],
            'yawns': [],
            'microsleeps': [],
            'head_movements': [],
            'timestamps': []
        }

        # Seuils de détection améliorés
        self.MICROSLEEP_ALERT_THRESHOLD = 5.0  # Alerte après 5 secondes de micro-sommeil
        self.YAWN_FREQUENCY_THRESHOLD = 3  # Nombre de bâillements par minute pour alerte
        self.HEAD_POSE_THRESHOLD = 30.0  # Degrés maximum de rotation de la tête
        
        # Points MediaPipe pour la posture de la tête
        self.HEAD_POSE_POINTS = [33, 263, 61, 291, 199]  # Points pour le visage
        
        # Historique pour le suivi temporel
        self.microsleep_start_time = None
        self.last_yawn_time = time.time()
        self.yawn_count_last_minute = 0
        self.head_pose_history = []
        
        # État de la détection
        self.current_state = {
            'eyes_closed': False,
            'yawning': False,
            'head_tilted': False,
            'microsleep_duration': 0.0,
            'alert_level': 0
        }

        # Configuration du dashboard moderne
        self.setWindowTitle("Système de Vigilance au Volant")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet("""
            QMainWindow {
                background: #1e1e2e;
            }
            QWidget {
                font-family: 'Segoe UI';
            }
            QLabel {
                color: #cdd6f4;
                background: transparent;
                border: none;
            }
            QProgressBar {
                border: 2px solid #89b4fa;
                border-radius: 8px;
                background-color: #313244;
                color: #cdd6f4;
                text-align: center;
                font-size: 14px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #89b4fa, stop:1 #f38ba8);
                border-radius: 6px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
        """)

        # Layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Panneau gauche (vidéo)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setStyleSheet("""
            QWidget {
                background-color: #313244;
                border-radius: 15px;
            }
        """)

        # Zone vidéo
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e2e;
                border-radius: 12px;
                padding: 10px;
            }
        """)
        left_layout.addWidget(self.video_label)

        main_layout.addWidget(left_panel, stretch=2)

        # Panneau droit (statistiques)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setStyleSheet("""
            QWidget {
                background-color: #313244;
                border-radius: 15px;
            }
        """)

        # En-tête avec niveau de vigilance
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        
        vigilance_title = QLabel("NIVEAU DE VIGILANCE")
        vigilance_title.setStyleSheet("""
            QLabel {
                color: #89b4fa;
                font-size: 18px;
                font-weight: bold;
                letter-spacing: 2px;
            }
        """)
        header_layout.addWidget(vigilance_title, alignment=Qt.AlignCenter)

        self.fatigue_bar = QProgressBar()
        self.fatigue_bar.setRange(0, 100)
        self.fatigue_bar.setValue(0)
        self.fatigue_bar.setFormat("%p%")
        self.fatigue_bar.setFixedHeight(15)
        header_layout.addWidget(self.fatigue_bar)

        right_layout.addWidget(header_widget)

        # État actuel
        self.status_label = QLabel("ÉTAT NORMAL")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #a6e3a1;
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
                border-radius: 8px;
                background-color: rgba(166, 227, 161, 0.1);
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        # Zone d'alerte
        self.alert_label = QLabel("")
        self.alert_label.setStyleSheet("""
            QLabel {
                color: #f38ba8;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 8px;
                background-color: rgba(243, 139, 168, 0.1);
            }
        """)
        self.alert_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.alert_label)

        # Statistiques principales
        stats_widget = QWidget()
        stats_layout = QGridLayout(stats_widget)
        stats_layout.setSpacing(15)

        self.metrics = {
            "blinks": self.create_stat_widget("CLIGNEMENTS", "0", "👁"),
            "microsleeps": self.create_stat_widget("MICRO-SOMMEILS", "0 s", "💤"),
            "yawns": self.create_stat_widget("BÂILLEMENTS", "0", "😴"),
            "head_pose": self.create_stat_widget("POSITION TÊTE", "Normale", "🔄"),
            "attention": self.create_stat_widget("ATTENTION", "100%", "🎯"),
            "session": self.create_stat_widget("DURÉE SESSION", "00:00", "⏱")
        }

        # Disposition en grille 2x3
        positions = [(i, j) for i in range(2) for j in range(3)]
        for (key, widget), pos in zip(self.metrics.items(), positions):
            stats_layout.addWidget(widget, *pos)

        right_layout.addWidget(stats_widget)

        # Boutons de contrôle
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setSpacing(15)

        self.reset_button = QPushButton("RÉINITIALISER")
        self.reset_button.clicked.connect(self.reset_stats)
        
        self.quit_button = QPushButton("QUITTER")
        self.quit_button.clicked.connect(self.close)
        self.quit_button.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
            }
            QPushButton:hover {
                background-color: #f5c2e7;
            }
        """)
        
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.quit_button)
        
        right_layout.addWidget(buttons_widget)
        main_layout.addWidget(right_panel, stretch=1)

        # Timer pour la mise à jour de la durée de session
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_session_duration)
        self.session_timer.start(1000)  # Mise à jour chaque seconde
        self.session_start_time = time.time()

        # Test des caméras disponibles et sélection de la caméra USB externe
        available_cameras = []
        for i in range(10):  # Teste les 10 premiers indices de caméra
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Utilise DirectShow pour Windows
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        available_cameras.append(i)
                        print(f"Caméra {i} disponible")
                    cap.release()
            except Exception as e:
                print(f"Erreur avec la caméra {i}: {str(e)}")
        
        if len(available_cameras) > 1:
            # Si plusieurs caméras sont disponibles, utilise la dernière (souvent la caméra USB externe)
            camera_index = available_cameras[-1]
            print(f"Utilisation de la caméra externe (index {camera_index})")
        elif len(available_cameras) == 1:
            camera_index = available_cameras[0]
            print(f"Utilisation de la seule caméra disponible (index {camera_index})")
        else:
            print("Aucune caméra trouvée!")
            sys.exit(1)

        # Capture vidéo avec la caméra sélectionnée
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        # Configuration de la résolution de la caméra
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Vérification de l'ouverture de la caméra
        if not self.cap.isOpened():
            print("Erreur: Impossible d'ouvrir la caméra")
            sys.exit(1)

        time.sleep(1.0)  # Attendre l'initialisation de la caméra

        # Gestion des threads
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        self.capture_thread.start()
        self.process_thread.start()

        # Timer pour animations
        self.alert_timer = QTimer(self)
        self.alert_timer.timeout.connect(self.toggle_alert_glow)
        self.alert_blink_state = False

        # Animation pour la barre de fatigue
        self.fatigue_animation = QPropertyAnimation(self.fatigue_bar, b"value")
        self.fatigue_animation.setEasingCurve(QEasingCurve.InOutQuad)

        # Initialisation du signal emitter
        self.signal_emitter = SignalEmitter()
        self.signal_emitter.update_ui.connect(self.update_ui_from_signal)
        self.signal_emitter.update_frame.connect(self.update_frame_from_signal)

    def create_stat_widget(self, title, initial_value, icon):
        """Crée un widget de statistique élégant"""
        widget = QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)

        # Titre avec icône
        title_label = QLabel(f"{icon} {title}")
        title_label.setStyleSheet("""
            QLabel {
                color: #89b4fa;
                font-size: 12px;
                font-weight: bold;
                letter-spacing: 1px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)

        # Valeur
        value_label = QLabel(initial_value)
        value_label.setStyleSheet("""
            QLabel {
                color: #cdd6f4;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        value_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(value_label)

        widget.value_label = value_label
        return widget

    def update_session_duration(self):
        """Met à jour la durée de la session"""
        duration = int(time.time() - self.session_start_time)
        minutes = duration // 60
        seconds = duration % 60
        self.metrics["session"].value_label.setText(f"{minutes:02d}:{seconds:02d}")

    def update_ui_from_signal(self, data):
        """Met à jour l'interface utilisateur depuis le thread principal"""
        if 'fatigue_level' in data:
            new_fatigue_level = data['fatigue_level']
            if new_fatigue_level != self.fatigue_level:
                self.fatigue_animation = QPropertyAnimation(self.fatigue_bar, b"value")
                self.fatigue_animation.setStartValue(self.fatigue_level)
                self.fatigue_animation.setEndValue(new_fatigue_level)
                self.fatigue_animation.setDuration(500)
                self.fatigue_animation.start()
            self.fatigue_level = new_fatigue_level

            # Mise à jour du niveau d'attention
            attention = max(0, 100 - new_fatigue_level)
            self.metrics["attention"].value_label.setText(f"{attention}%")

        if 'alert_text' in data:
            self.alert_text = data['alert_text']
            self.alert_label.setText(self.alert_text)
            if self.alert_text:
                self.alert_label.setStyleSheet("""
                    QLabel {
                        color: #f38ba8;
                        font-size: 16px;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 8px;
                        background-color: rgba(243, 139, 168, 0.1);
                    }
                """)
            else:
                self.alert_label.setStyleSheet("""
                    QLabel {
                        color: #f38ba8;
                        font-size: 16px;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 8px;
                        background-color: transparent;
                    }
                """)

        if 'status' in data:
            self.status_label.setText(data['status'].upper())
            if "CRITIQUE" in data['status']:
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #f38ba8;
                        font-size: 24px;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 8px;
                        background-color: rgba(243, 139, 168, 0.1);
                    }
                """)
            elif "ATTENTION" in data['status']:
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #fab387;
                        font-size: 24px;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 8px;
                        background-color: rgba(250, 179, 135, 0.1);
                    }
                """)
            else:
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #a6e3a1;
                        font-size: 24px;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 8px;
                        background-color: rgba(166, 227, 161, 0.1);
                    }
                """)

        if 'metrics' in data:
            metrics = data['metrics']
            for key, value in metrics.items():
                if key in self.metrics:
                    self.metrics[key].value_label.setText(value)

    def update_frame_from_signal(self, frame):
        """Met à jour l'affichage de la frame depuis le thread principal"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def update_stats(self):
        new_fatigue_level = min(100, int((self.microsleeps + self.yawn_duration) * 10))
        if new_fatigue_level != self.fatigue_level:
            self.fatigue_animation.setStartValue(self.fatigue_level)
            self.fatigue_animation.setEndValue(new_fatigue_level)
            self.fatigue_animation.setDuration(500)
            self.fatigue_animation.start()
        self.fatigue_level = new_fatigue_level

        if round(self.yawn_duration, 2) > 0.5:
            self.alert_text = "⚠ Alerte: Bâillement prolongé"
            self.status_label.setText("État: Attention")
            self.status_label.setStyleSheet("""
                color: #B22222;
                text-align: center;
                padding: 12px;
                background-color: rgba(178, 34, 34, 0.2);
                border-radius: 8px;
                box-shadow: 0 0 5px rgba(178, 34, 34, 0.3);
                font-size: 18px;
                font-weight: bold;
            """)
            self.alert_timer.start(400)
            self.play_sound_in_thread()

        elif round(self.microsleeps, 2) > 0.5:
            self.alert_text = "⚠ Alerte: Micro-sommeil détecté"
            self.status_label.setText("État: Critique")
            self.status_label.setStyleSheet("""
                color: #B22222;
                text-align: center;
                padding: 12px;
                background-color: rgba(178, 34, 34, 0.2);
                border-radius: 8px;
                box-shadow: 0 0 5px rgba(178, 34, 34, 0.3);
                font-size: 18px;
                font-weight: bold;
            """)
            self.alert_timer.start(400)
            self.play_sound_in_thread()

        else:
            self.alert_text = ""
            self.status_label.setText("État: Normal")
            self.status_label.setStyleSheet("""
                color: #FFFFFF;
                text-align: center;
                padding: 12px;
                background-color: rgba(70, 130, 180, 0.2);
                border-radius: 8px;
                box-shadow: 0 0 5px rgba(255, 255, 255, 0.1);
                font-size: 18px;
                font-weight: bold;
            """)
            self.alert_timer.stop()

        self.alert_label.setText(self.alert_text)
        self.metrics["blinks"].value_label.setText(f"👁 Clignements: {self.blinks}")
        self.metrics["microsleeps"].value_label.setText(f"💤 Micro-sommeils: {round(self.microsleeps, 2)} s")
        self.metrics["yawns"].value_label.setText(f"😴 Bâillements: {self.yawns}")
        self.metrics["yawn_duration"].value_label.setText(f"⏲ Durée bâillements: {round(self.yawn_duration, 2)} s")
        self.metrics["fps"].value_label.setText(f"📈 FPS: {round(self.fps, 1)}")

        # Calcul des FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.metrics["fps"].value_label.setText(f"📈 FPS: {round(self.fps, 1)}")
            self.frame_count = 0
            self.start_time = time.time()

        # Envoyer l'état à l'Arduino si nécessaire
        if self.use_arduino and self.arduino:
            try:
                if self.fatigue_level > 0:
                    self.arduino.write(b'S')  # S pour Somnolence
                else:
                    self.arduino.write(b'A')  # A pour Attentif
            except serial.SerialException:
                print("Erreur de communication avec l'Arduino")
                self.use_arduino = False

    def toggle_alert_glow(self):
        """Alterne l'effet de surbrillance de l'alerte"""
        self.alert_blink_state = not self.alert_blink_state
        background_color = "rgba(178, 34, 34, 0.4)" if self.alert_blink_state else "rgba(178, 34, 34, 0.2)"
        self.alert_label.setStyleSheet(f"""
            color: #B22222;
            text-align: center;
            padding: 10px;
            background-color: {background_color};
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
        """)

    def reset_stats(self):
        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_duration = 0
        self.fatigue_level = 0
        self.update_stats()

    def predict_eye(self, eye_frame, eye_state):
        """Prédit l'état de l'œil avec amélioration de la détection"""
        try:
            # Prétraitement amélioré de l'image
            eye_frame = cv2.resize(eye_frame, (64, 64))
            eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB)
            
            # Amélioration du contraste
            eye_frame = cv2.convertScaleAbs(eye_frame, alpha=1.2, beta=10)
            
            results_eye = self.detecteye.predict(
                eye_frame,
                conf=self.detection_params['eye_conf_threshold'],
                iou=self.detection_params['iou_threshold']
            )
            
            boxes = results_eye[0].boxes
            
            if len(boxes) == 0:
                return eye_state

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_confidence_index = np.argmax(confidences)
            class_id = int(class_ids[max_confidence_index])
            confidence = confidences[max_confidence_index]

            # Seuils de confiance ajustés avec validation temporelle
            if class_id == 1 and confidence > self.detection_params['eye_conf_threshold']:  # Œil fermé
                if eye_state == "Close Eye" or confidence > 0.45:  # Validation temporelle
                    eye_state = "Close Eye"
            elif class_id == 0 and confidence > self.detection_params['eye_conf_threshold']:  # Œil ouvert
                if eye_state == "Open Eye" or confidence > 0.45:  # Validation temporelle
                    eye_state = "Open Eye"
            
            return eye_state
        except Exception as e:
            print(f"Erreur lors de la prédiction de l'œil: {e}")
            return eye_state

    def predict_yawn(self, yawn_frame):
        """Prédit l'état du bâillement avec amélioration de la détection"""
        try:
            # Prétraitement amélioré de l'image
            yawn_frame = cv2.resize(yawn_frame, (64, 64))
            yawn_frame = cv2.cvtColor(yawn_frame, cv2.COLOR_BGR2RGB)
            
            # Amélioration du contraste
            yawn_frame = cv2.convertScaleAbs(yawn_frame, alpha=1.2, beta=10)
            
            results_yawn = self.detectyawn.predict(
                yawn_frame,
                conf=self.detection_params['yawn_conf_threshold'],
                iou=self.detection_params['iou_threshold']
            )
            
            boxes = results_yawn[0].boxes

            if len(boxes) == 0:
                return self.yawn_state

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_confidence_index = np.argmax(confidences)
            class_id = int(class_ids[max_confidence_index])
            confidence = confidences[max_confidence_index]

            # Seuils de confiance ajustés avec validation temporelle
            if class_id == 0 and confidence > self.detection_params['yawn_conf_threshold']:  # Bâillement
                if self.yawn_state == "Yawn" or confidence > 0.5:  # Validation temporelle
                    self.yawn_state = "Yawn"
            elif class_id == 1 and confidence > self.detection_params['yawn_conf_threshold']:  # Pas de bâillement
                if self.yawn_state == "No Yawn" or confidence > 0.5:  # Validation temporelle
                    self.yawn_state = "No Yawn"
            
        except Exception as e:
            print(f"Erreur lors de la prédiction du bâillement: {e}")

    def detect_objects(self, frame):
        """Détection améliorée des objets avec YOLOv8"""
        results = self.yolo_object.predict(
            frame,
            conf=self.detection_params['object_conf_threshold'],
            iou=self.detection_params['iou_threshold'],
            classes=[0, 67, 68]  # Personne, téléphone portable, téléphone
        )

        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf)
                class_id = int(box.cls)
                label = result.names[class_id]

                # Dessiner les boîtes et les étiquettes
                color = (0, 255, 0) if label == "person" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detected_objects.append({
                    'label': label,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })

        return frame, detected_objects

    def analyze_eye_state(self, eye_points, frame):
        """Analyse améliorée de l'état des yeux avec détection détaillée"""
        ear = self.eye_aspect_ratio(eye_points)
        
        # Extraction et amélioration de la région de l'œil
        x, y = np.min(eye_points, axis=0)
        w, h = np.max(eye_points, axis=0) - np.min(eye_points, axis=0)
        eye_region = frame[int(y):int(y+h), int(x):int(x+w)]
        
        if eye_region.size == 0:
            return 'unknown', 0.0, {'state': 'unknown', 'confidence': 0.0, 'details': 'Region non détectée'}

        # Amélioration du contraste de la région de l'œil
        eye_region = cv2.convertScaleAbs(eye_region, alpha=1.2, beta=10)
        
        # Analyse de la luminosité locale avec compensation
        brightness = np.mean(eye_region)
        contrast = np.std(eye_region)
        
        # Ajustement dynamique du seuil basé sur la luminosité
        brightness_factor = 1.0 + (128 - brightness) / 256.0
        dynamic_threshold = self.detection_params['eye_ar_threshold'] * brightness_factor
        
        # Classification détaillée avec compensation de luminosité
        if ear < dynamic_threshold * 0.8:  # Très fermé
            confidence = min(1.0, (dynamic_threshold - ear) / dynamic_threshold * 1.2)
            return 'closed', confidence, {
                'state': 'closed',
                'confidence': confidence,
                'details': 'Œil complètement fermé',
                'brightness': brightness,
                'contrast': contrast,
                'ear': ear,
                'threshold': dynamic_threshold
            }
        elif ear < dynamic_threshold:  # Partiellement fermé
            confidence = (dynamic_threshold - ear) / dynamic_threshold
            return 'partially_closed', confidence, {
                'state': 'partially_closed',
                'confidence': confidence,
                'details': 'Œil partiellement fermé',
                'brightness': brightness,
                'contrast': contrast,
                'ear': ear,
                'threshold': dynamic_threshold
            }
        else:  # Ouvert
            confidence = min(1.0, (ear - dynamic_threshold) / (1.0 - dynamic_threshold) * 1.2)
            return 'open', confidence, {
                'state': 'open',
                'confidence': confidence,
                'details': 'Œil ouvert',
                'brightness': brightness,
                'contrast': contrast,
                'ear': ear,
                'threshold': dynamic_threshold
            }

    def mouth_aspect_ratio(self, mouth_points):
        """Calcule le ratio d'aspect de la bouche (MAR) avec plus de précision"""
        # Points supplémentaires pour une meilleure détection
        vertical_distances = [
            np.linalg.norm(mouth_points[1] - mouth_points[5]),
            np.linalg.norm(mouth_points[2] - mouth_points[4])
        ]
        horizontal_distance = np.linalg.norm(mouth_points[0] - mouth_points[3])
        
        # Calcul du ratio avec moyenne pondérée
        mar = (sum(vertical_distances) * 0.5) / (horizontal_distance + 1e-6)
        
        if self.detection_params['debug_mode']:
            print(f"MAR: {mar:.3f}")
            
        return mar

    def detect_head_pose(self, face_landmarks):
        """Détecte la posture de la tête"""
        try:
            if not face_landmarks:
                return 0, 0, 0

            # Points 3D du modèle
            model_points = np.array([
                (0.0, 0.0, 0.0),          # Nez
                (0.0, -330.0, -65.0),     # Menton
                (-225.0, 170.0, -135.0),  # Œil gauche
                (225.0, 170.0, -135.0),   # Œil droit
                (-150.0, -150.0, -125.0), # Bouche gauche
                (150.0, -150.0, -125.0)   # Bouche droite
            ])

            # Points 2D du visage
            image_points = np.array([
                (face_landmarks.landmark[4].x, face_landmarks.landmark[4].y),   # Nez
                (face_landmarks.landmark[152].x, face_landmarks.landmark[152].y), # Menton
                (face_landmarks.landmark[33].x, face_landmarks.landmark[33].y),   # Œil gauche
                (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y), # Œil droit
                (face_landmarks.landmark[61].x, face_landmarks.landmark[61].y),   # Bouche gauche
                (face_landmarks.landmark[291].x, face_landmarks.landmark[291].y)  # Bouche droite
            ], dtype=np.float32)

            # Matrice de la caméra
            size = self.current_frame.shape
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float32
            )

            dist_coeffs = np.zeros((4,1))
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )

            # Conversion en angles d'Euler
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            return euler_angles[0][0], euler_angles[1][0], euler_angles[2][0]  # pitch, yaw, roll
            
        except Exception as e:
            print(f"Erreur dans la détection de la posture: {str(e)}")
            return 0, 0, 0

    def check_head_pose_alert(self, pitch, yaw, roll):
        """Vérifie si la posture de la tête nécessite une alerte"""
        if abs(pitch) > self.HEAD_POSE_THRESHOLD or \
           abs(yaw) > self.HEAD_POSE_THRESHOLD or \
           abs(roll) > self.HEAD_POSE_THRESHOLD:
            self.current_state['head_tilted'] = True
            return True
        self.current_state['head_tilted'] = False
        return False

    def update_microsleep_status(self, eyes_closed):
        """Met à jour le statut des micro-sommeils"""
        current_time = time.time()
        
        if eyes_closed:
            if self.microsleep_start_time is None:
                self.microsleep_start_time = current_time
            
            duration = current_time - self.microsleep_start_time
            self.current_state['microsleep_duration'] = duration
            
            if duration >= self.MICROSLEEP_ALERT_THRESHOLD:
                self.alert_text = f"⚠ ALERTE CRITIQUE: Micro-sommeil détecté ({duration:.1f}s)"
                self.current_state['alert_level'] = 3  # Niveau critique
                self.play_alert_sound()
                if self.cassandra:
                    self.log_event("microsleep", "critical", f"Micro-sommeil de {duration:.1f} secondes")
        else:
            if self.microsleep_start_time is not None:
                duration = current_time - self.microsleep_start_time
                if duration > 0.5:  # Enregistrer seulement les micro-sommeils > 0.5s
                    self.microsleeps += duration
                self.microsleep_start_time = None
            self.current_state['microsleep_duration'] = 0.0

    def update_yawn_status(self, is_yawning):
        """Met à jour le statut des bâillements"""
        current_time = time.time()
        
        if is_yawning:
            if current_time - self.last_yawn_time > 60:  # Réinitialiser le compteur après 1 minute
                self.yawn_count_last_minute = 0
            
            self.yawn_count_last_minute += 1
            self.last_yawn_time = current_time
            
            if self.yawn_count_last_minute >= self.YAWN_FREQUENCY_THRESHOLD:
                self.alert_text = f"⚠ ALERTE: Bâillements fréquents ({self.yawn_count_last_minute} en 1 minute)"
                self.current_state['alert_level'] = 2  # Niveau élevé
                self.play_alert_sound()
                if self.cassandra:
                    self.log_event("yawning", "high", f"{self.yawn_count_last_minute} bâillements en 1 minute")

    def update_fatigue_level(self, eyes_closed, yawning, head_tilted, microsleep_duration):
        """Calcul amélioré du niveau de fatigue"""
        base_score = 0
        
        # Contribution des micro-sommeils (40%)
        if microsleep_duration > self.MICROSLEEP_ALERT_THRESHOLD:
            base_score += 40
        elif microsleep_duration > 0:
            base_score += (microsleep_duration / self.MICROSLEEP_ALERT_THRESHOLD) * 40

        # Contribution des bâillements (30%)
        if self.yawn_count_last_minute >= self.YAWN_FREQUENCY_THRESHOLD:
            base_score += 30
        elif yawning:
            base_score += 15

        # Contribution de la posture de la tête (20%)
        if head_tilted:
            base_score += 20

        # Contribution des yeux fermés (10%)
        if eyes_closed:
            base_score += 10

        # Mise à jour progressive du niveau de fatigue
        self.fatigue_level = min(100, int((self.fatigue_level + base_score) / 2))

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame)
            else:
                break

    def eye_aspect_ratio(self, eye_points):
        """Calcule le ratio d'aspect de l'œil (EAR)"""
        # Calcul des distances verticales
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Calcul de la distance horizontale
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calcul du ratio
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def process_frames(self):
        """Traitement amélioré des frames avec multiples détections"""
        while True:
            if not hasattr(self, 'frame_queue') or self.frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = self.frame_queue.get()
            self.current_frame = frame.copy()
            
            # 1. Détection d'objets avec YOLO
            frame, detected_objects = self.detect_objects(frame)

            # 2. Analyse MediaPipe pour les points du visage
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Détection de la posture de la tête
                    pitch, yaw, roll = self.detect_head_pose(face_landmarks)
                    head_pose_alert = self.check_head_pose_alert(pitch, yaw, roll)

                    # Points des yeux et de la bouche
                    left_eye_points = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                               face_landmarks.landmark[i].y * frame.shape[0]]
                                              for i in self.points_ids['left_eye']])
                    right_eye_points = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                                face_landmarks.landmark[i].y * frame.shape[0]]
                                               for i in self.points_ids['right_eye']])
                    mouth_points = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                           face_landmarks.landmark[i].y * frame.shape[0]]
                                          for i in self.points_ids['mouth']])

                    # Analyse des yeux
                    left_eye_state, left_conf, left_details = self.analyze_eye_state(left_eye_points, frame)
                    right_eye_state, right_conf, right_details = self.analyze_eye_state(right_eye_points, frame)
                    
                    # Mise à jour du statut des yeux
                    eyes_closed = (left_eye_state == 'closed' and right_eye_state == 'closed')
                    self.update_microsleep_status(eyes_closed)

                    # Analyse de la bouche et des bâillements
                    mar = self.mouth_aspect_ratio(mouth_points)
                    is_yawning = mar > self.detection_params['yawn_ar_threshold']
                    self.update_yawn_status(is_yawning)

                    # Mise à jour du niveau de fatigue global
                    self.update_fatigue_level(
                        eyes_closed=eyes_closed,
                        yawning=is_yawning,
                        head_tilted=head_pose_alert,
                        microsleep_duration=self.current_state['microsleep_duration']
                    )

                    # Affichage des informations
                    self.display_debug_info(frame, {
                        'left_eye': left_details,
                        'right_eye': right_details,
                        'yawn': {'mar': mar, 'is_yawning': is_yawning},
                        'head_pose': {'pitch': pitch, 'yaw': yaw, 'roll': roll},
                        'microsleep': self.current_state['microsleep_duration'],
                        'alert_level': self.current_state['alert_level']
                    })

            # Mise à jour de l'interface
            self.signal_emitter.update_frame.emit(frame)
            self.signal_emitter.update_ui.emit({
                'fatigue_level': self.fatigue_level,
                'alert_text': self.alert_text,
                'status': self.get_status_message(),
                'metrics': self.get_metrics_data()
            })

    def display_eye_info(self, frame, left_details, right_details):
        """Affiche les informations détaillées sur les yeux"""
        try:
            # Informations sur l'œil gauche
            left_text = f"Gauche: {left_details['state']} ({left_details['confidence']:.2f})"
            cv2.putText(frame, left_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Informations sur l'œil droit
            right_text = f"Droit: {right_details['state']} ({right_details['confidence']:.2f})"
            cv2.putText(frame, right_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Détails supplémentaires
            details_text = f"Luminosité: {left_details['brightness']:.1f} | Contraste: {left_details['contrast']:.1f}"
            cv2.putText(frame, details_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Erreur lors de l'affichage des informations des yeux: {e}")

    def get_alert_message(self):
        """Génère un message d'alerte personnalisé"""
        if self.fatigue_level > 75:
            return "⚠ ALERTE CRITIQUE : Niveau de fatigue très élevé!"
        elif self.fatigue_level > 50:
            return "⚠ Attention : Signes de fatigue détectés"
        elif self.microsleeps > 0:
            return "⚠ Attention : Micro-sommeil détecté"
        elif self.yawn_duration > 1.5:
            return "⚠ Attention : Bâillement prolongé"
        return ""

    def get_status_message(self):
        """Retourne le message de statut en français"""
        if self.fatigue_level > 75:
            return "ÉTAT CRITIQUE"
        elif self.fatigue_level > 50:
            return "ATTENTION REQUISE"
        return "ÉTAT NORMAL"

    def get_metrics_data(self):
        """Prépare les données des métriques en français"""
        return {
            'blinks': str(self.blinks),
            'microsleeps': f"{round(self.microsleeps, 1)} s",
            'yawns': str(self.yawns),
            'head_pose': "Normale" if not self.current_state.get('head_tilted', False) else "Inclinée",
            'attention': f"{max(0, 100 - self.fatigue_level)}%"
        }

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def play_alert_sound(self):
        """Joue un son d'alerte via le buzzer"""
        if self.use_arduino and self.arduino:
            try:
                # Envoi de la commande au buzzer
                self.arduino.write(b'B\n')  # Commande pour activer le buzzer
                time.sleep(0.2)  # Durée du son
                self.arduino.write(b'b\n')  # Commande pour désactiver le buzzer
            except Exception as e:
                print(f"Erreur lors de l'activation du buzzer: {str(e)}")
        else:
            # Fallback sur le buzzer du PC
            frequency = 2200
            duration = 200
            winsound.Beep(frequency, duration)

    def log_event(self, event_type, confidence, details):
        """Enregistre un événement dans Cassandra"""
        if self.cassandra:
            try:
                event_id = str(uuid.uuid4())
                timestamp = datetime.now()
                
                # Conversion des détails en chaîne JSON
                details_json = json.dumps(details)
                
                # Insertion dans la table events
                query = """
                    INSERT INTO vigilance_db.events 
                    (event_id, event_type, device_id, session_id, confidence, details, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                self.cassandra.session.execute(query, (
                    event_id,
                    event_type,
                    self.device_id,
                    self.session_id,
                    float(confidence),
                    details_json,
                    timestamp
                ))
                
                print(f"Événement enregistré: {event_type} à {timestamp}")
            except Exception as e:
                print(f"Erreur lors de l'enregistrement de l'événement: {str(e)}")

    def log_alert(self, alert_type, severity, message):
        """Enregistre une alerte dans Cassandra"""
        if self.cassandra:
            try:
                alert_id = str(uuid.uuid4())
                timestamp = datetime.now()
                
                # Insertion dans la table alerts
                query = """
                    INSERT INTO vigilance_db.alerts 
                    (alert_id, alert_type, device_id, session_id, severity, message, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                self.cassandra.session.execute(query, (
                    alert_id,
                    alert_type,
                    self.device_id,
                    self.session_id,
                    severity,
                    message,
                    timestamp
                ))
                
                print(f"Alerte enregistrée: {alert_type} - {message} à {timestamp}")
            except Exception as e:
                print(f"Erreur lors de l'enregistrement de l'alerte: {str(e)}")

    def update_session_stats(self):
        """Met à jour les statistiques de la session dans Cassandra"""
        if self.cassandra:
            try:
                end_time = datetime.now()
                duration = (end_time - self.session_start_time).total_seconds()
                
                # Calcul de la moyenne du niveau de fatigue
                avg_fatigue = self.fatigue_level  # À améliorer avec une vraie moyenne
                
                # Insertion dans la table session_stats
                query = """
                    INSERT INTO vigilance_db.session_stats 
                    (session_id, device_id, start_time, end_time, total_blinks, 
                     total_yawns, total_microsleeps, max_fatigue_level, avg_fatigue_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                self.cassandra.session.execute(query, (
                    self.session_id,
                    self.device_id,
                    self.session_start_time,
                    end_time,
                    self.blinks,
                    self.yawns,
                    float(self.microsleeps),
                    self.fatigue_level,
                    avg_fatigue
                ))
                
                print(f"Statistiques de session mises à jour pour {self.session_id}")
            except Exception as e:
                print(f"Erreur lors de la mise à jour des statistiques de session: {str(e)}")

    def resizeEvent(self, event):
        self.display_frame(self.frame_queue.get() if not self.frame_queue.empty() else np.zeros((480, 640, 3), dtype=np.uint8))
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Gestion améliorée de la fermeture de l'application"""
        try:
            # Mise à jour des statistiques finales
            self.update_session_stats()
            
            # Fermeture de la connexion Cassandra
            if self.cassandra:
                self.cassandra.close()
            
            # Fermeture de la connexion Arduino
            if self.arduino:
                self.arduino.close()
            
            # Sauvegarde des statistiques locales
            with open("vigilance_stats.txt", "w") as f:
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Début de session: {self.session_start_time}\n")
                f.write(f"Fin de session: {datetime.now()}\n")
                f.write(f"Clignements: {self.blinks}\n")
                f.write(f"Micro-sommeils: {round(self.microsleeps, 2)} s\n")
                f.write(f"Bâillements: {self.yawns}\n")
                f.write(f"Durée bâillements: {round(self.yawn_duration, 2)} s\n")
                f.write(f"Niveau de fatigue final: {self.fatigue_level}%\n")
            
            # Arrêt des threads
            self.stop_event.set()
            
            # Fermeture de la caméra
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            
            print("Application fermée proprement")
            event.accept()
        except Exception as e:
            print(f"Erreur lors de la fermeture: {str(e)}")
            event.accept()

    def display_debug_info(self, frame, debug_data):
        """Affiche les informations de débogage sur la frame"""
        try:
            # Informations sur les yeux
            cv2.putText(frame, f"Left Eye: {debug_data['left_eye']['state']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Eye: {debug_data['right_eye']['state']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Information sur le bâillement
            cv2.putText(frame, f"Yawn: {'Yes' if debug_data['yawn']['is_yawning'] else 'No'}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Informations sur la posture de la tête
            head_pose = debug_data['head_pose']
            cv2.putText(frame, f"Head: P:{head_pose['pitch']:.1f} Y:{head_pose['yaw']:.1f} R:{head_pose['roll']:.1f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Informations sur les micro-sommeils
            cv2.putText(frame, f"Microsleep: {debug_data['microsleep']:.1f}s", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Niveau d'alerte
            alert_color = (0, 255, 0)  # Vert par défaut
            if debug_data['alert_level'] == 3:
                alert_color = (0, 0, 255)  # Rouge pour niveau critique
            elif debug_data['alert_level'] == 2:
                alert_color = (0, 165, 255)  # Orange pour niveau élevé
            
            cv2.putText(frame, f"Alert Level: {debug_data['alert_level']}", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
            
            # Compteurs
            cv2.putText(frame, f"Blinks: {self.blinks}", (10, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yawns: {self.yawns}", (10, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # FPS
            cv2.putText(frame, f"FPS: {round(self.fps, 1)}", (10, 270),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Erreur lors de l'affichage des informations de débogage: {e}")

    def start_calibration(self):
        """Démarre le mode de calibration"""
        self.calibration_mode = True
        self.calibration_samples = []
        self.status_label.setText("Calibration en cours...")
        self.alert_label.setText("Regardez la caméra normalement")

    def end_calibration(self):
        """Termine la calibration et ajuste les seuils"""
        if len(self.calibration_samples) > 0:
            # Calcul des seuils moyens
            left_eye_avg = np.mean([sample['left_eye'] for sample in self.calibration_samples])
            right_eye_avg = np.mean([sample['right_eye'] for sample in self.calibration_samples])
            yawn_avg = np.mean([sample['yawn'] for sample in self.calibration_samples])

            # Ajustement des seuils avec une marge de sécurité
            self.eye_ar_thresholds['left'] = left_eye_avg * 0.8
            self.eye_ar_thresholds['right'] = right_eye_avg * 0.8
            self.yawn_thresholds['normal'] = yawn_avg * 1.2
            self.yawn_thresholds['talking'] = yawn_avg * 1.5

        self.calibration_mode = False
        self.status_label.setText("Calibration terminée")
        self.alert_label.setText("")

    def collect_calibration_sample(self, left_eye_ar, right_eye_ar, yawn_ar):
        """Collecte un échantillon pour la calibration"""
        if self.calibration_mode:
            self.calibration_samples.append({
                'left_eye': left_eye_ar,
                'right_eye': right_eye_ar,
                'yawn': yawn_ar
            })
            if len(self.calibration_samples) >= 30:  # 30 échantillons suffisants
                self.end_calibration()

    def adjust_thresholds(self, current_conditions):
        """Ajuste dynamiquement les seuils en fonction des conditions"""
        if not self.adaptive_thresholds:
            return

        # Ajustement basé sur la luminosité
        brightness = np.mean(self.current_frame)
        if brightness < 50:  # Conditions sombres
            self.eye_ar_thresholds['left'] *= 0.9
            self.eye_ar_thresholds['right'] *= 0.9
        elif brightness > 200:  # Conditions très lumineuses
            self.eye_ar_thresholds['left'] *= 1.1
            self.eye_ar_thresholds['right'] *= 1.1

        # Ajustement basé sur la distance à la caméra
        face_size = current_conditions.get('face_size', 0)
        if face_size > 0:
            if face_size < 100:  # Visage trop loin
                self.yawn_thresholds['normal'] *= 0.9
            elif face_size > 300:  # Visage trop près
                self.yawn_thresholds['normal'] *= 1.1

    def validate_detection(self, eye_type, ratio):
        """Valide la détection de l'œil ou du bâillement"""
        if eye_type == 'left_eye':
            return ratio < self.eye_ar_thresholds['left']
        elif eye_type == 'right_eye':
            return ratio < self.eye_ar_thresholds['right']
        elif eye_type == 'yawn':
            return ratio > self.yawn_thresholds['normal']
        return False

    def detect_microsleep(self, eye_state, current_time):
        """Détecte un micro-sommeil en fonction de l'état de l'œil"""
        if eye_state == 'closed':
            if len(self.microsleep_history) > 0 and current_time - self.microsleep_history[-1] > self.microsleep_threshold:
                self.microsleep_history.append(current_time)
                return True
        else:
            self.microsleep_history = [current_time]  # Réinitialisation si les yeux sont ouverts
        return False

    def get_status_style(self, fatigue_level):
        """Retourne le style CSS approprié pour le niveau de fatigue"""
        if fatigue_level > 75:
            return """
                color: #FF0000;
                text-align: center;
                padding: 12px;
                background-color: rgba(255, 0, 0, 0.2);
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
            """
        elif fatigue_level > 50:
            return """
                color: #FFA500;
                text-align: center;
                padding: 12px;
                background-color: rgba(255, 165, 0, 0.2);
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
            """
        else:
            return """
                color: #00FF00;
                text-align: center;
                padding: 12px;
                background-color: rgba(0, 255, 0, 0.2);
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
            """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(31, 42, 68))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    window = VigilanceCore()
    window.show()
    sys.exit(app.exec_())