import queue
import threading
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QProgressBar
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve
import serial

class VigilanceCore(QMainWindow):
    def __init__(self, use_arduino=False, arduino=None):
        super().__init__()

        # Ajout des paramètres Arduino
        self.use_arduino = use_arduino
        self.arduino = arduino

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
        self.EYE_AR_THRESH = 0.25  # Seuil pour la détection des yeux fermés
        self.EYE_AR_CONSEC_FRAMES = 3  # Nombre de frames consécutives pour confirmer
        self.YAWN_THRESH = 0.5  # Seuil pour la détection des bâillements
        self.YAWN_CONSEC_FRAMES = 3  # Nombre de frames consécutives pour confirmer

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

        # Initialisation des modèles YOLO
        self.detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
        self.detecteye = YOLO("runs/detecteye/train/weights/best.pt")
        self.yolo_object = YOLO("yolov8n.pt")

        # Configuration de la fenêtre principale
        self.setWindowTitle("Détection de Somnolence")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0A0F23, stop:1 #1E2A44);
                border: none;
            }
        """)

        # Widget central et layout principal
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)

        # Contenu principal avec effet verre dépoli
        self.content_widget = QWidget()
        self.content_layout = QHBoxLayout(self.content_widget)
        self.content_layout.setSpacing(25)
        self.main_layout.addWidget(self.content_widget, stretch=1)

        # Zone vidéo avec bordure néon
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("""
            QLabel {
                border-radius: 20px;
                background-color: rgba(14, 20, 35, 0.8);
                border: 3px solid #00D4FF;
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
            }
        """)
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setScaledContents(True)
        self.content_layout.addWidget(self.video_label, stretch=3)

        # Panneau de contrôle avec effet verre dépoli
        self.control_widget = QWidget()
        self.control_layout = QVBoxLayout(self.control_widget)
        self.control_widget.setStyleSheet("""
            QWidget {
                background: rgba(14, 20, 35, 0.7);
                border-radius: 20px;
                border: 2px solid #007BFF;
                box-shadow: 0 0 15px rgba(0, 123, 255, 0.5);
                padding: 20px;
            }
        """)
        self.content_layout.addWidget(self.control_widget, stretch=1)

        # Barre de fatigue avec dégradé néon
        self.fatigue_bar = QProgressBar()
        self.fatigue_bar.setRange(0, 100)
        self.fatigue_bar.setValue(0)
        self.fatigue_bar.setTextVisible(True)
        self.fatigue_bar.setFormat("Fatigue: %p%")
        self.fatigue_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 10px;
                background: rgba(10, 15, 35, 0.8);
                text-align: center;
                color: #FFFFFF;
                font-family: 'Orbitron';
                font-size: 16px;
                box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D4FF, stop:1 #FF007A);
                border-radius: 10px;
                box-shadow: 0 0 15px rgba(255, 0, 122, 0.5);
            }
        """)
        self.control_layout.addWidget(self.fatigue_bar)

        # Statut principal avec effet néon
        self.status_label = QLabel("État: Optimal")
        self.status_label.setFont(QFont("Orbitron", 20, QFont.Bold))
        self.status_label.setStyleSheet("""
            QLabel {
                color: #00D4FF;
                text-align: center;
                padding: 15px;
                background: rgba(0, 212, 255, 0.2);
                border-radius: 12px;
                box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
            }
        """)
        self.control_layout.addWidget(self.status_label)

        # Alerte avec effet clignotant
        self.alert_label = QLabel("")
        self.alert_label.setFont(QFont("Orbitron", 18, QFont.Bold))
        self.alert_label.setStyleSheet("""
            QLabel {
                color: #FF007A;
                text-align: center;
                padding: 12px;
                background: rgba(255, 0, 122, 0.2);
                border-radius: 12px;
                box-shadow: 0 0 15px rgba(255, 0, 122, 0.4);
            }
        """)
        self.control_layout.addWidget(self.alert_label)

        # Métriques avec effet néon
        self.metrics = {
            "blinks": QLabel("👁 Clignements: 0"),
            "microsleeps": QLabel("💤 Micro-sommeils: 0 s"),
            "yawns": QLabel("😴 Bâillements: 0"),
            "yawn_duration": QLabel("⏲ Durée bâillements: 0 s"),
            "fps": QLabel("📈 FPS: 0")
        }
        for label in self.metrics.values():
            label.setFont(QFont("Orbitron", 14))
            label.setStyleSheet("""
                QLabel {
                    color: #FFFFFF;
                    padding: 12px;
                    background: rgba(0, 123, 255, 0.1);
                    border-radius: 10px;
                    margin: 5px 0;
                    box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
                }
            """)
            label.setMinimumHeight(50)
            self.control_layout.addWidget(label)

        self.control_layout.addStretch()

        # Boutons avec effet néon
        self.button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Réinitialiser")
        self.reset_button.setFont(QFont("Orbitron", 14, QFont.Bold))
        self.reset_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D4FF, stop:1 #007BFF);
                color: #FFFFFF;
                padding: 12px;
                border-radius: 10px;
                border: none;
                box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007BFF, stop:1 #00D4FF);
                box-shadow: 0 0 25px rgba(0, 212, 255, 0.8);
            }
        """)
        self.reset_button.clicked.connect(self.reset_stats)
        self.button_layout.addWidget(self.reset_button)

        self.quit_button = QPushButton("Arrêt")
        self.quit_button.setFont(QFont("Orbitron", 14, QFont.Bold))
        self.quit_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF007A, stop:1 #FF00D4);
                color: #FFFFFF;
                padding: 12px;
                border-radius: 10px;
                border: none;
                box-shadow: 0 0 15px rgba(255, 0, 122, 0.5);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF00D4, stop:1 #FF007A);
                box-shadow: 0 0 25px rgba(255, 0, 122, 0.8);
            }
        """)
        self.quit_button.clicked.connect(self.close)
        self.button_layout.addWidget(self.quit_button)

        self.control_layout.addLayout(self.button_layout)

        # Test des caméras disponibles et sélection de la caméra USB externe
        available_cameras = []
        for i in range(10):  # Teste les 10 premiers indices de caméra
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"Caméra {i} disponible")
                cap.release()
        
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
        self.cap = cv2.VideoCapture(camera_index)
        
        # Configuration de la résolution de la caméra (1280x720 pour de meilleures performances)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 30 FPS pour de meilleures performances
        
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
            self.status_label.setText("État: Optimal")
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
        self.metrics["blinks"].setText(f"👁 Clignements: {self.blinks}")
        self.metrics["microsleeps"].setText(f"💤 Micro-sommeils: {round(self.microsleeps, 2)} s")
        self.metrics["yawns"].setText(f"😴 Bâillements: {self.yawns}")
        self.metrics["yawn_duration"].setText(f"⏲ Durée bâillements: {round(self.yawn_duration, 2)} s")
        self.metrics["fps"].setText(f"📈 FPS: {round(self.fps, 1)}")

        # Calcul des FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.metrics["fps"].setText(f"📈 FPS: {round(self.fps, 1)}")
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
        self.alert_blink_state = not self.alert_blink_state
        glow = "0 0 15px rgba(178, 34, 34, 0.5)" if self.alert_blink_state else "0 0 5px rgba(178, 34, 34, 0.3)"
        self.alert_label.setStyleSheet(f"""
            color: #B22222;
            text-align: center;
            padding: 10px;
            background-color: rgba(178, 34, 34, 0.2);
            border-radius: 8px;
            box-shadow: {glow};
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
            # Prétraitement de l'image
            eye_frame = cv2.resize(eye_frame, (64, 64))
            eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB)
            
            results_eye = self.detecteye.predict(eye_frame)
            boxes = results_eye[0].boxes
            
            if len(boxes) == 0:
                return eye_state

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_confidence_index = np.argmax(confidences)
            class_id = int(class_ids[max_confidence_index])
            confidence = confidences[max_confidence_index]

            # Seuils de confiance ajustés
            if class_id == 1 and confidence > 0.25:  # Œil fermé
                eye_state = "Close Eye"
            elif class_id == 0 and confidence > 0.25:  # Œil ouvert
                eye_state = "Open Eye"
            
            return eye_state
        except Exception as e:
            print(f"Erreur lors de la prédiction de l'œil: {e}")
            return eye_state

    def predict_yawn(self, yawn_frame):
        """Prédit l'état du bâillement avec amélioration de la détection"""
        try:
            # Prétraitement de l'image
            yawn_frame = cv2.resize(yawn_frame, (64, 64))
            yawn_frame = cv2.cvtColor(yawn_frame, cv2.COLOR_BGR2RGB)
            
            results_yawn = self.detectyawn.predict(yawn_frame)
            boxes = results_yawn[0].boxes

            if len(boxes) == 0:
                return self.yawn_state

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_confidence_index = np.argmax(confidences)
            class_id = int(class_ids[max_confidence_index])
            confidence = confidences[max_confidence_index]

            # Seuils de confiance ajustés
            if class_id == 0 and confidence > 0.40:  # Bâillement
                self.yawn_state = "Yawn"
            elif class_id == 1 and confidence > 0.40:  # Pas de bâillement
                self.yawn_state = "No Yawn"
            
        except Exception as e:
            print(f"Erreur lors de la prédiction du bâillement: {e}")

    def detect_objects(self, frame):
        """Détection des objets (personnes, animaux, etc.) avec YOLOv8"""
        results = self.yolo_object.predict(frame, conf=0.5)  # Seuil de confiance à 0.5
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = box.conf.cpu().numpy()[0]
            class_id = int(box.cls.cpu().numpy()[0])
            label = self.yolo_object.names[class_id]

            # Dessiner un rectangle autour de l'objet détecté
            color = (0, 255, 0) if label == "person" else (255, 0, 0)  # Vert pour personne, rouge pour autres
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

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

    def mouth_aspect_ratio(self, mouth_points):
        """Calcule le ratio d'aspect de la bouche (MAR)"""
        # Calcul des distances verticales
        v1 = np.linalg.norm(mouth_points[1] - mouth_points[5])
        v2 = np.linalg.norm(mouth_points[2] - mouth_points[4])
        
        # Calcul de la distance horizontale
        h = np.linalg.norm(mouth_points[0] - mouth_points[3])
        
        # Calcul du ratio
        mar = (v1 + v2) / (2.0 * h)
        return mar

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        ih, iw, _ = frame.shape
                        
                        # Extraction des points pour les yeux
                        left_eye_points = np.array([(face_landmarks.landmark[id].x * iw, 
                                                   face_landmarks.landmark[id].y * ih) 
                                                  for id in self.points_ids['left_eye']])
                        right_eye_points = np.array([(face_landmarks.landmark[id].x * iw, 
                                                    face_landmarks.landmark[id].y * ih) 
                                                   for id in self.points_ids['right_eye']])
                        
                        # Extraction des points pour la bouche
                        mouth_points = np.array([(face_landmarks.landmark[id].x * iw, 
                                                face_landmarks.landmark[id].y * ih) 
                                               for id in self.points_ids['mouth']])
                        
                        # Calcul des ratios
                        left_ear = self.eye_aspect_ratio(left_eye_points)
                        right_ear = self.eye_aspect_ratio(right_eye_points)
                        mar = self.mouth_aspect_ratio(mouth_points)
                        
                        # Détection des yeux fermés
                        if left_ear < self.EYE_AR_THRESH:
                            self.left_eye_counter += 1
                            if self.left_eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                                self.left_eye_state = "Close Eye"
                        else:
                            self.left_eye_counter = 0
                            self.left_eye_state = "Open Eye"
                        
                        if right_ear < self.EYE_AR_THRESH:
                            self.right_eye_counter += 1
                            if self.right_eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                                self.right_eye_state = "Close Eye"
                        else:
                            self.right_eye_counter = 0
                            self.right_eye_state = "Open Eye"
                        
                        # Détection des bâillements
                        if mar > self.YAWN_THRESH:
                            self.yawn_counter += 1
                            if self.yawn_counter >= self.YAWN_CONSEC_FRAMES:
                                self.yawn_state = "Yawn"
                        else:
                            self.yawn_counter = 0
                            self.yawn_state = "No Yawn"
                        
                        # Mise à jour des états et des compteurs
                        if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
                            if not self.left_eye_still_closed and not self.right_eye_still_closed:
                                self.left_eye_still_closed = True
                                self.right_eye_still_closed = True
                                self.blinks += 1
                            self.microsleeps += 45 / 1000
                        else:
                            if self.left_eye_still_closed and self.right_eye_still_closed:
                                self.left_eye_still_closed = False
                                self.right_eye_still_closed = False
                            self.microsleeps = 0

                        if self.yawn_state == "Yawn":
                            if not self.yawn_in_progress:
                                self.yawn_in_progress = True
                                self.yawns += 1
                            self.yawn_duration += 45 / 1000
                        else:
                            if self.yawn_in_progress:
                                self.yawn_in_progress = False
                                self.yawn_duration = 0

                        # Mise à jour des statistiques
                        self.update_stats()
                        
                        # Affichage des informations de débogage
                        self.display_debug_info(frame, [self.left_eye_state, self.right_eye_state], self.yawn_state)
                        
                        # Affichage de la frame
                        self.display_frame(frame)

            except queue.Empty:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def play_alert_sound(self):
        frequency = 2200  # Son clair et professionnel
        duration = 200
        winsound.Beep(frequency, duration)

    def play_sound_in_thread(self):
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()

    def resizeEvent(self, event):
        self.display_frame(self.frame_queue.get() if not self.frame_queue.empty() else np.zeros((480, 640, 3), dtype=np.uint8))
        super().resizeEvent(event)

    def closeEvent(self, event):
        # Fermer la connexion Arduino si elle existe
        if self.arduino:
            self.arduino.close()
        with open("vigilance_stats.txt", "w") as f:
            f.write(f"Clignements: {self.blinks}\n")
            f.write(f"Micro-sommeils: {round(self.microsleeps, 2)} s\n")
            f.write(f"Bâillements: {self.yawns}\n")
            f.write(f"Durée bâillements: {round(self.yawn_duration, 2)} s\n")
            f.write(f"Niveau de fatigue: {self.fatigue_level}%\n")
        self.stop_event.set()
        self.cap.release()
        event.accept()

    def display_debug_info(self, frame, eye_states, yawn_state):
        """Affiche les informations de débogage sur la frame"""
        try:
            # Informations sur les yeux
            cv2.putText(frame, f"Left Eye: {eye_states[0]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Eye: {eye_states[1]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Information sur le bâillement
            cv2.putText(frame, f"Yawn: {yawn_state}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Informations sur les compteurs
            cv2.putText(frame, f"Blinks: {self.blinks}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Microsleeps: {round(self.microsleeps, 2)}s", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yawns: {self.yawns}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Niveau de fatigue
            fatigue_color = (0, 255, 0) if self.fatigue_level < 50 else (0, 255, 255) if self.fatigue_level < 75 else (0, 0, 255)
            cv2.putText(frame, f"Fatigue: {self.fatigue_level}%", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fatigue_color, 2)
            
            # FPS
            cv2.putText(frame, f"FPS: {round(self.fps, 1)}", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Erreur lors de l'affichage des informations de débogage: {e}")

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