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

class VigilanceCore(QMainWindow):
    def __init__(self):
        super().__init__()

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

        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False

        # Initialisation de MediaPipe FaceMesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Points d'intérêt pour les yeux et la bouche
        self.points_ids = {
            'left_eye': [33, 246, 161, 160],
            'right_eye': [362, 398, 384, 385],
            'mouth': [61, 291, 199, 419]
        }

        # Initialisation des modèles YOLO
        self.detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
        self.detecteye = YOLO("runs/detecteye/train/weights/best.pt")
        self.yolo_object = YOLO("yolov8n.pt")

        self.setup_ui()

        # Capture vidéo
        self.cap = cv2.VideoCapture(0)
        time.sleep(1.0)

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

    def calculate_fatigue_score(self):
        """Calcule un score de fatigue plus sophistiqué basé sur plusieurs facteurs"""
        try:
            # Facteurs de base
            blink_factor = min(1.0, self.blinks / 30)  # Normalisé pour 30 clignements
            microsleep_factor = min(1.0, self.microsleeps / 3)  # Normalisé pour 3 secondes
            yawn_factor = min(1.0, self.yawns / 5)  # Normalisé pour 5 bâillements
            
            # Facteurs temporels
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            time_factor = min(1.0, elapsed_time / 3600)  # Augmente sur une heure
            
            # Calcul du score composite
            base_score = (blink_factor * 0.3 + 
                        microsleep_factor * 0.4 + 
                        yawn_factor * 0.3)
            
            # Ajustement temporel
            fatigue_score = base_score * (1 + time_factor * 0.2)
            
            # Normalisation finale
            return min(100, int(fatigue_score * 100))
            
        except Exception as e:
            print(f"Erreur lors du calcul du score de fatigue: {e}")
            return self.fatigue_level  # Retourne le dernier niveau connu

    def update_drowsiness_state(self):
        """Mise à jour optimisée de l'état de somnolence avec détection améliorée"""
        try:
            # Détection des clignements
            if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
                if not self.left_eye_still_closed and not self.right_eye_still_closed:
                    self.blinks += 1
                    # Analyse du rythme des clignements
                    current_time = time.time()
                    if hasattr(self, 'last_blink_time'):
                        blink_interval = current_time - self.last_blink_time
                        if blink_interval < 0.5:  # Clignements rapides
                            self.microsleeps += 0.2
                    self.last_blink_time = current_time
                    
                    self.left_eye_still_closed = True
                    self.right_eye_still_closed = True
                
                # Détection des micro-sommeils
                if not hasattr(self, 'eyes_closed_start'):
                    self.eyes_closed_start = time.time()
                else:
                    closed_duration = time.time() - self.eyes_closed_start
                    if closed_duration > 0.5:  # Micro-sommeil détecté
                        self.microsleeps += closed_duration / 30
            else:
                if self.left_eye_still_closed and self.right_eye_still_closed:
                    self.left_eye_still_closed = False
                    self.right_eye_still_closed = False
                if hasattr(self, 'eyes_closed_start'):
                    delattr(self, 'eyes_closed_start')
                self.microsleeps = max(0, self.microsleeps - 1/60)  # Décroissance progressive

            # Détection des bâillements
            if self.yawn_state == "Yawn":
                if not self.yawn_in_progress:
                    self.yawn_in_progress = True
                    self.yawns += 1
                    # Analyse du rythme des bâillements
                    current_time = time.time()
                    if hasattr(self, 'last_yawn_time'):
                        yawn_interval = current_time - self.last_yawn_time
                        if yawn_interval < 60:  # Bâillements fréquents
                            self.fatigue_level += 5
                    self.last_yawn_time = current_time
                self.yawn_duration += 1/30
            else:
                if self.yawn_in_progress:
                    self.yawn_in_progress = False
                self.yawn_duration = max(0, self.yawn_duration - 1/60)

            # Mise à jour du niveau de fatigue
            self.fatigue_level = self.calculate_fatigue_score()

        except Exception as e:
            print(f"Erreur lors de la mise à jour de l'état de somnolence: {e}")

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
        try:
            results = self.yolo_object.predict(frame, conf=0.5, classes=[0, 2, 3, 5, 7])  # Person, car, motorcycle, bus, truck
            boxes = results[0].boxes

            detected_objects = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = box.conf.cpu().numpy()[0]
                class_id = int(box.cls.cpu().numpy()[0])
                label = self.yolo_object.names[class_id]

                # Ajouter l'objet détecté à la liste
                detected_objects.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })

                # Dessiner un rectangle autour de l'objet détecté
                color = (0, 255, 0) if label == "person" else (255, 0, 0)  # Vert pour personne, rouge pour autres
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return frame, detected_objects
        except Exception as e:
            print(f"Erreur lors de la détection d'objets: {e}")
            return frame, []

    def setup_camera(self):
        """Configure et initialise la caméra avec gestion des erreurs et reconnexion"""
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Active l'autofocus
            
            if not self.cap.isOpened():
                raise RuntimeError("Impossible d'ouvrir la caméra")
                
            return True
            
        except Exception as e:
            print(f"Erreur lors de l'initialisation de la caméra: {e}")
            return False
            
    def reconnect_camera(self):
        """Tente de reconnecter la caméra en cas de perte de connexion"""
        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            print(f"Tentative de reconnexion de la caméra ({attempt + 1}/{max_attempts})...")
            if self.setup_camera():
                print("Caméra reconnectée avec succès!")
                return True
            attempt += 1
            time.sleep(2)
            
        print("Impossible de reconnecter la caméra après plusieurs tentatives")
        return False

    def capture_frames(self):
        """Capture les frames avec gestion des erreurs et reconnexion automatique"""
        consecutive_errors = 0
        max_errors = 5
        
        while not self.stop_event.is_set():
            try:
                if not self.cap.isOpened():
                    if not self.reconnect_camera():
                        break
                        
                ret, frame = self.cap.read()
                if ret:
                    if self.frame_queue.qsize() < 3:
                        self.frame_queue.put(frame)
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        if not self.reconnect_camera():
                            break
                        consecutive_errors = 0
                        
            except Exception as e:
                print(f"Erreur lors de la capture: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    if not self.reconnect_camera():
                        break
                    consecutive_errors = 0
                    
        print("Arrêt de la capture des frames")

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)

                # Détection des objets
                frame, detected_objects = self.detect_objects(frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        ih, iw, _ = frame.shape
                        
                        # Points pour la bouche
                        mouth_points = []
                        for point_id in [61, 291, 0, 17]:  # Points clés de la bouche
                            lm = face_landmarks.landmark[point_id]
                            x, y = int(lm.x * iw), int(lm.y * ih)
                            mouth_points.append((x, y))
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                        
                        # Points pour les yeux
                        right_eye_points = []
                        left_eye_points = []
                        for point_id in [33, 133, 159, 145]:  # Points clés de l'œil droit
                            lm = face_landmarks.landmark[point_id]
                            x, y = int(lm.x * iw), int(lm.y * ih)
                            right_eye_points.append((x, y))
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                        
                        for point_id in [362, 263, 386, 374]:  # Points clés de l'œil gauche
                            lm = face_landmarks.landmark[point_id]
                            x, y = int(lm.x * iw), int(lm.y * ih)
                            left_eye_points.append((x, y))
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                        
                        # Extraction des ROIs avec marges
                        if len(mouth_points) == 4:
                            x_min = min(x for x, _ in mouth_points)
                            y_min = min(y for _, y in mouth_points)
                            x_max = max(x for x, _ in mouth_points)
                            y_max = max(y for _, y in mouth_points)
                            
                            # Ajout d'une marge pour la ROI
                            margin = 10
                            x_min = max(0, x_min - margin)
                            y_min = max(0, y_min - margin)
                            x_max = min(iw, x_max + margin)
                            y_max = min(ih, y_max + margin)
                            
                            if x_max > x_min and y_max > y_min:
                                mouth_roi = frame[y_min:y_max, x_min:x_max]
                                if mouth_roi.size > 0:
                                    self.predict_yawn(mouth_roi)
                                    # Dessiner le rectangle de la ROI
                                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)
                        
                        # Extraction des ROIs des yeux
                        if len(right_eye_points) == 4:
                            x_min = min(x for x, _ in right_eye_points)
                            y_min = min(y for _, y in right_eye_points)
                            x_max = max(x for x, _ in right_eye_points)
                            y_max = max(y for _, y in right_eye_points)
                            
                            margin = 5
                            x_min = max(0, x_min - margin)
                            y_min = max(0, y_min - margin)
                            x_max = min(iw, x_max + margin)
                            y_max = min(ih, y_max + margin)
                            
                            if x_max > x_min and y_max > y_min:
                                right_eye_roi = frame[y_min:y_max, x_min:x_max]
                                if right_eye_roi.size > 0:
                                    self.right_eye_state = self.predict_eye(right_eye_roi, self.right_eye_state)
                                    # Dessiner le rectangle de la ROI
                                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
                        
                        if len(left_eye_points) == 4:
                            x_min = min(x for x, _ in left_eye_points)
                            y_min = min(y for _, y in left_eye_points)
                            x_max = max(x for x, _ in left_eye_points)
                            y_max = max(y for _, y in left_eye_points)
                            
                            margin = 5
                            x_min = max(0, x_min - margin)
                            y_min = max(0, y_min - margin)
                            x_max = min(iw, x_max + margin)
                            y_max = min(ih, y_max + margin)
                            
                            if x_max > x_min and y_max > y_min:
                                left_eye_roi = frame[y_min:y_max, x_min:x_max]
                                if left_eye_roi.size > 0:
                                    self.left_eye_state = self.predict_eye(left_eye_roi, self.left_eye_state)
                                    # Dessiner le rectangle de la ROI
                                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
                        
                        # Mise à jour des états et des compteurs
                        self.update_drowsiness_state()
                        
                        # Affichage des informations de débogage
                        self.display_debug_info(frame, [self.left_eye_state, self.right_eye_state], self.yawn_state, detected_objects)
                        
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
        with open("vigilance_stats.txt", "w") as f:
            f.write(f"Clignements: {self.blinks}\n")
            f.write(f"Micro-sommeils: {round(self.microsleeps, 2)} s\n")
            f.write(f"Bâillements: {self.yawns}\n")
            f.write(f"Durée bâillements: {round(self.yawn_duration, 2)} s\n")
            f.write(f"Niveau de fatigue: {self.fatigue_level}%\n")
        self.stop_event.set()
        self.cap.release()
        event.accept()

    def display_debug_info(self, frame, eye_states, yawn_state, detected_objects):
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

            # Informations sur les objets détectés
            y_offset = 270
            cv2.putText(frame, "Objects Detected:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            for obj in detected_objects:
                text = f"{obj['label']}: {obj['confidence']:.2f}"
                cv2.putText(frame, text, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
        except Exception as e:
            print(f"Erreur lors de l'affichage des informations de débogage: {e}")

    def update_ui(self, frame):
        """Met à jour l'interface utilisateur"""
        try:
            # Conversion de l'image pour Qt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            
            # Mise à jour de la barre de fatigue
            self.fatigue_bar.setValue(self.fatigue_level)
            
            # Mise à jour des statistiques
            self.metrics["blinks"].setText(f"👁 Clignements: {self.blinks}")
            self.metrics["microsleeps"].setText(f"💤 Micro-sommeils: {round(self.microsleeps, 2)} s")
            self.metrics["yawns"].setText(f"😴 Bâillements: {self.yawns}")
            self.metrics["yawn_duration"].setText(f"⏲ Durée bâillements: {round(self.yawn_duration, 2)} s")
            self.metrics["fps"].setText(f"📈 FPS: {round(self.fps, 1)}")
            
            # Mise à jour du statut
            if self.fatigue_level > 70:
                self.status_label.setText("État: DANGER - Fatigue élevée")
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #ff0000;
                        font-weight: bold;
                        font-size: 18px;
                        padding: 10px;
                        background-color: rgba(255, 0, 0, 0.1);
                        border: 2px solid #ff0000;
                        border-radius: 10px;
                    }
                """)
                if not self.alert_timer.isActive():
                    self.alert_timer.start(500)
            else:
                self.status_label.setText("État: Optimal")
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #00ff00;
                        font-weight: bold;
                        font-size: 18px;
                        padding: 10px;
                        background-color: rgba(0, 255, 0, 0.1);
                        border: 2px solid #00ff00;
                        border-radius: 10px;
                    }
                """)
                if self.alert_timer.isActive():
                    self.alert_timer.stop()
                    
            # Forcer la mise à jour de l'interface
            self.video_label.update()
            self.fatigue_bar.update()
            self.status_label.update()
            self.alert_label.update()
            for label in self.metrics.values():
                label.update()
            
            # Traiter les événements Qt en attente
            QApplication.processEvents()
                    
        except Exception as e:
            print(f"Erreur lors de la mise à jour de l'interface: {e}")

    def setup_ui(self):
        """Configuration améliorée de l'interface utilisateur"""
        # Configuration de la fenêtre principale
        self.setWindowTitle("VigilanceGuard Pro")
        self.setGeometry(100, 100, 1280, 800)
        
        # Widget central et layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Panneau vidéo (gauche)
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        
        # Zone vidéo avec bordure néon
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 3px solid #00D4FF;
                border-radius: 15px;
                background-color: rgba(14, 20, 35, 0.8);
                padding: 5px;
            }
        """)
        video_layout.addWidget(self.video_label)
        
        # Barre de statut vidéo
        self.video_status = QLabel("Caméra active")
        self.video_status.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 14px;
                padding: 5px;
            }
        """)
        video_layout.addWidget(self.video_status)
        
        main_layout.addWidget(video_panel, stretch=2)
        
        # Panneau de contrôle (droite)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setStyleSheet("""
            QWidget {
                background: rgba(14, 20, 35, 0.9);
                border-radius: 15px;
                border: 2px solid #007BFF;
            }
        """)
        
        # Titre du panneau
        title_label = QLabel("Tableau de bord")
        title_label.setStyleSheet("""
            QLabel {
                color: #00D4FF;
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
                border-bottom: 2px solid #007BFF;
            }
        """)
        control_layout.addWidget(title_label)
        
        # Barre de fatigue
        fatigue_widget = QWidget()
        fatigue_layout = QVBoxLayout(fatigue_widget)
        
        fatigue_title = QLabel("Niveau de fatigue")
        fatigue_title.setStyleSheet("color: white; font-size: 18px;")
        fatigue_layout.addWidget(fatigue_title)
        
        self.fatigue_bar = QProgressBar()
        self.fatigue_bar.setRange(0, 100)
        self.fatigue_bar.setTextVisible(True)
        self.fatigue_bar.setFormat("Fatigue: %p%")
        self.fatigue_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 10px;
                background: rgba(10, 15, 35, 0.8);
                text-align: center;
                color: white;
                font-size: 16px;
                height: 25px;
            }
            QProgressBar::chunk {
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00D4FF,
                    stop:0.5 #FF7F00,
                    stop:1 #FF0000);
            }
        """)
        fatigue_layout.addWidget(self.fatigue_bar)
        control_layout.addWidget(fatigue_widget)
        
        # Statut et alertes
        self.status_label = QLabel("État: Optimal")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 20px;
                font-weight: bold;
                padding: 15px;
                border-radius: 10px;
                background: rgba(0, 255, 0, 0.1);
                margin: 10px;
            }
        """)
        control_layout.addWidget(self.status_label)
        
        self.alert_label = QLabel("")
        self.alert_label.setStyleSheet("""
            QLabel {
                color: #FF0000;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-radius: 10px;
                background: rgba(255, 0, 0, 0.1);
                margin: 10px;
            }
        """)
        control_layout.addWidget(self.alert_label)
        
        # Métriques
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_widget.setStyleSheet("""
            QWidget {
                background: rgba(14, 20, 35, 0.5);
                border-radius: 10px;
                margin: 10px;
            }
        """)
        
        self.metrics = {
            "blinks": QLabel("👁 Clignements: 0"),
            "microsleeps": QLabel("💤 Micro-sommeils: 0.0 s"),
            "yawns": QLabel("😴 Bâillements: 0"),
            "yawn_duration": QLabel("⏲ Durée bâillements: 0.0 s"),
            "fps": QLabel("📈 FPS: 0")
        }
        
        for label in self.metrics.values():
            label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 16px;
                    padding: 10px;
                    border-radius: 8px;
                    background: rgba(0, 123, 255, 0.1);
                    margin: 5px;
                }
            """)
            metrics_layout.addWidget(label)
        
        control_layout.addWidget(metrics_widget)
        
        # Boutons de contrôle
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        
        self.reset_button = QPushButton("Réinitialiser")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00D4FF, stop:1 #007BFF);
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #007BFF, stop:1 #00D4FF);
            }
        """)
        self.reset_button.clicked.connect(self.reset_stats)
        button_layout.addWidget(self.reset_button)
        
        self.quit_button = QPushButton("Quitter")
        self.quit_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF007A, stop:1 #FF00D4);
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF00D4, stop:1 #FF007A);
            }
        """)
        self.quit_button.clicked.connect(self.close)
        button_layout.addWidget(self.quit_button)
        
        control_layout.addWidget(button_widget)
        
        # Ajout du panneau de contrôle au layout principal
        main_layout.addWidget(control_panel, stretch=1)

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