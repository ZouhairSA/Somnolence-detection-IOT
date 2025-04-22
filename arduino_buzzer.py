import serial
import time
import sys

class ArduinoBuzzer:
    def __init__(self, port='COM3', baud_rate=115200):
        """
        Initialise la communication avec l'Arduino
        :param port: Port série de l'Arduino (ex: 'COM3' pour Windows)
        :param baud_rate: Vitesse de communication (115200 par défaut)
        """
        self.port = port
        self.baud_rate = baud_rate
        self.arduino = None
        self.connect()

    def connect(self):
        """Établit la connexion avec l'Arduino"""
        try:
            print(f"Tentative de connexion à l'Arduino sur le port {self.port}...")
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Attendre que l'Arduino se réinitialise
            
            # Vérifier la connexion
            self.arduino.write(b'?\n')  # Demander l'état
            response = self.arduino.readline().decode().strip()
            if response:
                print(f"Arduino connecté avec succès! Réponse: {response}")
            else:
                print("Pas de réponse de l'Arduino, vérifiez la connexion")
                self.arduino = None
                
        except serial.SerialException as e:
            print(f"Erreur de connexion à l'Arduino: {e}")
            print("Veuillez vérifier:")
            print("1. Le port COM est correct (actuellement: {})".format(self.port))
            print("2. L'Arduino est bien connecté")
            print("3. Aucun autre programme n'utilise le port")
            self.arduino = None

    def play_alert(self, alert_type='normal', duration_ms=1000):
        """
        Active le buzzer avec différents types d'alertes
        :param alert_type: Type d'alerte ('normal', 'attention', 'danger')
        :param duration_ms: Durée du son en millisecondes
        """
        if self.arduino is None:
            print("Erreur: Arduino non connecté")
            return False

        try:
            # Envoi de la commande appropriée
            if alert_type == 'normal':
                self.arduino.write(b'B\n')
                print("Alerte normale envoyée")
            elif alert_type == 'attention':
                self.arduino.write(b'A\n')
                print("Alerte attention envoyée")
            elif alert_type == 'danger':
                self.arduino.write(b'D\n')
                print("Alerte danger envoyée")
            else:
                print(f"Type d'alerte inconnu: {alert_type}")
                return False

            # Attendre la durée spécifiée
            time.sleep(duration_ms / 1000)
            
            # Arrêter l'alerte
            self.arduino.write(b'S\n')
            print("Arrêt de l'alerte envoyé")
            return True
            
        except serial.SerialException as e:
            print(f"Erreur lors de l'envoi de la commande: {e}")
            return False

    def check_connection(self):
        """Vérifie l'état de la connexion avec l'Arduino"""
        if self.arduino is None:
            print("Tentative de reconnexion...")
            self.connect()
            return False
        return True

    def close(self):
        """Ferme la connexion avec l'Arduino"""
        if self.arduino is not None:
            try:
                self.arduino.write(b'S\n')  # Arrêter toute alerte en cours
                time.sleep(0.1)
                self.arduino.close()
                print("Connexion Arduino fermée")
            except serial.SerialException as e:
                print(f"Erreur lors de la fermeture de la connexion: {e}")
        self.arduino = None 