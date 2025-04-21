import sys
from PyQt5.QtWidgets import QApplication
from DrowsinessDetector import VigilanceCore
import serial
import time

def main():
    app = QApplication(sys.argv)
    
    # Demander à l'utilisateur s'il veut utiliser l'Arduino
    use_arduino = input("Voulez-vous utiliser l'Arduino ? (o/n): ").lower() == 'o'
    
    # Initialiser la communication série si nécessaire
    arduino = None
    if use_arduino:
        try:
            arduino = serial.Serial('COM3', 9600)  # Arduino Mega est sur COM3
            time.sleep(2)  # Attendre que la connexion soit établie
            print("Arduino connecté sur COM3")
        except serial.SerialException:
            print("Erreur: Impossible de se connecter à l'Arduino sur COM3")
            use_arduino = False
    
    # Créer et lancer l'interface principale
    window = VigilanceCore(use_arduino=use_arduino, arduino=arduino)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 