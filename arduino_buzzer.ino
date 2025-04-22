// Définition des pins
const int BUZZER_PIN = 9;  // Pin connecté au buzzer
const int LED_PIN = 13;    // LED intégrée pour le débogage

// Paramètres des alertes
const int ALERT_DURATION = 1000;  // Durée de l'alerte en ms
const int ALERT_FREQUENCY = 2000; // Fréquence du buzzer en Hz
const int ALERT_PATTERN_COUNT = 3; // Nombre de répétitions du pattern

// Variables pour le débogage et le contrôle
unsigned long lastCommandTime = 0;
bool buzzerState = false;
int alertPattern = 0;  // 0: normal, 1: attention, 2: danger
unsigned long patternStartTime = 0;
int patternStep = 0;

void setup() {
  // Initialisation de la communication série
  Serial.begin(115200);  // Augmentation du débit pour plus de réactivité
  while (!Serial) {
    ; // Attendre que le port série soit disponible
  }
  
  // Configuration des pins
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  
  // État initial
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  
  // Message de démarrage
  Serial.println("Système d'alerte initialisé");
  Serial.println("En attente de commandes...");
  Serial.println("Commandes disponibles:");
  Serial.println("B - Alerte normale");
  Serial.println("A - Alerte attention");
  Serial.println("D - Alerte danger");
  Serial.println("S - Arrêter l'alerte");
  Serial.println("? - État actuel");
}

void playAlertPattern(int pattern) {
  switch (pattern) {
    case 0: // Alerte normale - bip court
      tone(BUZZER_PIN, ALERT_FREQUENCY, 200);
      digitalWrite(LED_PIN, HIGH);
      delay(200);
      digitalWrite(LED_PIN, LOW);
      delay(200);
      break;
      
    case 1: // Alerte attention - double bip
      for (int i = 0; i < 2; i++) {
        tone(BUZZER_PIN, ALERT_FREQUENCY, 100);
        digitalWrite(LED_PIN, HIGH);
        delay(100);
        digitalWrite(LED_PIN, LOW);
        delay(100);
      }
      delay(200);
      break;
      
    case 2: // Alerte danger - bip continu
      tone(BUZZER_PIN, ALERT_FREQUENCY);
      digitalWrite(LED_PIN, HIGH);
      break;
  }
}

void stopAlert() {
  noTone(BUZZER_PIN);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  buzzerState = false;
  alertPattern = 0;
  patternStep = 0;
}

void loop() {
  // Vérification de la connexion série
  if (Serial.available() > 0) {
    char command = Serial.read();
    lastCommandTime = millis();
    
    // Traitement des commandes
    switch (command) {
      case 'B':  // Alerte normale
        alertPattern = 0;
        buzzerState = true;
        patternStartTime = millis();
        Serial.println("Alerte normale activée");
        break;
        
      case 'A':  // Alerte attention
        alertPattern = 1;
        buzzerState = true;
        patternStartTime = millis();
        Serial.println("Alerte attention activée");
        break;
        
      case 'D':  // Alerte danger
        alertPattern = 2;
        buzzerState = true;
        patternStartTime = millis();
        Serial.println("Alerte danger activée");
        break;
        
      case 'S':  // Arrêter l'alerte
        stopAlert();
        Serial.println("Alerte désactivée");
        break;
        
      case '?':  // État actuel
        Serial.print("État du buzzer: ");
        Serial.println(buzzerState ? "ON" : "OFF");
        Serial.print("Pattern actuel: ");
        switch (alertPattern) {
          case 0: Serial.println("Normal"); break;
          case 1: Serial.println("Attention"); break;
          case 2: Serial.println("Danger"); break;
        }
        break;
        
      default:
        Serial.print("Commande inconnue: ");
        Serial.println(command);
        break;
    }
  }
  
  // Gestion des patterns d'alerte
  if (buzzerState) {
    unsigned long currentTime = millis();
    
    switch (alertPattern) {
      case 0: // Alerte normale
        if (currentTime - patternStartTime >= 1000) {
          playAlertPattern(0);
          patternStartTime = currentTime;
          patternStep++;
          if (patternStep >= ALERT_PATTERN_COUNT) {
            stopAlert();
          }
        }
        break;
        
      case 1: // Alerte attention
        if (currentTime - patternStartTime >= 1000) {
          playAlertPattern(1);
          patternStartTime = currentTime;
          patternStep++;
          if (patternStep >= ALERT_PATTERN_COUNT) {
            stopAlert();
          }
        }
        break;
        
      case 2: // Alerte danger
        playAlertPattern(2);
        break;
    }
  }
  
  // Vérification de la connexion (timeout)
  if (millis() - lastCommandTime > 10000) {  // 10 secondes sans commande
    if (buzzerState) {
      stopAlert();
      Serial.println("Timeout: Alerte désactivée");
    }
  }
} 