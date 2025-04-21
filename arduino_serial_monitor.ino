void setup() {
  Serial.begin(9600);
  Serial.println("Système de détection de somnolence initialisé");
}

void loop() {
  if (Serial.available()) {
    char data = Serial.read();
    if (data == 'S') {
      Serial.println("⚠️ Somnolence détectée !");
    } else if (data == 'A') {
      Serial.println("✅ Conducteur attentif.");
    }
  }
} 