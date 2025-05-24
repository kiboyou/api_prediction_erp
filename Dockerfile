# Utilisation de l'image officielle Python
FROM python:3.11

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers nécessaires
COPY requirements.txt .
COPY model_class.pkl model_reg.pkl scaler.pkl encoder.pkl ./
COPY main.py .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposition du port 8000
EXPOSE 8003

# Commande pour lancer l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
