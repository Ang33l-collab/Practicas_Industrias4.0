from ultralytics import YOLO

# Carga tu modelo recién colocado en la carpeta
model = YOLO("weights.pt")

# Realiza la predicción usando tu cámara web en vivo (source="0")
results = model.predict(source="0", show=True, conf=0.5)
