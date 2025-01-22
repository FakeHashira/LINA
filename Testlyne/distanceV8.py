# import cv2
# from ultralytics import YOLO 
# from ultralytics.solutions import distance_calculation

# Chargement du modèle
# model = YOLO("yolov8n.pt")
# names = model.model.names

# # Capture vidéo
# cap = cv2.VideoCapture("C:/Users/M_inko_H/Desktop/yolov8Doc/ultralytics/Testlyne/poeple.mov")
# assert cap.isOpened(), "Error open stream"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# # Video writer
# video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# # Initialisation de l'objet distance calculation
# dist_obj = distance_calculation.DistanceCalculation(view_img=True)

# # Process video
# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break
    
#     # Réaliser la détection et obtenir les résultats
#     results = model.track(im0, persist=True, show=False)
    
#     # Traiter l'image avec les résultats
#     if results:
#         # Utiliser la méthode process_frame au lieu d'appeler l'objet directement
#         annotated_frame = dist_obj.process_frame(im0, results[0])
        
#         # Écrire l'image traitée
#         if annotated_frame is not None:
#             video_writer.write(annotated_frame)
        
#         # Afficher l'image (optionnel)
#         cv2.imshow("Distance Calculation", annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Appuyez sur 'q' pour quitter
#             break

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()



# import cv2

# from ultralytics import solutions

# cap = cv2.VideoCapture("C:/Users/M_inko_H/Desktop/yolov8Doc/ultralytics/Testlyne/poeple.mov")
# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# # Video writer
# video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# # Init distance-calculation obj
# distance = solutions.DistanceCalculation(model="yolo11n.pt", show=True)

# # Process video
# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break
#     im0 = distance.calculate(im0)
#     video_writer.write(im0)

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()



import cv2
from ultralytics import YOLO
import numpy as np

# Chargement du modèle YOLOv8
model = YOLO("yolov8n.pt")

# Dictionnaire des hauteurs moyennes réelles (en cm) pour certains objets communs
KNOWN_HEIGHTS = {
    'person': 170,      # cm (hauteur moyenne adulte)
    'bicycle': 100,     # cm (hauteur moyenne vélo)
    'car': 150,        # cm (hauteur moyenne voiture)
    'motorcycle': 120,  # cm (hauteur moyenne moto)
    'airplane': 400,    # cm (hauteur moyenne petit avion commercial)
    'bus': 320,        # cm (hauteur moyenne bus)
    'train': 400,      # cm (hauteur moyenne wagon)
    'truck': 350,      # cm (hauteur moyenne camion)
    'boat': 200,       # cm (hauteur moyenne petit bateau)
    'traffic light': 300, # cm (hauteur moyenne feu de circulation)
    'fire hydrant': 90,  # cm (hauteur moyenne borne incendie)
    'stop sign': 210,    # cm (hauteur moyenne panneau stop avec poteau)
    'parking meter': 150, # cm (hauteur moyenne parcmètre)
    'bench': 75,        # cm (hauteur moyenne banc)
    'bird': 20,         # cm (hauteur moyenne oiseau)
    'cat': 25,          # cm (hauteur moyenne chat)
    'dog': 50,          # cm (hauteur moyenne chien)
    'horse': 160,       # cm (hauteur moyenne cheval)
    'sheep': 100,       # cm (hauteur moyenne mouton)
    'cow': 140,         # cm (hauteur moyenne vache)
    'elephant': 300,    # cm (hauteur moyenne éléphant)
    'bear': 180,        # cm (hauteur moyenne ours debout)
    'zebra': 140,       # cm (hauteur moyenne zèbre)
    'giraffe': 500,     # cm (hauteur moyenne girafe)
    'backpack': 45,     # cm (hauteur moyenne sac à dos)
    'umbrella': 80,     # cm (hauteur moyenne parapluie)
    'handbag': 30,      # cm (hauteur moyenne sac à main)
    'tie': 40,          # cm (longueur moyenne cravate)
    'suitcase': 70,     # cm (hauteur moyenne valise)
    'frisbee': 3,       # cm (hauteur/épaisseur moyenne frisbee)
    'skis': 170,        # cm (longueur moyenne ski)
    'snowboard': 150,   # cm (longueur moyenne snowboard)
    'sports ball': 22,  # cm (diamètre moyen ballon)
    'kite': 100,        # cm (hauteur moyenne cerf-volant)
    'baseball bat': 90, # cm (longueur moyenne batte)
    'baseball glove': 25, # cm (hauteur moyenne gant)
    'skateboard': 20,   # cm (hauteur moyenne skateboard)
    'surfboard': 180,   # cm (longueur moyenne planche de surf)
    'tennis racket': 70, # cm (longueur moyenne raquette)
    'bottle': 25,       # cm (hauteur moyenne bouteille)
    'wine glass': 15,   # cm (hauteur moyenne verre à vin)
    'cup': 10,          # cm (hauteur moyenne tasse)
    'fork': 15,         # cm (longueur moyenne fourchette)
    'knife': 20,        # cm (longueur moyenne couteau)
    'spoon': 15,        # cm (longueur moyenne cuillère)
    'bowl': 8,          # cm (hauteur moyenne bol)
    'banana': 20,       # cm (longueur moyenne banane)
    'apple': 8,         # cm (hauteur moyenne pomme)
    'sandwich': 8,      # cm (hauteur moyenne sandwich)
    'orange': 7,        # cm (diamètre moyen orange)
    'broccoli': 15,     # cm (hauteur moyenne brocoli)
    'carrot': 15,       # cm (longueur moyenne carotte)
    'hot dog': 15,      # cm (longueur moyenne hot dog)
    'pizza': 3,         # cm (hauteur moyenne pizza)
    'donut': 3,         # cm (hauteur moyenne donut)
    'cake': 10,         # cm (hauteur moyenne gâteau)
    'chair': 85,        # cm (hauteur moyenne chaise)
    'couch': 90,        # cm (hauteur moyenne canapé)
    'potted plant': 40, # cm (hauteur moyenne plante en pot)
    'bed': 60,          # cm (hauteur moyenne lit)
    'dining table': 75, # cm (hauteur moyenne table)
    'toilet': 70,       # cm (hauteur moyenne toilette)
    'tv': 50,           # cm (hauteur moyenne TV)
    'laptop': 25,       # cm (hauteur moyenne laptop ouvert)
    'mouse': 3,         # cm (hauteur moyenne souris)
    'remote': 15,       # cm (longueur moyenne télécommande)
    'keyboard': 3,      # cm (hauteur moyenne clavier)
    'cell phone': 15,   # cm (hauteur moyenne téléphone)
    'microwave': 30,    # cm (hauteur moyenne micro-ondes)
    'oven': 60,         # cm (hauteur moyenne four)
    'toaster': 20,      # cm (hauteur moyenne grille-pain)
    'sink': 85,         # cm (hauteur moyenne évier)
    'refrigerator': 170, # cm (hauteur moyenne réfrigérateur)
    'book': 25,         # cm (hauteur moyenne livre)
    'clock': 30,        # cm (diamètre moyen horloge)
    'vase': 30,         # cm (hauteur moyenne vase)
    'scissors': 15,     # cm (longueur moyenne ciseaux)
    'teddy bear': 40,   # cm (hauteur moyenne ours en peluche)
    'hair drier': 25,   # cm (longueur moyenne sèche-cheveux)
    'toothbrush': 20    # cm (longueur moyenne brosse à dents)
}

# Paramètres de la caméra (à ajuster selon votre caméra)
FOCAL_LENGTH = 1000  # distance focale en pixels
SENSOR_HEIGHT = 480  # hauteur du capteur en pixels

def calculate_distance(focal_length, real_height, apparent_height):
    """Calcule la distance en utilisant la similarité des triangles"""
    if apparent_height == 0:
        return 0
    return (focal_length * real_height) / apparent_height

# Capture vidéo
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    # Détection des objets
    results = model(frame)
    
    # Pour chaque détection
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Récupérer les coordonnées de la boîte
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            
            # Calculer la hauteur apparente
            apparent_height = y2 - y1
            
            # Si l'objet est dans notre dictionnaire de hauteurs connues
            if class_name in KNOWN_HEIGHTS:
                # Calculer la distance
                distance = calculate_distance(
                    FOCAL_LENGTH,
                    KNOWN_HEIGHTS[class_name],
                    apparent_height
                )
                
                # Dessiner la boîte
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Afficher les informations
                text = f"{class_name}: {distance:.2f}cm"
                cv2.putText(frame, text, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Écrire la frame
    video_writer.write(frame)
    
    # Afficher la frame (optionnel)
    cv2.imshow('Distance Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()