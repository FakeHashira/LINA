from ultralytics import YOLO
import cv2
import numpy as np

# chargement du modele yolov8n
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()                     #lire une image depuis la camera
    if not success:
        print("Erreur de capture de la camera")
        break
    
    
# Effectuer la detection avec modele yolov8n
    results = model(frame)                                                      # passer l'image a yolo pour obtenir les resultat les resultat
 
 
# boucle a travers les objets detectees
    for result in results:
        boxes = result.boxes                                                    #recuperer les boites englobante des objets detectes
        
        for box in boxes:                                                           #pacourir chaque boite detectes
            x1,y1, x2, y2 = map(int, box.xyxy[0])                              #extraire les coordonnes(Xmin,max ; Ymin,max) de la boites et les convertir en entier
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0,255, 0), 2)                   #dessiner la boite autour de l'object detecter de couleur verte
            class_id = int(box.cls[0])                                                      #affichage de la classe de l'objet ID de la classe de l'objet 
            class_name = model.names[class_id]                                               #NOM DE l'objet a parti de la classe
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    

                                                    # affichage de l'image avec les object detectes
    cv2.imshow("Detection en temps reel", frame)

                                                        # sorti de a la boucle si la touche q est presser
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
