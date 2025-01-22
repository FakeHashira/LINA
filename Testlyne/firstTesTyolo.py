import cv2
from ultralytics import YOLO

# recuperer la camera
cap = cv2.VideoCapture(0)

# importer le model
model = YOLO('yolov8n.pt')

# verifier si la webcam s'ouvrer 
# succes recupere le parametre qui affirme que la webcam esr ouvrer , et frame  recupere la frame
while cap.isOpened() :
    success, frame = cap.read()
    if success:
        # le model prend en parametre la frame 
        results = model(frame)
        
        
        # plot permet d'afficher de maniere ordonner  votre resultat ctd votre frame avec toute ces caracterisque(poid, taille,...)
        annoteted_frame = results[0].plot()
        cv2.imshow('frame', annoteted_frame)
        
        if cv2.waitKey(1) & 0xff==ord("q"):
            break
    else:
        break
    
    # relaease permet de libere la camera et destroyall... permet de fermer toute les fenetres 
cap.release()
cv2.destroyAllWindows()