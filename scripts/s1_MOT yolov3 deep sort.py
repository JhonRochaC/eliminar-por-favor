
#Dado que Yolo usa el interfaz de comandos de linea, se debe importar flags setting para usar versión 3 de YOLO en tersorflow posterioemnte.
#Inicializa la configuración de flags para el YOLO_v3
from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)
#-----

a = 10
b=11
c = 30

from os import listdir
from os.path import isfile, join, splitext, exists, getsize
import time #Para calcular los frames por segundo
import csv
import numpy as np
import cv2 #Para visualizar el tracking
import matplotlib.pyplot as plt #Solo para el mapa de color

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images #Para cambiar el tamaño de las imágenes
from yolov3_tf2.utils import convert_boxes #Para cambiar formato de cajas

from deep_sort import preprocessing
from deep_sort import nn_matching #Para la asociación de matriz
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet #Para generar características

##Choose files path
cd = 'CD2'
path = f'data/{cd}/'
videoNames = [splitext(f)[0] for f in listdir(path) if isfile(join(path, f))]
##--

class_names = [c.strip() for c in open('commons/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names)) #Se ingresa el número de clases
yolo.load_weights('commons/yolov3.tf') #cargar los pesos

max_cosine_distance = 0.5 #si es la distancia coseno es mayor significa que las características son similares entre los objetos del frame anterior y actual
nn_budget = None #Para formar una librería de las características, se pone none para que no se guarde
nms_max_overlap = 0.8 #Para tratar cuando hayan varias detecciones para un mismo objeto, por defecto es 1 (lo que significa que se mantienen todas las detecciones, lo cual no es conveniente siempre).

model_filename = 'scripts/model_data/mars-small128.pb' #CNN preentrenada para chequear peatones, solo para fines ilustivos
encoder = gdet.create_box_encoder(model_filename, batch_size = 1) #Para la generación de features
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget) #Creación de la matriz de asociación 
tracker = Tracker(metric) #Se pasa al tracker de DeepSORT

#Required class indices
required_class_id = [2, 3, 5, 7]
classes_names = ['Carro', 'Motocicleta', 'Bus', 'Camión']

#Styles
font_size = 0.5
font_thickness = 2

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
font_color = [[i * 255 for i in colors[0]],
                [i * 255 for i in colors[1]],
                [i * 255 for i in colors[2]],
                [i * 255 for i in colors[3]]]

def createOutputHeader():
    with open(f"output/{cd}/detalleConteo.csv", 'a', newline='') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Video', 'Timestamp', 'Id', 'Dirección', 'Vehículo'])
    f1.close()

def saveRecord(record):
    with open(f"output/{cd}/detalleConteo.csv", 'a', newline='') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(record)
    f1.close()

def processVideo(videoName):
    #Video
    vid = cv2.VideoCapture(f'{path}{videoName}.mp4')
    #Fin video

    codec = cv2.VideoWriter_fourcc(*'XVID') #Formatod e guardado de video
    vid_fps =int(vid.get(cv2.CAP_PROP_FPS)) #Obtiene FPS del video original
    vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) #Obtiene ancho y altura del video original
    out = cv2.VideoWriter(f'output/{cd}/{videoName}.avi', codec, vid_fps, (vid_width, vid_height)) #Crea el output del video, en fomrato .avi

    #from _collections import deque
    #pts = [deque(maxlen=30) for _ in range(1000)]

    counter = []

    # List to store vehicle count information
    temp_top_list = []
    temp_bottom_list = []
    top_list = [0, 0, 0, 0]
    bottom_list = [0, 0, 0, 0]

    while True: #Usado para capturar todos los frames del video, lee el video uno por uno
        _, img = vid.read()
        if img is None: #Para el final del video cuando no haya imágen
            print('Completed')
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#Convierte los colores de RGR a RGB
        img_in = tf.expand_dims(img_in, 0) #Expasión de las diemnsiones de la imágen, se le agrega una dimensión de para el batch
        img_in = transform_images(img_in, 416) #resize to Yolo V3 size

        #resize a la mitad 480, 544. Solo para pruebas
        #img = cv2.resize(img, (0,0), None, 0.5, 0.5)

        t1 = time.time() #inicia el temporizador

        boxes, scores, classes, nums = yolo.predict(img_in) #Realiza la predicción Yolo

        # boxes, 3D shape (1, 100, 4), 3 dimensiones, máximo de 100 boxes, para cada una se guardan 4 datos ( (x,y) center coordinates, width y height)
        # scores, 2D shape (1, 100), puntuajes de confianza
        # classes, 2D shape (1, 100), las clases, en número respecto del array class_names ingresado desde commons
        # nums, 1D shape (1,), número total de objetos detectados

        classes = classes[0]#primera fila
        names = []

        # Gate
        height, width, _ = img.shape
        middle_gate = int(14*height/24)
        top_gate = int(14*height/24-1.8*height/24)
        bottom_gate = int(14*height/24+1.8*height/24)

        for i in range(len(classes)):
            names.append(class_names[int(classes[i])]) #Aña de a array de nombres el nombre de la clase identificada i, para todas las clases detectadas
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0]) #primera fila de boxes, pasa las cajas al tamaño original de la imágen, y convierte las cajas en una lista
        features = encoder(img, converted_boxes)#Generar las caracteríaticas de cada objeto detectado

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)] #Almacena la información en un objeto

        #Se extrae la info de detections a np.arrays
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)#Se suprimen los objetos detectados
        detections = [detections[i] for i in indices]#Y se vuelven a agregar los objetos

        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            #Skip track if it can't be tracked
            if not track.is_confirmed() or track.time_since_update >1 : #Si no el filtro de kalman no pudo asignar un track y no hay actualización en el track, se salta el track
                continue
            bbox = track.to_tlbr() #format that cv2 uses
            class_name= track.get_class() 
            class_id = class_names.index(class_name)
            index = -1
            if class_id in required_class_id:
                index = required_class_id.index(class_id)
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2) #Crea rectandgulo, el númedo es el ancho de la linea
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)), (int(bbox[0])+(len(class_name)
                        +len(str(track.track_id)))*17, int(bbox[1])), color, -1)# rectangulo para el label y id, -1 para que rellene el rectangulo
            cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                        (255, 255, 255), 2) #Texto del rectangulo anterior

            #Find the center of the bbox
            center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2)) 
            ix, iy = center

            #count
            id = int(track.track_id)
            if (index != -1):
                if (iy > top_gate) and (iy < middle_gate):
                    if id not in temp_top_list:
                        temp_top_list.append(id)

                elif (iy < bottom_gate) and (iy > middle_gate):
                    if id not in temp_bottom_list:
                        temp_bottom_list.append(id)

                elif iy > bottom_gate:
                    if id in temp_top_list:
                        temp_top_list.remove(id)
                        top_list[index] = top_list[index]+1
                        saveRecord([videoName, vid.get(cv2.CAP_PROP_POS_MSEC), id, 'Arriba', classes_names[index]])

                elif iy < top_gate:
                    if id in temp_bottom_list:
                        temp_bottom_list.remove(id)
                        bottom_list[index] = bottom_list[index] + 1
                        saveRecord([videoName, vid.get(cv2.CAP_PROP_POS_MSEC), id, 'Abajo', classes_names[index]])

        total_count = len(set(counter))
        #cv2.putText(img, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
        #cv2.putText(img, "Total Vehicle Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)

        # Draw gates
        cv2.line(img, (0, middle_gate), (width, middle_gate), (0, 200, 0), thickness=2)
        cv2.line(img, (0, top_gate), (width, top_gate), (0, 255, 0), thickness=2)
        cv2.line(img, (0, bottom_gate), (width, bottom_gate), (0, 255, 0), thickness=2)

        # Draw counting texts in the frame
        cv2.putText(img, "Arriba", (130, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), font_thickness)
        cv2.putText(img, "Abajo", (220, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), font_thickness)
        cv2.putText(img, "Carro:          "+str(top_list[0]).zfill(3)+"       "+ str(bottom_list[0]).zfill(3), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thickness)
        cv2.putText(img, "Motocicleta:    "+str(top_list[1]).zfill(3)+"       "+ str(bottom_list[1]).zfill(3), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thickness)
        cv2.putText(img, "Bus:            "+str(top_list[2]).zfill(3)+"       "+ str(bottom_list[2]).zfill(3), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thickness)
        cv2.putText(img, "Camion:         "+str(top_list[3]).zfill(3)+"       "+ str(bottom_list[3]).zfill(3), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thickness)

        fps = 1./(time.time()-t1)
        cv2.putText(img, "FPS: {:.2f}".format(fps), (vid_width-200,40), 0, 1, (0,0,255), 2)

        cv2.imshow('output', img)
        #cv2.resizeWindow('output', 1024, 768)
        out.write(img)

        if cv2.waitKey(1) == ord('q'):
            break

    # Write the vehicle counting information in a file and save it
    with open(f"output/{cd}/conteo.csv", 'a', newline='') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Video', 'Dirección', 'Carro', 'Motocicleta', 'Bus', 'Camión'])
        top_list.insert(0, "Arriba")
        top_list.insert(0, videoName)
        bottom_list.insert(0, "Abajo")
        bottom_list.insert(0, videoName)
        cwriter.writerow(top_list)
        cwriter.writerow(bottom_list)
    f1.close()

    vid.release()
    out.release()
    cv2.destroyAllWindows()

def startAnalysis():
    outputPath = f"output/{cd}/detalleConteo.csv"
    if not (exists(outputPath) and getsize(outputPath) > 0):
        createOutputHeader()
    for videoName in videoNames:
        processVideo(videoName)

if __name__ == '__main__':
    startAnalysis()