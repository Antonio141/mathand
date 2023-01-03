from unittest import result
import numpy as np

import numpy as np
import cv2,time
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

def genera_equazione_randomica(num_el):
    _ops = ['+','-','*','/']
    while True:
        termini = [np.random.randint(1,11) for i in range(num_el)]
        operatori = [_ops[np.random.randint(0,4)] for i in range(num_el-1)]
        equazione = str()
        for j in range(num_el):
            equazione += str(termini[j])
            if j < num_el-1:
                equazione += str(operatori[j])
        result = eval(equazione)
        if (1 <= result <= 10) and type(result)==int:
            break

    return equazione, result

# num_el = 3
# equazione,result = genera_equazione_randomica(num_el)
# print(equazione,result)

# Prepara per il sistema di gestione della computer vision
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)

# Prepara per il sistema di gioco
start = time.time()
numero_gioco = 0
end_game = 10

num_el = np.random.randint(2,5)
equazione,result = genera_equazione_randomica(num_el)

while numero_gioco <= end_game:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    aggiungi = 0
    cv2.putText(img,str(equazione), (70,100), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)

    if results.multi_hand_landmarks:
        # Conta dita giocatore
        cnt = 0
        aggiungi = 0
        for handOrs, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
            # Get the current hand (left/right)
            handedness_dict = MessageToDict(handOrs)
            orientation = handedness_dict['classification'][0]['label']

            # Crea dizionario per lavorare con conta delle dita per entrambe le mani
            dizionario_dita_left = dict()
            dizionario_dita_right = dict()
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

                if id == 0:
                    cv2.putText(img,str(orientation), (cx+10,cy+10), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,205), 1)
                else:
                    cv2.putText(img,str(id), (cx+1,cy), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,205), 1)
                if orientation == 'Left':
                    falangi = MessageToDict(lm)
                    dizionario_dita_left[id] = falangi
                if orientation == 'Right':
                    falangi = MessageToDict(lm)
                    dizionario_dita_right[id] = falangi
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # conta numero di dita alzate mano sinistra
            if orientation == 'Left':
                if dizionario_dita_left[19]['y'] < dizionario_dita_left[18]['y']:
                    aggiungi += 1

                if dizionario_dita_left[15]['y'] < dizionario_dita_left[14]['y']:
                    aggiungi += 1

                if dizionario_dita_left[11]['y'] < dizionario_dita_left[10]['y']:
                    aggiungi += 1

                if dizionario_dita_left[7]['y'] < dizionario_dita_left[6]['y']:
                    aggiungi += 1

                if dizionario_dita_left[0]['x'] < dizionario_dita_left[1]['x']:
                    if dizionario_dita_left[3]['x'] < dizionario_dita_left[4]['x']:
                        aggiungi += 1
                elif dizionario_dita_left[0]['x'] > dizionario_dita_left[1]['x']:
                    if dizionario_dita_left[3]['x'] > dizionario_dita_left[4]['x']:
                        aggiungi += 1

            # conta numero di dita alzate mano destra
            if orientation == 'Right':
                if dizionario_dita_right[19]['y'] < dizionario_dita_right[18]['y']:
                    aggiungi += 1

                if dizionario_dita_right[15]['y'] < dizionario_dita_right[14]['y']:
                    aggiungi += 1

                if dizionario_dita_right[11]['y'] < dizionario_dita_right[10]['y']:
                    aggiungi += 1

                if dizionario_dita_right[7]['y'] < dizionario_dita_right[6]['y']:
                    aggiungi += 1

                if dizionario_dita_right[0]['x'] < dizionario_dita_right[1]['x']:
                    if dizionario_dita_right[3]['x'] < dizionario_dita_right[4]['x']:
                        aggiungi += 1
                elif dizionario_dita_right[0]['x'] > dizionario_dita_right[1]['x']:
                    if dizionario_dita_right[3]['x'] > dizionario_dita_right[4]['x']:
                        aggiungi += 1


            if result == aggiungi:
                cv2.putText(img,str(equazione), (70,100), cv2.FONT_HERSHEY_PLAIN, 3, (127,255,0), 3)
                num_el = np.random.randint(2,5)
                equazione,result = genera_equazione_randomica(num_el)
                if numero_gioco == end_game:
                    tot = time.time()-start
                    message = 'Total time: %f'%tot
                    cv2.putText(img, message, (70,200), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                    # time.sleep(5)
                numero_gioco += 1

        cv2.putText(img,str(aggiungi), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime



    cv2.imshow("Image", img)
    cv2.waitKey(1)

print(time.time()-start)