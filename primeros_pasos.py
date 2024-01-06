import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time

import math
import re

tips_id = [4,8,12,16,20]
landmark_hist= []
letters_hist = []
dirs_hist = []

max_hist = 40

mov_letter_size = 0
mov_letter = None
max_mov_letter_size = 30

def update_hist(letter, lm):

  global tips_id 
  global landmark_hist
  global letters_hist
  global dirs_hist 
  global max_hist

  x = lm[tips_id[1]].x
  y = lm[tips_id[1]].y

  if len(dirs_hist) == 0:
    dirs_hist.append('-')
  else:
    lm_ant = landmark_hist[-1]
    x_ant = lm_ant[tips_id[1]].x
    y_ant = lm_ant[tips_id[1]].y
    if x_ant == x:
      dirs_hist.append('-')
    elif x_ant < x:
      dirs_hist.append('D')
    else:
      dirs_hist.append('I')

  letters_hist.append(letter)
  landmark_hist.append(lm)
  if len(letters_hist) > max_hist:
    letters_hist.pop(0)
    landmark_hist.pop(0)
    dirs_hist.pop(0)

#-------------------------------------------------------

def draw_letter_by_size(image,letter,size):
  
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_size = size
  font_color = (255,255,255)
  font_thickness = 3

  h,w,_ = image.shape
  text_size, _ = cv2.getTextSize(letter,font, font_size, font_thickness)
  text_w, text_h = text_size
  cv2.putText(image, letter,(int(w/2)-int(text_w/2), int(h/2)+int(text_h/2)), font, font_size, font_color, font_thickness)
  
  return image
#-------------------------------------------------------------
#-------------------------------------------------------------
def moving_letter(letter, pattern = r'D{5,}I{5,}D{5,}|I{5,}D{5,}I{5,}'):
  #comprobar movimiento
  directions = ''.join(dirs_hist)
  directions = re.sub(r'D-D', 'DD', directions)
  directions = re.sub(r'D-I', 'DI', directions)
  directions = re.sub(r'I-I', 'II', directions)
  directions = re.sub(r'I-D', 'ID', directions)
  print(directions)
  matches = re.findall(pattern, directions)

  #comprobar letra
  letters = ''.join(letters_hist)
  pattern2 = r'('+letter+'){20,}'
  matches2 = re.findall(pattern2, letters)

  if len(matches) and len(matches2):
    return True
  return False

#-------------------------------------------------------------

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
detection_result = None


def get_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
  global detection_result
  detection_result = result

def draw_bb_with_letter(image,detection_result,letter):
  
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_size = 3
  font_color = (255,255,255)
  font_thickness = 3

  bb_color = (0,255,0)
  margin = 10
  bb_thickness = 3

  # Loop through the detected hands to visualize.
  hand_landmarks_list = detection_result.hand_landmarks
  for hand_landmarks in hand_landmarks_list:
    
    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    min_x = int(min(x_coordinates) * width) - margin
    min_y = int(min(y_coordinates) * height) - margin
    max_x = int(max(x_coordinates) * width) + margin
    max_y = int(max(y_coordinates) * height) + margin

    # Draw a bounding-box
    cv2.rectangle(image, (min_x,min_y),(max_x,max_y),bb_color,bb_thickness)

    # Get the text size
    text_size, _ = cv2.getTextSize(letter, font, font_size, font_thickness)
    text_w, text_h = text_size
    # Draw background filled rectangle
    cv2.rectangle(image, (min_x,min_y), (min_x + text_w, min_y - text_h), bb_color, -1)  
    # Draw the letter
    cv2.putText(image, letter,(min_x, min_y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
  
  return image


def draw_landmarks_on_image(rgb_image, detection_result):

  hand_landmarks_list = detection_result.hand_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for hand_landmarks in hand_landmarks_list:
 
    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image

#--------------------------------------------------------------------------------------------------------------------------
def finger_info(lm):
  global tips_id
  info=[]
  for tip in tips_id:
    x1 = lm[tip].x
    y1 = lm[tip].y # el -3 nos da la base
    x2 = lm[tip -1].x
    y2 = lm[tip -1].y
    x3 = lm[tip -2].x
    y3 = lm[tip -2].y
    x4 = lm[tip -3].x
    y4 = lm[tip -3].y
    x5 = lm[0].x
    y5 = lm[0].y

    d1 = math.sqrt((x1-x5)**2+(y1-y5)**2)
    d2 = math.sqrt((x2-x5)**2+(y2-y5)**2)
    d3 = math.sqrt((x3-x5)**2+(y3-y5)**2)
    d4 = math.sqrt((x4-x5)**2+(y4-y5)**2)

    max_d = max([d1,d2,d3,d4])
    extended = 0

    if d1 == max_d:
      extended =1
    ang = math.atan2(y4 - y1, x1 - x4)*180/np.pi
    info.append((extended, int(ang)))

  print(info)
  return info



def print_pos(detection_result):
  for lm in detection_result.hand_landmarks :
   #print(lm[tips_id[1]].x, lm[tips_id[1]].y, lm[tips_id[1]].z)
   print(lm[tips_id[1]])


def print_angle(detection_result):
  for lm in detection_result.hand_landmarks :
   x = - lm[tips_id[1]].x + lm[tips_id[1]-3].x # el -3 nos da la base
   y = - lm[tips_id[1]].y + lm[tips_id[1]-3].y # el angulo da negativo, invertimos los operadores para que de positivo
  
   print(math.atan2(y,x)*180/np.pi)

def print_dis(detection_result):
 #la distacia es relativa a la distancia de la camara asi que cuidado, hay que buscar relaciones entre las falanges del dedo.
 for lm in detection_result.hand_landmarks :
   x1 = lm[tips_id[1]].x
   x2 = lm[tips_id[1]-3].x # el -3 nos da la base
   y1 = lm[tips_id[1]].y
   y2 = lm[tips_id[1]-3].y
  
   print(math.sqrt((x1-x2)**2+(y1-y2)**2))
#--------------------------------------------------------------------------------------------------------------------------
def check_extended(info, fingers_ext): # Ejemplo: fingers_ext = [0,1,0,0,1]
  valor_retorno = True
  i = 0
  for e,ang in info: # Bucle en el que comparamos los valores de info con los de fingers_ext
    if (e == None) :
      continue
    if (e != fingers_ext[i]): 
      valor_retorno = False
    i += 1

  return valor_retorno  
  
def check_angle(info, fingers_ang, tolerance): # Si no nos interesa un angulo ponemos un none en lugar de la tolerancia en ese angulo
  i = 0
  for e,ang in info: # Bucle en el que comparamos los valores de info con los de fingers_ext
    if (ang == None) :
      continue
    if (ang > fingers_ang[i] + tolerance) or (ang < fingers_ang[i] - tolerance): 
      # (a > ang - tol) or (a < ang - tol) 
      return 1
    i += 1

    return 0

def draw_letter_A(info, image):
    fingers_ext_A = [0, 0, 0, 0, 0]  # A: Thumb not extended, other fingers extended
    fingers_ang_A = [100, -120, -100, -100, -160]  # A: No specific angle requirements

    if check_extended(info, fingers_ext_A) and check_angle(info, fingers_ang_A, 10):
      letter = 'A'
      image = draw_bb_with_letter(image, detection_result, letter)
    return image

def draw_letter_E(info, image):
    fingers_ext_E = [1, 0, 0, 0, 0]  # A: Thumb not extended, other fingers extended
    fingers_ang_E = [70, -90, -90, -90, -90]  # A: No specific angle requirements

    if check_extended(info, fingers_ext_E) and check_angle(info, fingers_ang_E, 20):
      letter = 'E'
      image = draw_bb_with_letter(image, detection_result, letter)
    return image

def draw_letter_I(info, image):
    fingers_ext_I = [1, 0, 0, 0, 1]  # I: Thumb not extended, other fingers extended
    fingers_ang_I = [100, -90, -90, -90, 90]  # I: No specific angle requirements

    if check_extended(info, fingers_ext_I) and check_angle(info, fingers_ang_I, 10):
      letter = 'I'
      image = draw_bb_with_letter(image, detection_result, letter)
    return image

def draw_letter_P(info, image):
    fingers_ext_P = [1, 1, 1, 1, 0]  # I: Thumb not extended, other fingers extended
    fingers_ang_P = [100, 90, 90, 90, -40]  # I: No specific angle requirements

    if check_extended(info, fingers_ext_P) and check_angle(info, fingers_ang_P, 10):
      letter = 'P'
      image = draw_bb_with_letter(image, detection_result, letter)
    return image

def draw_letter_U(info, image):
    fingers_ext_U = [1, 1, 1, 0, 0]  # I: Thumb not extended, other fingers extended
    fingers_ang_U = [100, 90, 90, -90, -40]  # I: No specific angle requirements

    if check_extended(info, fingers_ext_U) and check_angle(info, fingers_ang_U, 30):
      letter = 'U'
      image = draw_bb_with_letter(image, detection_result, letter)
    return image

def draw_letter_L(info, image):
    fingers_ext_L = [1, 1, 0, 0, 0]  # I: Thumb not extended, other fingers extended
    fingers_ang_L = [100, 90, -90, -90, -40]  # I: No specific angle requirements

    if check_extended(info, fingers_ext_L) and check_angle(info, fingers_ang_L, 30):
      letter = 'L'
      image = draw_bb_with_letter(image, detection_result, letter)
    return image

def draw_letter_O(info, image):
    fingers_ext_O = [1, 0, 1, 1, 1]  # I: Thumb not extended, other fingers extended
    fingers_ang_O = [130, -180, 110, 100, 90]  # I: No specific angle requirements

    if check_extended(info, fingers_ext_O) and check_angle(info, fingers_ang_O, 20):
      letter = 'O'
      image = draw_bb_with_letter(image, detection_result, letter)
    return image

def draw_letter_B(info, image):
    fingers_ext_B = [1, 1, 1, 1, 1]  # I: Thumb not extended, other fingers extended
    fingers_ang_B = [160, -170, 180, 180, 180]  # I: No specific angle requirements

    if check_extended(info, fingers_ext_B) and check_angle(info, fingers_ang_B, 15):
      letter = 'B'
      image = draw_bb_with_letter(image, detection_result, letter)
    return image

def draw_letter_C(info, image):
    fingers_ext_C = [1, 1, 1, 1, 1]  # I: Thumb not extended, other fingers extended
    fingers_ang_C = [160, 140, 140, 140, 140]   # I: No specific angle requirements

    if check_extended(info, fingers_ext_C) and check_angle(info, fingers_ang_C, 20):
      letter = 'C'
      image = draw_bb_with_letter(image, detection_result, letter)
    return image

def draw_finger_info(info, lm, image):
  
 global tips_id
 font = cv2.FONT_HERSHEY_SIMPLEX
 font_size = 0.5
 font_thickness = 1 
 margin = 10
 cont = 0
 for e,ang  in info:
   #Draw the letter
   h,w,_ = image.shape
   x = int(lm[tips_id[cont]].x * w)
   y = int(lm[tips_id[cont]].y * h) - margin
   if e:
     font_color = (0, 255, 0)
   else:
     font_color = (0, 0, 255)

   cv2.putText(image, str(ang), (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
   cont += 1

 return image
#--------------------------------------------------------------------------------------------------------------------------


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result)

with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
  cap = cv2.VideoCapture(0)
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image = cv2.flip(image,1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    frame_timestamp_ms = int(time.time() * 1000)
    landmarker.detect_async(mp_image, frame_timestamp_ms)
    
    if detection_result is not None:
      image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
      image = draw_bb_with_letter(image,detection_result, '-')
      
      if len(detection_result.hand_landmarks) > 0:
        lm = detection_result.hand_landmarks[0]
        info = finger_info(lm)
        draw_finger_info(info, lm, image)
        #letra = letras_estaticas(info)
        #image = draw_finger_info(info, lm, image)
        #update_hist("H", lm)
        image = draw_letter_A(info, image)  # Llamada a la función para dibujar la letra A
        update_hist("A", lm)
        image = draw_letter_E(info, image)  # Llamada a la función para dibujar la letra A
        update_hist("E", lm)
        image = draw_letter_I(info, image)  # Llamada a la función para dibujar la letra I
        update_hist("I", lm)
        image = draw_letter_P(info, image)  # Llamada a la función para dibujar la letra P
        update_hist("P", lm)
        image = draw_letter_U(info, image)  # Llamada a la función para dibujar la letra U
        update_hist("U", lm)
        image = draw_letter_L(info, image)  # Llamada a la función para dibujar la letra L
        update_hist("L", lm)
        image = draw_letter_O(info, image)  # Llamada a la función para dibujar la letra O
        update_hist("O", lm)
        image = draw_letter_B(info, image)  # Llamada a la función para dibujar la letra B
        update_hist("B", lm)
        image = draw_letter_C(info, image)  # Llamada a la función para dibujar la letra C
        update_hist("C", lm)
        #if mov_letter_size == 0:
            #if moving_letter("H"):
              #mov_letter = "Hola"
              #mov_letter_size = max_mov_letter_size


        #if mov_letter_size > 0:
          #image = draw_letter_by_size(image,mov_letter, mov_letter_size)
          #mov_letter_size -= 1
        
    cv2.imshow('MediaPipe Hands', image)
    #print_pos(detection_result)  
    #print_angle(detection_result)
    #print_dis(detection_result)
    if cv2.waitKey(5) & 0xFF == 27:
      break
  # terminar codigo de foto

