import cv2
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

images = {'son':cv2.imread('init_face/son.jpg')}

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3)

init_face = open('init_face/son.txt', 'w', encoding='utf-8')

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for name, image in images.items():
    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face landmarks of each face.
    print(f'Face landmarks of {name}:')
    if not results.multi_face_landmarks:
      print('fail')
      continue
    annotated_image = image.copy()
    for i, face_landmarks in enumerate(results.multi_face_landmarks):
      if i == 0:
        print(len(face_landmarks.landmark))
        for landmark in face_landmarks.landmark:
          x = landmark.x
          y = landmark.y
          z = landmark.z
          init_face.write(str(x) + '\t' + str(y) + '\t' + str(z) + '\n')
    for face_landmarks in results.multi_face_landmarks:
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS ,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)
    cv2.imwrite('init_face/annotated_' + name + '.jpg', annotated_image)

init_face.close()
