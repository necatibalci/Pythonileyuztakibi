import cv2
import face_recognition


video_path = "video.mp4"
video_capture = cv2.VideoCapture(video_path)


known_face_encodings = []


known_face_ids = []


current_id = 0

while True:

    ret, frame = video_capture.read()

    if not ret:
        break

   
    rgb_frame = frame[:, :, ::-1]

   
    face_locations = face_recognition.face_locations(rgb_frame)

  
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

 
    for face_encoding, face_location in zip(face_encodings, face_locations):
       
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_id = None

        if True in matches:
         
            first_match_index = matches.index(True)
            face_id = known_face_ids[first_match_index]
        else:
           
            known_face_encodings.append(face_encoding)
            face_id = current_id
            known_face_ids.append(face_id)
            current_id += 1

     
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)


    cv2.imshow('Video Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
