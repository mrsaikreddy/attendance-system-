import cv2
import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import joblib

app = Flask(__name__)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# Train the model on the faces
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        user_images = os.listdir(f'static/faces/{user}')
        for imgname in user_images:
            img = cv2.imread(f'static/faces/{user}/{imgname}', cv2.IMREAD_GRAYSCALE)
            if img is not None:
                resized_img = cv2.resize(img, (100, 100))  # Resize images to a standard size
                faces.append(resized_img.ravel())
                labels.append(user)
    if faces:
        faces = np.array(faces)
        labels = np.array(labels)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    return faces

def identify_face(facearray):
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(facearray.reshape(1, -1))
    except Exception as e:
        print(f"Model load error: {e}")
        return ["Unknown"]

@app.route('/')
def home():
    datetoday = datetime.now().strftime("%m_%d_%y")
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        names = df.get('Name', []).tolist()
        rolls = df.get('Roll', ["N/A"] * len(df)).tolist()
        times = df.get('Time', []).tolist()
        l = len(df)
    else:
        names, rolls, times, l = [], [], [], 0

    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=len(os.listdir('static/faces')), datetoday=datetoday, zip=zip)

@app.route('/start', methods=['GET'])
def start():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            predicted_user = identify_face(gray)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_user, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            add_attendance(predicted_user)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return home()

def add_attendance(name):
    datetoday = datetime.now().strftime("%m_%d_%y")
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    now = datetime.now().strftime('%H:%M:%S')
    new_entry = pd.DataFrame({'Name': [name], 'Time': [now]})

    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(attendance_file, index=False)
    else:
        new_entry.to_csv(attendance_file, index=False)


@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        username = request.form['username']
        user_folder = f'static/faces/{username}'
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
            cap = cv2.VideoCapture(0)
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                faces = extract_faces(frame)
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    cv2.imwrite(f'{user_folder}/{count}.jpg', face)
                    count += 1
                if count >= 5:  # Capture 5 images for each user
                    break
            cap.release()
            cv2.destroyAllWindows()
            train_model()
        return home()
    return render_template('add_user.html')

if __name__ == '__main__':
    app.run(debug=True)
