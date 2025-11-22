from flask import Flask, render_template, request, redirect, Response
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"

model = YOLO("yolov8n.pt")
current_count = 0
video_path = None


@app.route('/')
def upload_page():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path

    if "video" not in request.files:
        return "No file uploaded"

    file = request.files["video"]
    if file.filename == "":
        return "No file selected"

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    video_path = save_path
    return redirect("/player")


@app.route('/player')
def player_page():
    return render_template("player.html")


def generate_frames():
    global current_count, video_path

    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True)

        person_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    person_count += 1

                # draw box
                cv2.rectangle(
                    frame,
                    (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                    (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                    (0, 255, 0), 2
                )

        current_count = person_count

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/count')
def get_count():
    return str(current_count)


if __name__ == "__main__":
    app.run(debug=True)
