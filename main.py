from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import sqlite3
import math
from datetime import datetime

app = Flask(__name__)

conn = sqlite3.connect(
    "events.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    animal TEXT,
    status TEXT
)
""")

conn.commit()

net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "table",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (320, 240)}
)

picam2.configure(config)
picam2.start()

picam2.set_controls({
    "AwbEnable": False,
    "ColourGains": (1.0, 1.0)
})

last_x = None
last_y = None

still_frames = 0

current_status = "SEARCHING"
current_animal = "NONE"

frame_count = 0

def save_event(animal, status):

    event_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO events
        (time, animal, status)
        VALUES (?, ?, ?)
        """,
        (event_time, animal, status)
    )

    conn.commit()

def generate_frames():

    global last_x
    global last_y
    global still_frames
    global current_status
    global current_animal
    global frame_count

    while True:

        frame = picam2.capture_array()

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR
        )

        frame_count += 1

        found = False

        if frame_count % 5 == 0:

            (h, w) = frame.shape[:2]

            small = cv2.resize(
                frame,
                (160, 120)
            )

            blob = cv2.dnn.blobFromImage(
                cv2.resize(small, (300, 300)),
                0.007843,
                (300, 300),
                127.5
            )

            net.setInput(blob)

            detections = net.forward()

            for i in range(detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:

                    idx = int(
                        detections[0, 0, i, 1]
                    )

                    label = CLASSES[idx]

                    if label in ["cat", "dog"]:

                        found = True

                        current_animal = label.upper()

                        box = detections[0, 0, i, 3:7] * np.array(
                            [w, h, w, h]
                        )

                        (x1, y1, x2, y2) = box.astype("int")

                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        if last_x is not None:

                            distance = math.sqrt(
                                (center_x - last_x) ** 2 +
                                (center_y - last_y) ** 2
                            )

                            if distance < 10:

                                still_frames += 1

                            else:

                                still_frames = 0

                            if still_frames > 5:

                                current_status = "RESTING"

                            else:

                                current_status = "ACTIVE"

                        last_x = center_x
                        last_y = center_y

                        save_event(
                            current_animal,
                            current_status
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            current_animal,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

        if not found:

            current_status = "SEARCHING"

        cv2.putText(
            frame,
            f"STATUS: {current_status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        ret, buffer = cv2.imencode(
            '.jpg',
            frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

@app.route('/')

def index():

    cursor.execute(
        "SELECT COUNT(*) FROM events"
    )

    total_events = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM events
        WHERE status='ACTIVE'
        """
    )

    active_count = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM events
        WHERE status='RESTING'
        """
    )

    resting_count = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM events
        WHERE status='SEARCHING'
        """
    )

    searching_count = cursor.fetchone()[0]

    if total_events > 0:

        active_percent = round(
            active_count / total_events * 100,
            1
        )

        resting_percent = round(
            resting_count / total_events * 100,
            1
        )

    else:

        active_percent = 0
        resting_percent = 0

    cursor.execute(
        """
        SELECT time, animal, status
        FROM events
        ORDER BY id DESC
        LIMIT 10
        """
    )

    rows = cursor.fetchall()

    table_rows = ""

    for row in rows:

        table_rows += f"""
        <tr>
            <td>{row[0]}</td>
            <td>{row[1]}</td>
            <td>{row[2]}</td>
        </tr>
        """

    html = f"""
    <html>

    <head>

    <title>Pet Monitor</title>

    <style>

    body {{
        background: #111;
        color: white;
        font-family: Arial;
        text-align: center;
    }}

    img {{
        border: 3px solid lime;
        border-radius: 10px;
    }}

    table {{
        margin: auto;
        border-collapse: collapse;
        width: 70%;
    }}

    td, th {{
        border: 1px solid white;
        padding: 8px;
    }}

    .box {{
        background: #222;
        padding: 15px;
        margin: 15px;
        border-radius: 10px;
    }}

    </style>

    </head>

    <body>

    <h1>INTELLIGENT PET MONITOR</h1>

    <img src="/video">

    <div class="box">

    <h2>Current Status</h2>

    <p><b>Animal:</b> {current_animal}</p>

    <p><b>Status:</b> {current_status}</p>

    </div>

    <div class="box">

    <h2>Statistics</h2>

    <p>Total Events: {total_events}</p>

    <p>ACTIVE: {active_count}</p>

    <p>RESTING: {resting_count}</p>

    <p>SEARCHING: {searching_count}</p>

    </div>

    <div class="box">

    <h2>Activity Analysis</h2>

    <p>ACTIVE: {active_percent}%</p>

    <p>RESTING: {resting_percent}%</p>

    </div>

    <div class="box">

    <h2>Recent Events</h2>

    <table>

    <tr>
        <th>Time</th>
        <th>Animal</th>
        <th>Status</th>
    </tr>

    {table_rows}

    </table>

    </div>

    </body>

    </html>
    """

    return html

@app.route('/video')

def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

app.run(
    host='0.0.0.0',
    port=5000
)
