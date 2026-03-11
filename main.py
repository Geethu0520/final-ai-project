from flask import Flask, Response, jsonify, stream_with_context, render_template_string, request
from flask_cors import CORS
import threading
from surveillance import generate_frames, get_latest_status
from ultralytics import YOLO
import time
import json

app = Flask(__name__)
CORS(app)

streaming_thread = None
video_stream_started = False

model = YOLO("model/best_yolov12.pt")
print(model.names)

@app.route("/")
def home():
    return render_template_string('''
        <h2>Live Surveillance Feed</h2>
        <img src="{{ url_for('video_feed') }}" width="640" height="480" />
        <h3>Status (via /status_feed):</h3>
        <pre id="statusOutput">Waiting...</pre>
        <script>
            const evtSource = new EventSource("/status_feed");
            evtSource.onmessage = function(e) {
                document.getElementById("statusOutput").textContent = e.data;
            };
        </script>
    ''')


@app.route("/start")
def start_surveillance():
    global streaming_thread, video_stream_started
    if not video_stream_started:
        streaming_thread = threading.Thread(target=generate_frames, daemon=True)
        streaming_thread.start()
        video_stream_started = True
        return jsonify({"status": "started", "message": "Streaming started. Visit / to see video."})
    return jsonify({"status": "running", "message": "Already streaming"})


@app.route("/video_feed")
def video_feed():
    view=generate_frames()
    return Response(view, mimetype="multipart/x-mixed-replace; boundary=frame")


# @app.route("/status_feed")
# def status_feed():
#     student_id = request.args.get('studentId', 'unknown')
#     quiz_id = request.args.get('quizId', 'unknown')
#     print("data : ", student_id)
#     print("data : ", quiz_id)
#     def event_stream():
#         while True:
#             status = get_latest_status()
#             status['studentId'] = student_id
#             status['quizId'] = quiz_id
#             print(" status : ", status)
#             yield f"data: {json.dumps(status)}\n\n"
#             time.sleep(1)
#     return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

@app.route("/status_feed")
def status_feed():
    student_id = request.args.get('studentId', 'unknown')
    quiz_id = request.args.get('quizId', 'unknown')
    
    def event_stream():
        while True:
            status = get_latest_status()
            status['studentId'] = student_id
            status['quizId'] = quiz_id
            yield f"data: {json.dumps(status)}\n\n"
            time.sleep(1)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)

