from flask import Flask, request, jsonify
import threading
from main import start_translation_process, stop_translation_process

app = Flask(__name__)

@app.route("/start", methods=["POST"])
def start_translation():
    data = request.get_json()
    url = data.get("url")
    title = data.get("title")
    threading.Thread(target=start_translation_process, args=(url, title)).start()
    return jsonify({"status": "started", "url": url, "title": title})

@app.route("/stop", methods=["POST"])
def stop_translation():
    stop_translation_process()
    return jsonify({"status": "stopped"})

if __name__ == "__main__":
    app.run(port=5000)
