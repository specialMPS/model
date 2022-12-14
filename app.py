import json
import tracemalloc

from flask import Flask, request
from model.chat import kogpt2_chat as chatbot
from model.emotion import kobert_emotion as emotion
import os

app = Flask(__name__)


@app.route('/')
def hello_world():
    return "Hello World"


@app.route('/chat')
def chat():
    sent = request.args.get('s')
    if sent is None or len(sent) == 0:
        return json.dumps({
            "answer" : "계속 얘기해주세요~~"
        }, ensure_ascii=False)

    answer = chatbot.predict(sent)
    return json.dumps({
        "answer" : answer
    }, ensure_ascii=False)


@app.route('/emotion')
def emotion():
    sent = request.args.get('s')
    if sent is None or len(sent) == 0:
        return json.dumps({
            "answer" : "계속 얘기해주세요~~"
        }, ensure_ascii=False)

    answer = emotion.predict(sent)
    return json.dumps({
        "emotion" : answer
    }, ensure_ascii=False)


if __name__ == '__main__':
    tracemalloc.start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[TOP 10]")
    for stat in top_stats[:10]:
        print(stat)