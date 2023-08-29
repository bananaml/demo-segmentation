from potassium import Potassium, Request, Response
from pyannote.audio import Model, Inference
import numpy as np
import os
import base64
import matplotlib.pyplot as plt
import io
import requests
from pyannote.core import notebook

MODEL = "pyannote/segmentation"
HF_TOKEN = os.getenv('HF_TOKEN')
SPEAKER_AXIS = 2
TIME_AXIS = 1

app = Potassium("segmentation")

@app.init
def init():
    """
    Initialize the application with the pretrained model.
    """
    model = Model.from_pretrained(MODEL, use_auth_token=HF_TOKEN)
    context = {
        "model": model
    }
    return context

def process_audio(model, hook, label, filename):
    """
    Process the audio file with the given model, hook, label, and filename.
    """
    inference = Inference(model, pre_aggregation_hook=hook)
    prob = inference(filename)
    prob.labels = [label]
    fig, ax = plt.subplots()
    notebook.plot_feature(prob, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

def voice_activity_detection(model):
    to_vad = lambda o: np.max(o, axis=SPEAKER_AXIS, keepdims=True)
    result = process_audio(model, to_vad, 'SPEECH', "/tmp/temp.wav")
    return result

def overlapped_speech_detection(model):
    to_osd = lambda o: np.partition(o, -2, axis=SPEAKER_AXIS)[:, :, -2, np.newaxis]
    result = process_audio(model, to_osd, 'OVERLAP', "/tmp/temp.wav")
    return result

def instantaneous_speaker_counting(model):
    to_cnt = lambda probability: np.sum(probability, axis=SPEAKER_AXIS, keepdims=True)
    result = process_audio(model, to_cnt, 'SPEAKER_COUNT', "/tmp/temp.wav")
    return result

def speaker_change_detection(model):
    to_scd = lambda probability: np.max(
        np.abs(np.diff(probability, n=1, axis=TIME_AXIS)), 
        axis=SPEAKER_AXIS, keepdims=True)
    result = process_audio(model, to_scd, 'SPEAKER_CHANGE', "/tmp/temp.wav")
    return result

@app.handler()
def handler(context: dict, request: Request) -> Response:
    """
    Handle the incoming request and return the response.
    """
    model = context.get("model")
    audio_input = request.json.get("audio")
    option = request.json.get("option")
    response = requests.get(audio_input)
    filename = '/tmp/temp.wav'
    with open(filename, 'wb') as f:
        f.write(response.content)
    if option == "voice_activity_detection":
        image_base64 = voice_activity_detection(model)
    elif option == "overlapped_speech_detection":
        image_base64 = overlapped_speech_detection(model)
    elif option == "instantaneous_speaker_counting":
        image_base64 = instantaneous_speaker_counting(model)
    elif option == "speaker_change_detection":
        image_base64 = speaker_change_detection(model)
    return Response(json={"output": image_base64}, status=200)  

if __name__ == "__main__":
    app.serve()