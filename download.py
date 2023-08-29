from pyannote.audio import Model
import os

HF_TOKEN = os.getenv('HF_TOKEN')

def download_model():
    """Load model"""
    model = Model.from_pretrained("pyannote/segmentation", use_auth_token=HF_TOKEN)

if __name__ == "__main__":
    download_model()