import os
from flask import Flask, request, render_template

from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType

# Get environmental variables
from dotenv import load_dotenv
load_dotenv()

# Load the values from environmental variables, how magical!
COGSVCS_KEY = os.getenv('COGSVCS_KEY')
COGSVCS_CLIENTURL = os.getenv('COGSVCS_CLIENTURL')

# Create the core Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # User is requesting the form
        return render_template('form.html')
    elif request.method == 'POST':
        # User has sent us data
        image = request.files['image']
        face_client = FaceClient(COGSVCS_CLIENTURL, CognitiveServicesCredentials(COGSVCS_KEY))
        detected_faces = face_client.face.detect_with_stream(image)
        return render_template('result.html', size=len(detected_faces))