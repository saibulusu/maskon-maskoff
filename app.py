import os
import cv2 as cv2
import numpy as np
from base64 import b64encode
from flask import Flask, request, render_template

from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType

# from FaceMaskDetection.load_model.tensorflow_loader import *
from FaceMaskDetection.tensorflow_infer import inference

from PIL import Image

# Get environmental variables
from dotenv import load_dotenv
load_dotenv()

# Load the values from environmental variables, how magical!
COGSVCS_KEY = os.getenv('COGSVCS_KEY')
COGSVCS_CLIENTURL = os.getenv('COGSVCS_CLIENTURL')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = '/'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create the core Flask app
app = Flask(__name__)
app.config['TESTING']=True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # User is requesting the form
        return render_template('form.html')
    elif request.method == 'POST':
        # User has sent us data
        image = request.files['image']
        if image.filename == '' or not allowed_file(image.filename):
            # Bad input
            return render_template('error.html')
        
        # face_client = FaceClient(COGSVCS_CLIENTURL, CognitiveServicesCredentials(COGSVCS_KEY))
        # detected_faces = face_client.face.detect_with_stream(image, detection_model='detection_02')

        filestr = image.read()
        #convert string data to numpy array
        npimg = np.fromstring(filestr, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = inference(img, show_result=False, target_shape=(260, 260))

        # Loop through faces from output and check for masks
        mask_count = 0
        for face in output:
            if face[0] == 0:
                mask_count += 1

        encoded = b64encode(img)
        mime = "image/jpg"
        uri = "data:%s;base64,%s" % (mime, encoded)

        return render_template('result.html', face_count=len(output), mask_count=mask_count, uri=uri)