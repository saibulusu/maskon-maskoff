import os
import uuid
# import cv2 as cv2
from cv2 import cv2
import numpy as np
from base64 import b64encode
from flask import Flask, request, render_template
from easydict import EasyDict as edict

from azure.cosmos import CosmosClient, PartitionKey, exceptions
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType

# Credit: https://github.com/AIZOOTech/FaceMaskDetection
from FaceMaskDetection.tensorflow_infer import inference

from PIL import Image

# Get environmental variables
from dotenv import load_dotenv
load_dotenv()

# Load the values from environmental variables, how magical!
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

endpoint = os.getenv('COSMOS_URI')
key = os.getenv('COSMOS_KEY')

client = CosmosClient(endpoint, key)

database_name = 'db_Masks'
database = client.create_database_if_not_exists(id=database_name)

container_name = 'Tasks'
container = database.create_container_if_not_exists(
    id=container_name, 
    partition_key=PartitionKey(path="/city"),
    offer_throughput=400
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_city_name(argument):
    if argument == "hong-kong":
        return "Hong Kong"
    elif argument == "london":
        return "London"
    elif argument == "madison":
        return "Madison"
    elif argument == "new-delhi":
        return "New Delhi"
    elif argument == "paris":
        return "Paris"
    elif argument == "seattle":
        return "Seattle"

# Create the core Flask app
app = Flask(__name__)

item_id = 1

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # User is requesting the form
        return render_template('form.html')
    elif request.method == 'POST':
        # User has sent us data
        image = request.files['image']
        city = request.form.get('cities')
        if image.filename == '' or not allowed_file(image.filename):
            # Bad input
            return render_template('error.html')

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

        face_count = len(output)
        if face_count > 0:
            non_mask_count = face_count - mask_count
            score = non_mask_count / face_count
            json_data = edict({'id': str(uuid.uuid4()), 'city': city, 'face_count': face_count, 'mask_count': mask_count, 'non_mask_count': non_mask_count, 'score': score})
            container.create_item(body=json_data)

        return render_template('result.html', city=get_city_name(city), face_count=len(output), mask_count=mask_count)
