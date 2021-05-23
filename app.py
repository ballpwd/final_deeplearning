# Flask
from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
from tensorflow.keras.models import load_model

# Some utilites
import numpy as np
from util import base64_to_pil
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Declare a flask app
app = Flask(__name__)

# Find Model Path
MODEL_PATH = 'models/saved_model/imageClassification'

# Load Trained Model
model = load_model(MODEL_PATH)


# Print Port
print('Model loaded. Check http://127.0.0.1:5000/')

# model predict function  
def model_predict(img, model):

    classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #Resize the image
    from skimage.transform import resize
    resized_image = resize(img, (32,32,3))

    #Get model predctions
    predictions = model.predict(np.array([resized_image]))

    #Show the prediction
    print(predictions)

    #Sort the predictions from least to greatest
    list_index = [0,1,2,3,4,5,6,7,8,9]
    x = predictions

    for i in range(10):
        for j in range(10):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp

    #Show the sed laels in order
    print(list_index)

    # Process your result for human
    result = classification[list_index[0]]
    probability = predictions[0][list_index[0]] * 100, '%'
    print(result,probability)
    
    return [result,probability]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        upimg = base64_to_pil(request.json)
        # Save the image to ./uploads
        upimg.save("./uploads/image.png")
        # Read Image
        img = plt.imread('./uploads/image.png')
       
        # Make prediction
        preds = model_predict(img, model)

        # Return Result as Json
        return jsonify(result=preds[0], probability=preds[1])

    return None


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
