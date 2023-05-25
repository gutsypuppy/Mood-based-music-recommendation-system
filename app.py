from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import pandas as pd
import random
import cv2
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

classes = {0:['https://open.spotify.com/embed/track/5zGHViDOEuNaZySWpLgPw6?si=RKt5QeqiSvOXmZe7LSirQA','https://open.spotify.com/embed/track/4jLUiSNFQvOb5EKsid8YHn?si=p7TfKDZpRoyTUo9irFRkDQ','https://open.spotify.com/embed/track/6d1VAGIsaxCDMaQfVdLtRC?si=z-42eMIiR2Cyf4vm7nOyTg','https://open.spotify.com/embed/track/6d1VAGIsaxCDMaQfVdLtRC?si=z-42eMIiR2Cyf4vm7nOyTg'],
           1:['https://open.spotify.com/embed/track/3USxtqRwSYz57Ewm6wWRMp?si=C7XvmuChQc-NeZpbrmKSbA','https://open.spotify.com/embed/track/4TgxFMOn5yoESW6zCidCXL?si=C76eGF5uQqekaJIRovCfeg','https://open.spotify.com/embed/track/6dG2zPUOWXk3eMC7Hb3wh3?si=pWROmN9GQSCcmNwIkcYlxg'],
           2:['https://open.spotify.com/embed/track/6H7fLdt0AeWpuxUKXuXWrx?si=IShBh_BNR5OGasn86AJLyA'],
           3:['https://open.spotify.com/embed/track/2JAsg6txhUsMgRaKbLkCmm?si=R06U1XskRNKtFxVVWS1XkQ','https://open.spotify.com/embed/track/6Z40IRipd6pNcUULY6SXng?si=Qfyo1e53TtWJKjKTrMj58w','https://open.spotify.com/embed/track/0gE8ZP2cTXpRDz53bTkpu6?si=R_aaU3ZLRMizkshR3n--Rw','https://open.spotify.com/embed/track/2mFkD2KjL1dUhkpNKteaKP?si=S32K2L8QTzyuEgI-CpflbA'],
           4:['https://open.spotify.com/embed/track/6aPMWbbdhDhiJHlknZb9Yx?si=-nUxQVpqSGSBqVGYsp7m2Q','https://open.spotify.com/embed/track/0czcoKJbJt08NqKrvSbbz7?si=RknDwa6tSfmW8CCrtQR3JQ','https://open.spotify.com/embed/track/45xajAqfrDhxpIOV2dBkYp?si=tfM6HDt5R--Z30Trt6xOiw','https://open.spotify.com/embed/track/18r28YOE2nO8hU0bv0lmcv?si=yf2M9-gzSGGlDGtHVP6U2g'],
           5:['https://open.spotify.com/embed/track/6nssUBefI5iewifpENzsqL?si=6dSSxDtwTSGXpy3hG7jjag','https://open.spotify.com/embed/track/3tslb9CtxCLBx5gdLgeZjA?si=5H9uSFIZR-eIg36g4rMpzg&context=spotify%3Aplaylist%3A37i9dQZF1DWWokU1YHhufF'],
           6:['https://open.spotify.com/embed/track/0TK2YIli7K1leLovkQiNik?si=F9HMHgRGSKKXTiwLfLLhQA','https://open.spotify.com/embed/track/3RiPr603aXAoi4GHyXx0uy?si=Ng_WtY_GQSat1dsUgtW3-Q']}
mood=['Why so serious? son','What made you feel disgusting?','What are you afraid of?',"Someone's looking happy",'you look so Sad :( ','Suprised','Give some expressions!!']
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/', methods=['POST'])
def get_diagnosis():
    new_model= tf.keras.models.load_model('Final_Model.h5')
    path = 'haarcascade_frontalface_default.xml'
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    rectangle_bgr = (255, 255, 255)
    img = np.zeros((500, 500))
    text = "Some text in box!!"
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    text_offset_x = 10
    text_offset_y = img.shape[0] - 25
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

    
    file = request.files['imageFile']
    img = Image.open(file).convert('RGB')
    print(img)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    face_roi = None
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        
        if len(facess) == 0:
            print("Face Not Detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey:ey + eh, ex:ex + ew]
        if not face_roi.size:  # Check if face_roi is empty
            print("Face Not Detected")
            continue
        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

        Predictions = new_model.predict(final_image)
    

    # Return the diagnosis result and the image as a JSON response
    response = {
        'mood': mood[np.argmax(Predictions)],
        'class': random.choice(classes[np.argmax(Predictions)])
    }
    print(response)
    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True)
