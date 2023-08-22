

import base64
from flask import Flask, request, render_template
from transformers import pipeline
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the word-to-index dictionary
with open("wordtoix.pkl", "rb") as f:
    wordtoix = pickle.load(f)

# Load the index-to-word dictionary
with open("ixtoword.pkl", "rb") as f:
    ixtoword = pickle.load(f)

# Load the image captioning model
model = load_model('model_20.h5')

# Load the image encoding model
resnet = load_model('resnet.h5')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has a file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Read the image file and convert it to a binary string
            img_bytes = file.read()

            # Preprocess the image
            img = Image.open(io.BytesIO(img_bytes))
            img = img.resize((299, 299))
            x = img_to_array(img)
            print(x)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Encode the image
            vec = resnet.predict(x)
            vec = np.reshape(vec, (vec.shape[1]))
            max_length=74

            start = 'startseq'
            for i in range(max_length):
                seq = [wordtoix[word] for word in start.split() if word in wordtoix]
                seq = pad_sequences([seq], maxlen=max_length)
                yhat = model.predict([vec.reshape(1, 2048), seq])
                yhat = np.argmax(yhat)
                word = ixtoword[yhat]
                start += ' ' + word
                if word == 'endseq':
                    break
            caption = start.split()[1:-1]
            caption = ' '.join(caption)
            print(caption)
            img_b64 = base64.b64encode(img_bytes).decode()
            # Pass the generated caption to the HTML template
            return render_template('index2.html', caption=caption, img_b64=img_b64)
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True)




