import io
import utils
import config

from models import CACNet
from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify, request


app = Flask(__name__, static_folder='./static')
run_with_ngrok(app)

@app.route('/crop/image', methods=['POST'])
def predict():
    global model

    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        input_image = io.BytesIO(img_bytes)

        best_crop = utils.apply_cacnet(input_image)

        best_crop.save('./static/output.png')

        return jsonify({'result': 'static/output.png'})


if __name__ == '__main__':
    model = CACNet()
    model = model.to(config.DEVICE)

    if utils.can_load_checkpoint():
        utils.load_checkpoint(model)

    app.run()
