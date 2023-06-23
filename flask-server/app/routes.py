from flask import Blueprint, render_template, request
from app.neural_network import ChurnExitClassifier
from app.transformer import Transformer

import logging

views = Blueprint('views', __name__,
                  template_folder='templates',
                  static_folder='static')


@views.route('/', methods=['GET', 'POST'])
def classifier():
    if request.method == "POST":
        file = request.files['csvFile']
        logging.info(f'File received: {file.filename}')
        logging.info(f'File type: {type(file)}')
        transformer = Transformer(file)
        inputs = transformer.prepare_for_prediction()
        classifier = ChurnExitClassifier()
        classified_data = classifier.predict(inputs)
        data = transformer.get_html(classified_data)
        return render_template('index.html', data=data)
    
    return render_template('index.html')
    