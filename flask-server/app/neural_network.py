from app.singleton import GlobalMLOPS


class ChurnExitClassifier():

    def predict(self, inputs):
        import logging
        weights, biases = GlobalMLOPS().model.layers[0].get_weights()
        logging.info(f'Weights: {weights}')
        logging.info(f'Biases: {biases}')
        return GlobalMLOPS().model.predict(inputs)