from flask import Flask, json, request
import client
import base64
import pickle
import model_maintainer as models

api = Flask(__name__)


@api.route('/prediction', methods=['POST'])
def get_prediction():
    data = pickle.loads(base64.b64decode(request.data))
    print(f'got request /prediction, entries: {len(data)}')
    return base64.b64encode(pickle.dumps(models.make_prediction(data)))


if __name__ == '__main__':
    model_list = [('B', '../data/model_b'), ('A', '../data/model_a')]
    client.initialize(3)
    models.initialize(model_list, 3)
    api.run(port=8080)

