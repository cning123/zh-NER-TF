#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json

from flask import Flask, request
from ner_demo import NER_DEMO,get_args

app = Flask(__name__)
args = get_args()
ner = NER_DEMO(args)

@app.route("/predict/",methods=['POST','GET'])
def predict():
    if request.method == 'GET':
        sentence = request.args['sentence']
    else:
        sentence = request.form['sentence']

    if  sentence:
        entities = ner.predict(sentence)
    # print('x')
        return json.dumps(entities,ensure_ascii=False)
    else:
        return json.dumps({'msg':'can not find sentence',
                            'error_code':'200'})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)