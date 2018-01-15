import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir)

import json
import numpy as np

import model_helper as m

#Flask
from flask import Flask, render_template, json, jsonify,request,abort

app = Flask(__name__)

config = {
    'MODEL_JSON_DIR':'./model/imperium/imperium_tfidf_vec_30e_valacc_8538.json',
    'MODEL_WEIGHTS_DIR':'./model/imperium/imperium_tfidf_vec_30e_w_valacc_8538.h5',
    'BATCH_SIZE':128,   
    'MAX_SEQUENCE_LENGTH':140,
    'MAX_VOCAB_SIZE':190000,
    'EMBEDDING_DIM':100,
    'TOKENZER_PATH':'./support/tokanizer_imperium_on_train_tfidf.pickle'
}

global saved_model, tf_graph, TextTransformer
saved_model, tf_graph = m.init(config)
TextTransformer = m.TextTransformer(config)

def _prediction(text_transformed):
    with tf_graph.as_default():
        result = saved_model.predict(text_transformed)
    return result
    

@app.route("/")
def main():
    return render_template('home.html')



@app.route("/v1/profanity/prediction",methods=['POST'])
def prediction():
    if not request.json:
        abort(400)
    data = request.json
    if data['text'] is None or data['text'] is False:
        abort(400)
    _text = TextTransformer.pre_proc_text_prod(data['text'])
    result =_prediction(_text)
    return jsonify(result= str(result[0][0]))


if __name__ == "__main__":
    print("App is running.....")
    app.run(host='0.0.0.0', port=9000, debug=True,use_reloader = False)

