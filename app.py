from flask import Flask, request, jsonify, render_template
from encode_schema import get_seq_concolutional_array
from keras.models import load_model
import numpy as np

app = Flask(__name__, static_folder='static')

class Pred():
    def __init__(self, model_path='./model_save/model_'):
        self.item_model_path = model_path + str(0) + '.hdf5'

    def sample_predict(self, string, str_len=600):
        if len(string) >= str_len:
            string = string[:str_len]
        else:
            string = string + 'Z' * (str_len - len(string))
        assert len(string) == str_len

        string_vector = get_seq_concolutional_array(string)
        string_vector = np.expand_dims(string_vector, 0)

        self.model = load_model(self.item_model_path)
        predict = self.model.predict(string_vector)[0]
        result = predict.tolist()
        
        return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sequence = request.form['sequence']
    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400
    
    predictor = Pred()
    prediction = predictor.sample_predict(sequence)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
