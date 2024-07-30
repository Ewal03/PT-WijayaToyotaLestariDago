from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

app = Flask(__name__)

modelfile = open('trained_models.pkl', 'rb')
model = pickle.load(modelfile, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', Job_Desc=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    Model, TahunKendaraan,Kota,Job_Desc = [x for x in request.form.values()]

    data = []
    x_trans = OrdinalEncoder()
    data.append(str(Model))
    data.append(TahunKendaraan)
    data.append(str(Kota))
    data.append(str(Job_Desc))
    
    print(data)
    # #mengkodekan semua value menjadi ordinal
  

    x_trans = OrdinalEncoder()
    X = x_trans.fit_transform(data)

    print(X)      
    X = np.reshape(np.ravel(X), (16, 1))   

    print(X)

    prediction = model.predict([X])

    output = round(prediction[0], 1)

    return render_template('index.html', Job_Desc=output, 
                           Model=Model,TahunKendaraan=TahunKendaraan, Kota=Kota)


if __name__ == '__main__':
    app.run(debug=True)