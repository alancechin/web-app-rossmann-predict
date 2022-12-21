import pandas as pd
import os
import pickle
## Importando classe Rossmann do arquivo Rossmann.py da pasta rossmann
#from nome_pasta.nome_arquivo import class_in_arquivo
from rossmann.Rossmann import Rossmann

## Biblioteca para construir interfaces(app) Web em Python para construir API´s
from flask import Flask, request, Response

# loading model
model = pickle.load(open('model/model_xgb_rossmann.pkl', 'rb'))

# instaciando objeto da classe Flask que será a API
app = Flask( __name__ )

# criando endpoint com método POST (envia algum dado para poder receber)
## método GET (pede algum dado para poder receber)
@app.route('/rossmann/predict', methods = ['POST'])

def rossmann_predict():
    test_json = request.get_json() ## Classe request com método get_json() para puxar o dado enviado para a API

    if test_json: # Para checar se há dado
        if isinstance(test_json, dict): #unique example/observations/sample in dict
            test_raw = pd.DataFrame(test_json, index = [0])

        else: #multiple example/observations/sample in dict (dict aninhado)
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys() )

        ## Intanciando objeto da classe Rossmann
        pipeline = Rossmann()

        # data cleaning - começo a usar os métodos da classe Rossmann criada
        df1 = pipeline.data_cleaning( test_raw )

        # feature engineering
        df2 = pipeline.feature_engineering( df1 )

        # data preparation
        df3 = pipeline.data_preparation( df2 )

        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response ## Dado resposta para a entidade que solitou algo via dado


    else:

        return Response( '{}', status = 200, mimetype = 'application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run( host = '0.0.0.0', port = port ) ## dizer para endpoint rodar no localhost (rodando na máquina)
# 192.168.0.6 -> endereço IPv4 pc local
