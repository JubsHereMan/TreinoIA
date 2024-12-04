
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Inicializar o app Flask
app = Flask(__name__)

# Carregar o modelo de regressão treinado
model = joblib.load("model_regressor_intensidade.pkl")

# Definir o limiar para a classificação
limiar_intensidade = 1.5  # ajuste este valor conforme necessário

# Definir o endpoint de previsão
@app.route('/predict', methods=['POST'])
def predict():
    # Obter os dados JSON enviados pela requisição
    data = request.get_json()
    
    # Validar a presença dos campos necessários
    if not all(k in data for k in ("contagem_problemas_eletricos", "contagem_problemas_mecanicos")):
        return jsonify({"error": "Campos faltantes"}), 400
    
    # Extrair as contagens dos problemas
    contagem_eletricos = data["contagem_problemas_eletricos"]
    contagem_mecanicos = data["contagem_problemas_mecanicos"]
    
    # Transformar os dados para o formato esperado pelo modelo
    input_data = np.array([[contagem_eletricos, contagem_mecanicos]])
    
    # Fazer a previsão de intensidade
    intensidade_pred = model.predict(input_data)[0]
    
    # Classificar o tipo de problema com base na intensidade
    tipo_problema = "Mecânico" if intensidade_pred > limiar_intensidade else "Eletricidade Geral"
    
    # Retornar a resposta em JSON
    response = {
        "intensidade_predita": intensidade_pred,
        "tipo_problema": tipo_problema
    }
    return jsonify(response)

# Rodar o app Flask
if __name__ == '__main__':
    app.run(debug=True)
