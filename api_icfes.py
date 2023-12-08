from flask import Flask, request, jsonify
import pandas as pd
import joblib
import pickle

app = Flask(__name__)
model = joblib.load('xgb_model.pkl')

# Cargar el diccionario de mapeo
with open('data/mapping_dict.pkl', 'rb') as f:
    mapping_dict = pickle.load(f)

# Mapeo inverso para convertir la predicción numérica a etiquetas
prediction_mapping = {0: 'bajo', 1: 'medio', 2: 'alto'}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Verificar y convertir entradas categóricas a numéricas según el mapeo
    for feature, value in data.items():
        if feature in mapping_dict:
            if value in mapping_dict[feature]:
                data[feature] = mapping_dict[feature][value]
            else:
                return jsonify({
                    "error": f"Value '{value}' for feature '{feature}' is not valid."
                }), 400
        elif feature not in model.feature_names_in_:
            return jsonify({
                "error": f"Feature '{feature}' is not expected by the model."
            }), 400
    
    # Crear DataFrame de entrada para el modelo
    input_data = pd.DataFrame([data])
    
    # Verificar que no faltan columnas
    missing_features = set(model.feature_names_in_) - set(input_data.columns)
    if missing_features:
        return jsonify({
            "error": "Missing features in the input data",
            "missing_features": list(missing_features)
        }), 400
    
    # Asegúrate de que el DataFrame tenga el mismo orden de columnas que el modelo espera
    input_data = input_data.reindex(columns=model.feature_names_in_)
    
    # Realizar la predicción
    prediction = model.predict(input_data)
    
    # Convertir la predicción numérica a etiquetas
    prediction_labels = [prediction_mapping.get(pred, 'unknown') for pred in prediction]
    
    return jsonify(prediction_labels)

if __name__ == '__main__':
    app.run(debug=True)
