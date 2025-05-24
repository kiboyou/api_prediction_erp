# Au début de app.py
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from fastapi import FastAPI
import numpy as np
import uvicorn

from model import product_model, quantity_model, X_prod, X_quant, le_product, le_customer
from utils import encode_features, create_product_input, create_quantity_input
from request import FullClientRequest, ProductPredictionRequest, QuantityPredictionRequest



app = FastAPI(
    title="API de Prédiction ERP",
    description="Prédit les produits et quantités à partir des données client. Utilise des modèles de classification et de régression.",
    version="1.0.0"
)

predictions_log = []


@app.post("/predict", summary="Prédire produit et quantité")
def predict(request: FullClientRequest):
    encoded = encode_features(request.dict())
    input_cls = create_product_input(encoded | {'Price': request.Price, 'Cost': request.Cost})
    pred_product_encoded = product_model.predict(input_cls)[0]
    pred_product = le_product.inverse_transform([pred_product_encoded])[0]

    input_reg = create_quantity_input({
        'ProductName': pred_product_encoded,
        **encoded,
        'Price': request.Price,
        'Cost': request.Cost
    })

    pred_quantity = int(quantity_model.predict(input_reg)[0])
    result = {
        "PredictedProduct": pred_product,
        "PredictedQuantity": pred_quantity
    }
    predictions_log.append({
        "CustomerName": request.CustomerName,
        **result
    })
    return result


@app.post("/predict_product", summary="Prédire le produit")
def predict_product(request: ProductPredictionRequest):
    encoded = encode_features(request.dict())
    input_data = create_product_input(encoded | {'Price': request.Price, 'Cost': request.Cost})
    pred_encoded = product_model.predict(input_data)[0]
    predicted_product = le_product.inverse_transform([pred_encoded])[0]
    return {"PredictedProduct": predicted_product}


@app.post("/predict_quantity", summary="Prédire la quantité")
def predict_quantity(request: QuantityPredictionRequest):
    encoded = encode_features(request.dict() | {'ProductName': request.ProductName}, encode_product=True)
    input_data = create_quantity_input(encoded | {'Price': request.Price, 'Cost': request.Cost})
    pred_quantity = int(quantity_model.predict(input_data)[0])
    return {"PredictedQuantity": pred_quantity}


@app.get("/predict_all", summary="Prédire pour tous les clients")
def predict_all_clients():
    all_clients = X_prod['CustomerName'].unique()
    results = []

    for client_id in all_clients:
        client_info = X_prod[X_prod['CustomerName'] == client_id]
        if client_info.empty:
            continue

        pred_products = product_model.predict(client_info)
        most_common_encoded = np.bincount(pred_products).argmax()

        client_product_info = X_quant[
            (X_quant['CustomerName'] == client_id) &
            (X_quant['ProductName'] == most_common_encoded)
        ]

        if client_product_info.empty:
            quantity = "Inconnu"
        else:
            quantity = int(quantity_model.predict(client_product_info)[0])

        results.append({
            "CustomerName": le_customer.inverse_transform([client_id])[0],
            "PredictedProduct": le_product.inverse_transform([most_common_encoded])[0],
            "PredictedQuantity": quantity
        })
    return results


@app.get("/predictions", summary="Historique des prédictions")
def get_predictions():
    return predictions_log


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)