import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn

from model import product_model, quantity_model, X_prod, X_quant, le_product, le_customer, df, le_category, le_favorite
from utils import encode_features, create_product_input, create_quantity_input
from request import ClientRequest, FullClientRequest, ProductPredictionRequest, QuantityPredictionRequest

app = FastAPI(
    title="API de Prédiction ERP",
    description="Prédit les produits et quantités à partir des données client. Utilise des modèles de classification et de régression.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictions_log = []


def get_customer_category_details_internal(CustomerName: str, CategoryName: str, ProductName: str = None):
    # Recherche dans df encodé
    try:
        cust_enc = le_customer.transform([CustomerName])[0]
        cat_enc = le_category.transform([CategoryName])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Client ou catégorie inconnus: {str(e)}")

    df_filtered = df[
        (df['CustomerName'] == cust_enc) &
        (df['CategoryName'] == cat_enc)
    ]

    if ProductName:
        try:
            prod_enc = le_product.transform([ProductName])[0]
            df_filtered = df_filtered[df_filtered['ProductName'] == prod_enc]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Produit inconnu: {str(e)}")

    if df_filtered.empty:
        raise HTTPException(status_code=404, detail="Aucune donnée trouvée pour ce client/catégorie/produit.")

    random_row = df_filtered.sample(1).iloc[0]

    # Favorite category decode
    favorite_cat_enc = random_row.get('FavoriteCategory', None)
    if favorite_cat_enc is not None and favorite_cat_enc in le_favorite.classes_:
        favorite_category = le_favorite.inverse_transform([favorite_cat_enc])[0]
    else:
        favorite_category = None

    return {
        "CustomerName": CustomerName,
        "CategoryName": CategoryName,
        "FavoriteCategory": favorite_category,
        "Price": float(random_row['Price']),
        "Cost": float(random_row['Cost'])
    }


@app.post("/predict", summary="Prédire produit et quantité")
def predict(request: ClientRequest):
    # On récupère price/cost via la fonction interne
    details = get_customer_category_details_internal(
        request.CustomerName,
        request.CategoryName,
        getattr(request, 'ProductName', None)
    )

    # Encodage avec prix/cost récupérés
    encoded = encode_features({
        **request.dict(exclude={"Price", "Cost", "ProductName"}),
        "Price": details['Price'],
        "Cost": details['Cost']
    })

    input_cls = create_product_input(encoded | {'Price': details['Price'], 'Cost': details['Cost']})
    pred_product_encoded = product_model.predict(input_cls)[0]
    pred_product = le_product.inverse_transform([pred_product_encoded])[0]

    input_reg = create_quantity_input({
        'ProductName': pred_product_encoded,
        **encoded,
        'Price': details['Price'],
        'Cost': details['Cost']
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
def predict_product(request: ClientRequest):
    details = get_customer_category_details_internal(
        request.CustomerName,
        request.CategoryName,
        getattr(request, 'ProductName', None)
    )
    encoded = encode_features({
        **request.dict(exclude={"Price", "Cost"}),
        "Price": details['Price'],
        "Cost": details['Cost']
    })
    input_data = create_product_input(encoded | {'Price': details['Price'], 'Cost': details['Cost']})
    pred_encoded = product_model.predict(input_data)[0]
    predicted_product = le_product.inverse_transform([pred_encoded])[0]
    return {"PredictedProduct": predicted_product}


@app.post("/predict_quantity", summary="Prédire la quantité")
def predict_quantity(request: ClientRequest):
    # Pour la quantité on a besoin du produit, qui est obligatoire ici
    details = get_customer_category_details_internal(
        request.CustomerName,
        request.CategoryName,
    )
    encoded = encode_features({
        **request.dict(exclude={"Price", "Cost"}),
        "Price": details['Price'],
        "Cost": details['Cost']
    }, encode_product=True)

    input_data = create_quantity_input(encoded | {'Price': details['Price'], 'Cost': details['Cost']})
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


@app.get("/customers", summary="Lister les clients")
def get_customers():
    customers_encoded = X_prod['CustomerName'].unique()
    customers = le_customer.inverse_transform(customers_encoded)
    return {"customers": customers.tolist()}


@app.get("/products", summary="Lister les produits")
def get_products():
    products_encoded = df['ProductName'].unique()
    products = le_product.inverse_transform(products_encoded)
    return {"products": products.tolist()}


@app.get("/categories", summary="Lister les catégories")
def get_categories():
    if 'CategoryName' not in X_prod.columns:
        return {"error": "La colonne 'CategoryName' est absente des données."}
    categories = df['CategoryName'].unique()
    categories = le_category.inverse_transform(categories)
    return {"categories": categories.tolist()}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
