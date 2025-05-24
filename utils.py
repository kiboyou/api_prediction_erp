from model import le_customer, le_category, le_favorite, le_product
from fastapi import HTTPException
import pandas as pd

def encode_features(request_data, encode_product=False):
    try:
        customer = le_customer.transform([request_data['CustomerName']])[0]
        category = le_category.transform([request_data['CategoryName']])[0]
        favorite = le_favorite.transform([request_data['FavoriteCategory']])[0]
        product = None
        if encode_product:
            product = le_product.transform([request_data['ProductName']])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur dans l'encodage des valeurs: {e}")

    return {
        'CustomerName': customer,
        'CategoryName': category,
        'FavoriteCategory': favorite,
        'ProductName': product
    }

def create_product_input(data):
    return pd.DataFrame([{
        'CustomerName': data['CustomerName'],
        'CategoryName': data['CategoryName'],
        'Price': data['Price'],
        'Cost': data['Cost'],
        'FavoriteCategory': data['FavoriteCategory']
    }])


def create_quantity_input(data):
    return pd.DataFrame([{
        'ProductName': data['ProductName'],
        'CustomerName': data['CustomerName'],
        'CategoryName': data['CategoryName'],
        'Price': data['Price'],
        'Cost': data['Cost'],
        'FavoriteCategory': data['FavoriteCategory']
    }])
