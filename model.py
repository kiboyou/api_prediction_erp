import pandas as pd
import joblib

# Charger les modèles
product_model = joblib.load("./models/product_model.pkl")
quantity_model = joblib.load("./models/quantity_model.pkl")

# Charger les encodeurs
le_product = joblib.load("./models/encoder_product.pkl")
le_customer = joblib.load("./models/encoder_customer.pkl")
le_category = joblib.load("./models/encoder_category.pkl")
le_favorite = joblib.load("./models/encoder_favorite_category.pkl")

# Charger les données pour prédictions groupées
df = pd.read_csv("../data/ERP_dataset_new.csv", encoding="Windows-1252")

df['CustomerName'] = le_customer.transform(df['CustomerName'])
df['ProductName'] = le_product.transform(df['ProductName'])
df['CategoryName'] = le_category.transform(df['CategoryName'])
df['FavoriteCategory'] = le_favorite.transform(df['FavoriteCategory'])

X_prod = df[['CustomerName', 'CategoryName', 'Price', 'Cost', 'FavoriteCategory']]
X_quant = df[['ProductName', 'CustomerName', 'CategoryName', 'Price', 'Cost', 'FavoriteCategory']]
