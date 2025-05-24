from pydantic import BaseModel, Field

class FullClientRequest(BaseModel):
    CustomerName: str = Field(..., example="Béatrice Ben Salah")
    CategoryName: str = Field(..., example="Café en grains entiers")
    Price: float = Field(..., example=16.49)
    Cost: float = Field(..., example=8)
    FavoriteCategory: str = Field(..., example="Café moulu")

class ProductPredictionRequest(BaseModel):
    CustomerName: str = Field(..., example="Béatrice Ben Salah")
    CategoryName: str = Field(..., example="Café en grains entiers")
    Price: float = Field(..., example=16.49)
    Cost: float = Field(..., example=8)
    FavoriteCategory: str = Field(..., example="Café moulu")

class QuantityPredictionRequest(BaseModel):
    ProductName: str = Field(..., example="Light Roast Sumatra")
    CustomerName: str = Field(..., example="Béatrice Ben Salah")
    CategoryName: str = Field(..., example="Café en grains entiers")
    Price: float = Field(..., example=16.49)
    Cost: float = Field(..., example=8)
    FavoriteCategory: str = Field(..., example="Café moulu")