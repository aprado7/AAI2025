import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Sample dataset ---
data = pd.DataFrame({
    "square_footage": [1000, 1500, 2000, 2500, 3000],
    "location": ["Downtown", "Suburb", "Suburb", "Downtown", "Rural"],
    "price": [300000, 320000, 400000, 500000, 350000]
})

X = data[["square_footage", "location"]]
y = data["price"]

# Preprocessing: one-hot encode location
preprocessor = ColumnTransformer(
    transformers=[
        ('loc', OneHotEncoder(), ['location'])
    ],
    remainder='passthrough'
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train
model.fit(X, y)

# Predict price for a 2000 sq ft Downtown house
new_house = pd.DataFrame({"square_footage": [2000], "location": ["Downtown"]})
pred_price = model.predict(new_house)[0]

print("Predicted Price for 2000 sq ft Downtown house: $", round(pred_price, 2))
print("Model coefficients:", model.named_steps['regressor'].coef_)
print("Intercept:", model.named_steps['regressor'].intercept_)
