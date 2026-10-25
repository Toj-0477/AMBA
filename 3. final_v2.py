"""
3. final_v2.py - the final model. Run it from this folder after notebooks 1 and 2:  python "3. final_v2.py"

What the EDA (2. eda_analysis.ipynb) showed:
  * sales = price x units sold, and the STORE decides how many units a typical row sells
  * once the store is removed, the only column with a real signal is the ITEM NUMBER (the 15 in FDA15):
    it moves demand by about +-10% in a smooth wave
  * every other column is at chance level and does not improve the predictions

So the model uses just three columns:  Outlet_Identifier (store), Item_MRP (price), Item_Identifier (number part).

    predicted sales = typical units of the store  x  item-number effect  x  price

The first two ingredients come from the training rows only; the item-number effect is a Ridge model on a smooth curve.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer

# ---- load: the training file has the sales, the test file does not ---------------------------------------------
train = pd.read_csv("train_cl.csv")
test = pd.read_csv("test_cl.csv")
both = pd.concat([train.drop(columns="Item_Outlet_Sales"), test], ignore_index=True)    # train rows first, then test
n = len(train)

# ---- the three columns we use ----------------------------------------------------------------------------------
store = both["Outlet_Identifier"].values
price = both["Item_MRP"].values
number = both["Item_Identifier"].str[3:].astype(int).values                             # FDA15 -> 15
sales = train["Item_Outlet_Sales"].values
units = sales / price[:n]                                                               # units sold = sales / price

# smooth curve of the item number (18 columns). It uses no sales, so it is safe to build on train + test together.
curve = SplineTransformer(n_knots=16, degree=3).fit_transform(number.reshape(-1, 1))


def predict_sales(learn, predict):
    """Learn from the rows `learn` (positions in `both`, all training rows) and return sales for the rows `predict`."""
    store_mean = pd.Series(units[learn]).groupby(store[learn]).mean()               # typical units of each store
    mean_learn = pd.Series(store[learn]).map(store_mean).values
    mean_predict = pd.Series(store[predict]).map(store_mean).values
    relative_units = units[learn] / mean_learn                                      # 1.0 = a normal row of that store
    effect = Ridge(alpha=10).fit(curve[learn], relative_units)                      # item number -> relative units
    return mean_predict * effect.predict(curve[predict]) * price[predict]


# ---- how good is it? 5-fold cross-validation, repeated 3 times (the sales of each hidden fold are known) ----------
scores = []
for seed in range(3):
    predicted = np.zeros(n)
    for learn, check in KFold(5, shuffle=True, random_state=seed).split(np.arange(n)):
        predicted[check] = predict_sales(learn, check)
    scores.append(np.sqrt(np.mean((sales - predicted) ** 2)))
print(f"CV RMSE: {np.mean(scores):.2f}  (+/- {np.std(scores):.2f} over 3 repeats)")

# ---- final model: learn from ALL training rows, predict the test rows --------------------------------------------
final = predict_sales(np.arange(n), np.arange(n, len(both)))
submission = test[["Item_Identifier", "Outlet_Identifier"]].copy()
submission["Item_Outlet_Sales"] = np.clip(final, 0, None)
submission.to_csv("Submissions/submission_final.csv", index=False)
print("saved Submissions/submission_final.csv", submission.shape)
