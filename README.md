
# 📈 Time Series Forecasting with Sales Data

This project transforms monthly sales data into a supervised learning problem for forecasting using machine learning models like linear regression or LSTM.

## 🗂 Project Structure

```
.
├── data/
├── train.csv             # Original sales data
├── main.py                   # Main script for preprocessing and modeling
└── README.md                 # This documentation
```

---

## 🔍 What the Code Does

### 1. **Convert date to datetime and aggregate monthly sales**

```python
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.to_period("M")
monthly_sales = df.groupby('date').sum().reset_index()
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
```

> **Explanation:**  
This block parses your `date` column, aggregates the data by month, and converts it back to a timestamp format for plotting.

---

### 2. **Create a supervised learning dataset**

```python
monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna().reset_index(drop=True)

supervised_data = monthly_sales[['sales_diff']].copy()
for i in range(1, 13):
    supervised_data[f'month_{i}'] = supervised_data['sales_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop=True)
```

> **Explanation:**
- `sales_diff` = Difference between current and previous month (helps make the data stationary).
- Creates lag features like `month_1`, `month_2`, ... `month_12`, representing previous months’ sales difference.
- This converts time series into a tabular format suitable for ML models.

---

### 3. **Train-test split**

```python
train_data = supervised_data[:-12]
test_data = supervised_data[-12:]
```

> **Explanation:**  
The last 12 months are held out for testing. The rest is used for training.

---

### 4. **Normalize the data**

```python
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
```

> **Explanation:**  
Normalizes features between -1 and 1 using `MinMaxScaler`. This is crucial for most ML models to perform well.

---

### 5. **Split into features (X) and target (y)**

```python
X_train, y_train = train_data[:,1:], train_data[:,0:1]
X_test, y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
```

> **Explanation:**
- `X` = Previous 12 months’ differences.
- `y` = Sales difference for the current month.
- `.ravel()` flattens the target array into 1D for model compatibility.

---

### 6. **Inverse transform to original scale**

```python
lr_pre_test_set = np.concatenate([lr_pre, X_test], axis=1)
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)
```

> **Explanation:**
- Concatenate predicted y (`lr_pre`) with features to match scaler input.
- Inverse transform to bring predictions back to original sales scale.

---

### 7. **Plotting the results**

```python
plt.plot(monthly_sales['date'], monthly_sales['sales'], color='red', linestyle='--', linewidth=3)
```

> **Explanation:**
This line plots the original sales data with:
- Red dashed line (`--`)
- Line width of 3 for emphasis

---

## ✅ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## 📌 Notes

- This approach is useful for any **univariate time series** forecasting.
- You can replace the regression model with **LSTM**, **XGBoost**, or others after preprocessing.

---
