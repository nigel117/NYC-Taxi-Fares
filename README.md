# NYC Taxi Fare Predictions 🚖💵

## 📝 Description  
Entry for the **Taxi Trip Fare Prediction Challenge** (hosted by Gaurav Dutta on Kaggle):  
> *"This hackathon provides a historical dataset of NYC taxi trip details and fares. Participants must build ML models to predict trip fares using relevant trip features."*  
*(Dutta, 2022)*  

---

## 🛠️ Technologies  
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn)  
- **Machine Learning**: K-Means Clustering, Decision Trees  
- **Data Analysis**: Entropy Calculation, Outlier Detection  

---

## 🚀 Key Steps  

### 1. Data Cleaning 🧹  
Processed **55M+ records** by:  
- Removing missing values  
- Filtering unrealistic fares/distances (e.g., $0 trips, extreme outliers)  
- Visualizing distributions for validation  

| Before Cleaning | After Cleaning |
|----------------|----------------|
| <img src="https://github.com/user-attachments/assets/934ffc07-aa21-4867-a722-0dfb3ed89c0b" width="300"> | <img src="https://github.com/user-attachments/assets/48d242b3-ea74-49d2-9bd1-0a78e2d38bd9" width="300"> |

---

### 3. Feature Correlation Analysis 🔗

Before building fare predictions, I analyzed potential relationships between fare amounts and other variables to identify useful features:

#### Key Insights:
- **Passenger Count**: Minimal correlation with fare amount  
- **Time of Day**: Slightly higher fares during night hours  
- **Weekdays vs Weekends**: No significant fare differences  

#### Correlation Visualizations:
<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/c458cfff-c97e-4897-a718-0603aa43f482" width="280" alt="Passenger Count vs Fare">
  <img src="https://github.com/user-attachments/assets/41733e24-20ee-45f3-827d-53bb56d2de8e" width="280" alt="Day/Night Fare Distribution">
  <img src="https://github.com/user-attachments/assets/c2bdfb90-1e91-4f43-9dfc-4f45b610d69f" width="280" alt="Weekday Fare Patterns">
</div>

#### Findings:
- Distance showed the **strongest correlation (0.82)** with fare amount  
- Other features had correlations < 0.1 and were excluded from final models  
- Visual analysis confirmed machine learning feature importance results




### 2. Distance & Fare Categorization 📊  
**Distance Ranges (km):**  
- `0-5` km  
- `5-10` km  
- `10-15` km  
- `15-20` km  

**Fare Ranges ($):**  
| Distance (km) | Fare Range ($) |  
|--------------|---------------|  
| 0-5          | 2.5 - 23.0     |  
| 5-10         | 6.0 - 37.87    |  
| 10-15        | 17.7 - 45.0    |  
| 15-20        | 26.5 - 57.54   |  

---

### 3. Entropy Analysis 🔍  
Calculated **Shannon Entropy** to quantify uncertainty:  
- **80% confidence** in predicting fare given distance (or vice versa)  

<img src="https://github.com/user-attachments/assets/2656487d-25fd-4937-bc7f-d743d2e0cbf2" width="500">  

---

## 📂 Repository Structure  
