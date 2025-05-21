# Wine Quality Prediction
![alt](https://www.coravin.dk/cdn/shop/articles/AdobeStock_451146882_c67abea8-b0f0-4357-aba2-35f409370f5a.jpg)

## Overview
This project focuses on predicting wine quality based on various physicochemical properties. Using machine learning techniques, wines are classified into different quality categories to support decision-making in the wine production industry.

## Dataset
- **Source**: [Wine Quality Dataset](https://www.kaggle.com/datasets/stephanierodriguezz/wineqt)
- **Description**: This dataset contains physicochemical properties of red and white wines, such as pH, alcohol content, and acidity, along with their quality ratings.

## Methods
1. **Data Preprocessing**:
   - Normalizing continuous features.
   - Splitting the data into training and testing sets.

2. **Modeling**:
   - Machine learning models used:
     - Linear Regression
     - Decision Tree
     - Random Forest
     - KNN
     - Gradient Boosting

3. **Evaluation**:
   - Metrics: Accuracy Score.

## Results
- The Linear Regression model achieved the best performance, with an accuracy of **100%**.

## Technologies and Tools
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn, XGBoost

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Pratham-gupta-235/Machine-Learning-Projects.git
   ```
2. Navigate to the project folder:
   ```bash
   cd WIne\Quality
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```bash
   python main.py
   ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
