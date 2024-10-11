# house_price_prediction_ml_algo
# House Price Prediction App

This project is a web-based application built using Streamlit that predicts house prices based on various features like area, number of bedrooms, bathrooms, and other amenities. The model uses linear regression for the prediction.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Dataset](#dataset)
- [Model](#model)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)

## Project Description

The House Price Prediction app allows users to input various features about a house (such as area, number of bedrooms, bathrooms, air conditioning, etc.) and get a predicted price. The goal is to help users estimate property values based on historical data.

## Features

- User-friendly web interface to input house details.
- Predicts house prices using a trained linear regression model.
- Sliders and dropdowns for easy data input.
- Displays predicted price in real-time.
- Feature encoding for categorical variables like 'Furnishing Status', 'Basement', etc.

## Technologies Used

- **Python 3.x**
- **Streamlit** for building the web app.
- **Scikit-learn** for model building (Linear Regression).
- **Pandas** and **NumPy** for data manipulation.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the URL provided by Streamlit in your browser to start using the app.

> **Note**: Make sure you have Python installed and the required libraries from `requirements.txt`.

## Dataset

The dataset used for training the model is a housing dataset, containing features like area, number of bedrooms, bathrooms, etc. Here's a sample of the dataset structure:

```csv
price,area,bedrooms,bathrooms,stories,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus.


### 8. **Model**
```markdown
## Model

The model used is a **Linear Regression** model, trained using the scikit-learn library. The following preprocessing steps were performed:

- Categorical variables were encoded.
- Feature scaling was applied where necessary.
- The model was trained using an 80-20 train-test split.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or raise an Issue for suggestions or improvements.
