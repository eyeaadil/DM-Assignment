# Titanic Survival Prediction Web Application

This project is a complete end-to-end machine learning application that predicts passenger survival on the Titanic. It includes data analysis, model training, and a web interface for making predictions.

## 🚀 Features

- **Data Analysis**: Comprehensive EDA and visualization of the Titanic dataset
- **Machine Learning**: Implements and compares multiple models including Random Forest and XGBoost
- **Web Interface**: User-friendly web interface for making predictions
- **REST API**: Flask-based backend serving model predictions
- **Responsive Design**: Mobile-friendly interface built with Tailwind CSS

## 📦 Project Structure

```
.
├── app.py                # Flask web application
├── best_titanic_model.joblib  # Trained model
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface
└── titanic_final_project.py  # Model training and evaluation
```

## 🛠️ Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/eyeaadil/DM-Assignment.git
   cd DM-Assignment
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Usage

### 1. Train the Model
Run the model training script to train and save the model:
```bash
python titanic_final_project.py
```

This will:
- Perform data preprocessing and feature engineering
- Train and evaluate multiple models
- Save the best model as `best_titanic_model.joblib`

### 2. Run the Web Application
Start the Flask development server:
```bash
python app.py
```

Open your browser and visit: http://localhost:5000

## 🌐 Web Interface

The web interface allows you to:
- Input passenger details (class, age, sex, etc.)
- Get instant survival predictions
- View prediction results with confidence indicators

## 🤖 Model Details

- **Algorithms Used**:
  - Random Forest Classifier (tuned with GridSearchCV)
  - XGBoost Classifier (baseline comparison)
- **Features**: Passenger class, sex, age, fare, family size, etc.
- **Performance**: Model accuracy and metrics are displayed during training

## 📊 Data

Based on the classic Titanic dataset, containing information about 891 passengers including:
- Survival status
- Passenger class
- Name, sex, and age
- Number of siblings/spouses/parents/children aboard
- Ticket and fare information
- Port of embarkation

## 📝 Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
