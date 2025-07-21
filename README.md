# Student Exam Performance Predictor

This is an end-to-end machine learning project that predicts a student's math score based on various personal and academic factors. The project covers the entire ML lifecycle, including data ingestion, data transformation, model training, and deployment as a web application.

<!-- 🚀 Live Demo
*[Link to your deployed web application (e.g., on AWS Elastic Beanstalk)]* -->

---

## ✨ Features

- **End-to-End ML Pipeline**: A fully automated pipeline for data processing and model training.
- **Model Evaluation**: Automatically trains and evaluates multiple regression models (e.g., Random Forest, Gradient Boosting, XGBoost) to find the best performer.
- **Hyperparameter Tuning**: Uses `GridSearchCV` to find the optimal hyperparameters for the best model.
- **Web Interface**: A user-friendly web application built with Flask to get predictions in real-time.
- **Modular Codebase**: The code is organized into a clean, modular structure for easy maintenance and scalability.
- **Deployment Ready**: Includes configuration files for seamless deployment to AWS Elastic Beanstalk.

---

## ⚙️ Project Structure

The project follows a standard structure for scalable machine learning applications:

```
├── artifacts/            # Stores output files like the preprocessor and model
├── src/
│   ├── components/       # Core ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/         # Prediction pipeline logic
│   │   └── predict_pipeline.py
│   ├── init.py
│   ├── exception.py      # Custom exception handling
│   ├── logger.py         # Logging setup
│   └── utils.py          # Utility functions (save/load objects, evaluate models)
├── templates/
│   └── index.html        # Frontend HTML file
├── .ebextensions/
│   └── python.config     # AWS Elastic Beanstalk configuration
├── app.py                # Main Flask application script
├── requirements.txt      # Project dependencies
└── setup.py              # Script for packaging the project
```

---

## 🛠️ Installation and Setup

Follow these steps to set up the project on your local machine.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a Virtual Environment**
    It is highly recommended to use a virtual environment to manage project dependencies.
    
    **For Conda users:**
    ```bash
    conda create -n student_performance python=3.9 -y
    conda activate student_performance
    ```
    
    **For venv users:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    The `setup.py` file will install the project as a local package and automatically handle the installation of all required libraries from `requirements.txt`.
    ```bash
    pip install .
    ```

---

## 🚀 Usage

1.  **Running the Full Training Pipeline**
    To run the entire machine learning pipeline from data ingestion to model training, execute the `model_trainer.py` script:
    ```bash
    python src/components/model_trainer.py
    ```
    This will:
    - Ingest the raw data.
    - Perform data transformation and preprocessing.
    - Train and evaluate multiple models.
    - Save the best preprocessor object and trained model to the `artifacts` directory.

2.  **Running the Web Application**
    To start the Flask web server, run the `app.py` script:
    ```bash
    python app.py
    ```
    Now, open your web browser and navigate to `http://127.0.0.1:5000`. You can use the form to input student data and get a prediction for their math score.

---

## ☁️ Deployment to AWS Elastic Beanstalk

This project is configured for easy deployment to AWS Elastic Beanstalk.

1.  **Create an Elastic Beanstalk Environment**: Go to the AWS Management Console and create a new Elastic Beanstalk application with a "Python" platform.
2.  **Zip Your Code**: Create a `.zip` archive of your project, making sure to include the `.ebextensions` folder.
3.  **Upload and Deploy**: Upload the zip file to your Elastic Beanstalk environment. The `.ebextensions/python.config` file will automatically configure the WSGI path, and Elastic Beanstalk will install the dependencies from `requirements.txt` and run your application.

---

## 💻 Technologies Used

- **Backend**: Flask
- **Frontend**: HTML, Tailwind CSS
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **Advanced Models**: XGBoost, CatBoost
- **Deployment**: AWS Elastic Beanstalk
- **Packaging**: `setuptools`
