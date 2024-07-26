### Description:

Welcome to the Sentiment Analysis of Movie Reviews project! This repository contains the code and resources for building a sentiment analysis system using a pre-trained DistilBERT model from Hugging Face. The aim of this project is to classify movie reviews as positive, negative, or neutral.

#### Key Features:
- Utilizes the `distilbert-base-uncased-finetuned-sst-2-english` model, a lightweight version of BERT fine-tuned on the Stanford Sentiment Treebank v2 (SST-2) dataset.
- Processes the IMDB Movie Reviews dataset for training and evaluation.
- Employs the Hugging Face `datasets` and `transformers` libraries for efficient data handling and model implementation.
- Provides a Jupyter notebook documenting the data preprocessing, model training, evaluation, and prediction steps.
- Optionally includes steps for deploying the model as a web application using Flask or FastAPI.

### Contents:

1. **Data Loading and Preprocessing:**
   - Load the IMDB dataset.
   - Tokenize movie reviews using DistilBERT tokenizer.
   - Prepare data for training and evaluation.

2. **Model Loading and Fine-tuning:**
   - Load the pre-trained DistilBERT model.
   - Fine-tune the model on the IMDB dataset.

3. **Evaluation:**
   - Evaluate the model's performance on the test dataset.
   - Generate evaluation metrics.

4. **Prediction:**
   - Predict the sentiment of new movie reviews using the trained model.

5. **Deployment (Optional):**
   - Deploy the sentiment analysis model as a web application.

### How to Use:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-movie-reviews.git
   cd sentiment-analysis-movie-reviews
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
   Open the Jupyter notebook `sentiment_analysis.ipynb` and run the cells to see the full workflow, including data loading, preprocessing, model training, and evaluation.

4. **Predict Sentiment:**
   Use the provided `predict_sentiment` function to classify new movie reviews.

5. **Deploy the Model (Optional):**
   Follow the steps in the `deployment` folder to deploy the model as a web application using Flask or FastAPI.

### Dependencies:

- Python 3.x
- `transformers`
- `datasets`
- `torch`
- `flask` (for optional deployment)
- `fastapi` (for optional deployment)

### Contributing:

Contributions are welcome! Please feel free to submit a Pull Request.
