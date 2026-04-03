# Predicting Article Engagement Based on Title using Machine Learning

This project uses machine learning to predict article engagement metrics (Retweets, Likes, and Claps) based on article titles. The model analyzes the linguistic features of article titles and provides predictions for engagement on Twitter and Medium platforms.

## 📊 Project Overview

Predicting content engagement is crucial for content creators and publishers. This project builds predictive models using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization combined with additional text features to estimate how well an article title might perform on social media platforms.

### Key Features
- **Multi-platform predictions**: Estimates Retweets and Likes for Twitter, and Claps for Medium
- **Text preprocessing**: Removes noise, handles stopwords, and normalizes text
- **Feature engineering**: Combines TF-IDF vectors with text-based features (length, word count)
- **Machine Learning Models**: Trained regression models for each engagement metric
- **Interactive Web Interface**: Streamlit-based application for easy predictions

## 🎯 Objectives

- Build accurate machine learning models to predict article engagement
- Provide content creators with insights into title effectiveness
- Create an easy-to-use interface for making predictions

## 📁 Project Structure

```
├── Predicting_Article_Engagement_Based_on_Title_using_Machine_Learning.ipynb  # Main analysis notebook
├── app.py                          # Streamlit application
├── retweet_model.pkl              # Trained model for retweet predictions
├── likes_model.pkl                # Trained model for likes predictions
├── claps_model.pkl                # Trained model for claps predictions
├── vectorizer.pkl                 # TF-IDF vectorizer for text transformation
├── requirements.txt               # Project dependencies
└── README.md                       # This file
```

## 🚀 Live Demo

**[View the live Streamlit app](https://predicting-article-engagement-based-on-title-machine-learning.streamlit.app/)**

Try the interactive predictor to get engagement predictions for your article titles!

## 💻 Technologies Used

- **Python**: Core programming language
- **Streamlit**: Interactive web application framework
- **Scikit-learn**: Machine learning models and preprocessing
- **NLTK**: Natural Language Toolkit for text processing
- **Joblib**: Model serialization and caching
- **Jupyter Notebook**: Data exploration and model development

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AutumLeaf/Predicting-Article-Engagement-Based-on-Title-using-Machine-Learning.git
   cd Predicting-Article-Engagement-Based-on-Title-using-Machine-Learning
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

**How to use**:
1. Enter your article title in the text input field
2. Click "Predict Performance"
3. View the predicted engagement metrics:
   - **Retweets (Twitter)**: Estimated number of retweets
   - **Likes (Twitter)**: Estimated number of likes
   - **Claps (Medium)**: Estimated number of claps

### Explore the Notebook

Open `Predicting_Article_Engagement_Based_on_Title_using_Machine_Learning.ipynb` in Jupyter Notebook to see:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training and evaluation
- Performance metrics and insights

```bash
jupyter notebook
```

## 🔍 Model Details

### Data Preprocessing
- Removal of Twitter mentions and special characters
- Text normalization (lowercase conversion)
- Stopword removal using NLTK
- TF-IDF vectorization of article titles

### Features Used
1. **TF-IDF Vectors**: Importance of words in the title
2. **Text Length**: Total character count of the title
3. **Word Count**: Number of words in the title

### Models Trained
- **Retweet Predictor**: Regression model for predicting tweet retweets
- **Likes Predictor**: Regression model for predicting tweet likes
- **Claps Predictor**: Regression model for predicting Medium claps

## 📊 Results & Performance

The trained models demonstrate reasonable accuracy in predicting engagement metrics. Evaluation metrics and detailed analysis can be found in the Jupyter notebook.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add improvement'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details (if applicable).

## 👤 Author

**AutumLeaf**

- GitHub: [@AutumLeaf](https://github.com/AutumLeaf)

## 💡 Future Enhancements

- [ ] Add more engagement metrics (shares, comments, etc.)
- [ ] Include additional features (publication time, author popularity, category)
- [ ] Support for multiple languages
- [ ] API endpoint for programmatic access
- [ ] Model retraining pipeline with new data
- [ ] Advanced visualization of predictions and insights

## 📧 Contact & Support

If you have questions or suggestions about this project, feel free to open an issue or reach out directly.

---

**Last Updated**: March 26, 2026
