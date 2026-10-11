# 🤝 Profile Recommendation System

An intelligent, machine learning-based **Profile Recommendation System** that identifies compatible user profiles using professional information, personal interests, and demographic features.

Built with Python, Scikit-learn, Pandas, and Streamlit, the application combines Natural Language Processing (NLP) and machine learning to rank potential matches and display the top five recommendations through an interactive web interface.

## 🚀 Features

- **Profile Text Processing:** Combines users' locations, professions, professional summaries, personal descriptions, and interests.
- **TF-IDF Vectorization:** Converts textual profile information into numerical feature vectors.
- **Cosine Similarity:** Measures textual similarity between user profiles.
- **Logistic Regression:** Learns from historical feedback to estimate the probability of a positive match.
- **Compatibility Ranking:** Ranks potential matches and displays the top five recommendations.
- **Interactive Streamlit UI:** Search for a member by name and view their profile information and recommended matches.
- **Feedback-Based Learning:** Uses historical user feedback to train the recommendation classifier.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Natural Language Processing:** TF-IDF Vectorizer
- **Similarity Measurement:** Cosine Similarity
- **Web Interface:** Streamlit

## ⚙️ How It Works

1. **Load the data:** Read user profiles from `users.csv` and historical interactions from `feedback.csv`.
2. **Preprocess profile text:** Convert selected text fields to lowercase and combine them into a single text representation.
3. **Extract text features:** Apply TF-IDF to transform profile text into numerical vectors.
4. **Calculate similarity:** Use cosine similarity to compare profiles.
5. **Prepare training data:** Extract similarity, age difference, and experience difference for user pairs with available feedback.
6. **Train the model:** Train a Logistic Regression classifier using historical feedback.
7. **Generate recommendations:** Calculate match probabilities for other profiles and rank them in descending order.
8. **Display results:** Present the top five recommendations through the Streamlit application.

## 📁 Project Structure

```text
profile-recommendation-system/
│
├── app.py                 # Streamlit user interface
├── major_project_1.py     # Data processing and ML model
├── users.csv              # User profile dataset
├── feedback.csv           # Historical user feedback
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd profile-recommendation-system
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the datasets

Place `users.csv` and `feedback.csv` in the project directory. The application expects both files to be available when the ML module loads.

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 📊 Model Features

The Logistic Regression classifier uses three features:

| Feature | Description |
|---|---|
| Similarity | Cosine similarity between two profile text representations |
| Age Difference | Absolute difference between users' ages |
| Experience Difference | Absolute difference between users' years of experience |

The target label is derived from the `action` column in `feedback.csv`.

## 🖥️ Using the Application

1. Launch the Streamlit application.
2. Enter a member's name in the search field.
3. Click **Find Matches**.
4. Review the selected member's profile.
5. Explore the top five recommendations, including profession, location, MBTI type, and compatibility score.

## 📈 Evaluation

The model training script evaluates the Logistic Regression classifier using:

- Classification Report
- ROC-AUC Score

These metrics evaluate the classifier on a held-out test set. Recommendation quality may also depend on the amount and quality of historical feedback data.

## 👨‍💻 Author

Developed as a machine learning project exploring NLP, similarity-based recommendations, and supervised classification.
