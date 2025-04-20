# 📧 Email Spam Detection

A machine learning project to classify emails as **spam** or **ham (not spam)** using NLP techniques and classification algorithms. The model leverages natural language processing to clean and vectorize the text, then applies a Naive Bayes classifier to make predictions.

## 🚀 Project Overview

Email spam detection is a common application of machine learning and NLP. This project demonstrates how to preprocess raw email text, extract meaningful features, and train a classifier to accurately distinguish between spam and ham emails.

## 🧰 Features

- Text preprocessing (cleaning, tokenization, stemming)
- Feature extraction using **TF-IDF**
- Spam classification using **Multinomial Naive Bayes**
- Evaluation metrics: Accuracy, Confusion Matrix, Precision, Recall
- Simple and modular code structure

## 🧪 Example

```bash
Input: "You have won a $1000 gift card! Click now!"
Output: SPAM
```

```bash
Input: "Hi, are we still meeting at 3 PM today?"
Output: HAM
```

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- Numpy
- NLTK
- Matplotlib / Seaborn (for visualization)

## 📁 Folder Structure

```
├── spam_detection.ipynb     # Main Jupyter notebook with all code
├── data/                    # Folder containing datasets
└── README.md                # Project documentation
```

## 📊 Dataset

The dataset used is the **SMS Spam Collection Dataset**. It's a public set of SMS messages tagged as spam or legitimate.  
🔗 [SMS Spam Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

## 📈 Results

The Naive Bayes model achieved:

- **Accuracy**: ~98%
- **Precision**: High precision for spam detection
- **Recall**: Balanced recall to reduce false negatives

## 🔧 Installation

Clone the repo and install the required packages:

```bash
git clone https://github.com/AmirAmemi/Email-spam-Detection.git
cd Email-spam-Detection
pip install -r requirements.txt
```

> You may need to download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## ▶️ Usage

Run the notebook:

```bash
jupyter notebook spam_detection.ipynb
```

## 📌 To-Do

- [ ] Add support for other classifiers (e.g., SVM, Random Forest)
- [ ] Deploy as a web app using Streamlit or Flask
- [ ] Add unit tests and modular scripts

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**AmirAmemi**  
🔗 [GitHub Profile](https://github.com/AmirAmemi)

Feel free to fork the repo, open issues, or contribute!
