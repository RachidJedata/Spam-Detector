# 📧 EMAIL SPAM CLASSIFIER PROJECT OUTLINE

## 1. CORE PROBLEM & OBJECTIVE
# The initial MNB model showed high Precision (1.0) but poor Recall (missed 67% of spam).
# Objective: Build a high-Recall classifier by combining sparse lexical features (TF-IDF)
# with manually engineered structural features.

## 2. KEY PREPROCESSING COMPONENTS

# Required NLTK Imports:
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# nltk.download(['punkt', 'stopwords']) 
# # Note: 'punkt_tab' error fix requires 'nltk.download('punkt_tab')' if needed

def clean_text(text):
    """Cleans text: lowercases, removes punctuation/numbers, tokenizes, removes stopwords, and stems."""
    # 1. Cleaning
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    # 2. Tokenization & Stopword Removal
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 3. Stemming
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)


## 3. FEATURE ENGINEERING & VECTORIZATION

# Initialization (Must be FIT on Training Data only!)
# tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
# scaler = MinMaxScaler()

# A. Manual (Structural) Features
# These features must be calculated on the raw message ('message')

# df['tokens_count'] = df['message'].apply(lambda x: len(nltk.word_tokenize(x)))
# df['char_count'] = df['message'].apply(len)
# df['caps_ratio'] = df['message'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-6))
# df['num_dollar_sign'] = df['message'].apply(lambda x: x.count('$'))
# df['has_url'] = df['message'].apply(lambda x: 1 if re.search(r'http[s]?://', x) else 0)

# B. Vectorization and Combination (Training Step)

# X_text = tfidf_vectorizer.fit_transform(df['clean_message'])

# manual_features = df[['char_count', 'tokens_count', 'caps_ratio', 'num_dollar_sign', 'has_url']]
# manual_features_sparse = csr_matrix(manual_features.values)

# X_combined = hstack([X_text, manual_features_sparse]) # Total columns should match (e.g., 5000 + 5 = 5005)

# X_final_scaled = scaler.fit_transform(X_combined.toarray())


## 4. MODELING & OPTIMIZATION

# A. Recommended Model Switch (To increase Recall)
# Use Logistic Regression or LinearSVC over Multinomial Naive Bayes.
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC

# model = LogisticRegression(solver='liblinear', penalty='l2', C=0.5) # Example
# # OR
# # model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, C=1.0) 

# B. Crucial Optimization: Threshold Adjustment
# If using Logistic Regression (or MNB with predict_proba):

# y_pred_proba = model.predict_proba(X_test)[:, 1]
# NEW_THRESHOLD = 0.3 # Lower from 0.5 to increase sensitivity (Recall)
# y_pred_adjusted = (y_pred_proba >= NEW_THRESHOLD).astype(int)

# Evaluate using the adjusted predictions (y_pred_adjusted)


## 5. PREDICTION PIPELINE (For Deployment)

def preprocess_for_predict(new_messages, tfidf_vectorizer, scaler):
    """
    Applies fitted transformations to new messages.
    Requires: fitted tfidf_vectorizer, fitted scaler.
    """
    if isinstance(new_messages, str):
        new_messages = pd.Series([new_messages])

    temp_df = pd.DataFrame({'message': new_messages})

    # 1. Calculate ALL Manual Features
    # (Ensure the feature list is identical to training)
    temp_df['tokens_count'] = temp_df['message'].apply(lambda x: len(nltk.word_tokenize(x)))
    temp_df['char_count'] = temp_df['message'].apply(len)
    temp_df['caps_ratio'] = temp_df['message'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-6))
    temp_df['num_dollar_sign'] = temp_df['message'].apply(lambda x: x.count('$'))
    temp_df['has_url'] = temp_df['message'].apply(lambda x: 1 if re.search(r'http[s]?://', x) else 0)

    # 2. Text Cleaning & Stemming
    temp_df['clean_message'] = temp_df['message'].apply(clean_text)

    # 3. Vectorization (MUST use .transform())
    X_text = tfidf_vectorizer.transform(temp_df['clean_message'])

    # 4. Combination and Scaling (MUST use .transform() on fitted objects)
    manual_features = temp_df[['char_count', 'tokens_count', 'caps_ratio', 'num_dollar_sign', 'has_url']]
    manual_features_sparse = csr_matrix(manual_features.values)
    
    X_combined = hstack([X_text, manual_features_sparse])
    
    # Scale (Ensure the input dimension matches the scaler's fitted dimension)
    X_final_scaled = scaler.transform(X_combined.toarray())
    
    return X_final_scaled

# # Example Usage:
# # X_predict = preprocess_for_predict(new_email, tfidf_vectorizer, scaler)
# # prediction = model.predict(X_predict)