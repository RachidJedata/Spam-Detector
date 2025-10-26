# 📧 EMAIL SPAM CLASSIFIER PROJECT OUTLINE

## 1. CORE PROBLEM & OBJECTIVE
### The initial MNB model showed high Precision (1.0) but poor Recall (missed 67% of spam).
### Objective: Build a high-Recall classifier by combining sparse lexical features (TF-IDF)
### with manually engineered structural features.

```
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
```

## 3. FEATURE ENGINEERING & VECTORIZATION

#### Initialization (Must be FIT on Training Data only!)
#### tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
#### scaler = MinMaxScaler()

#### A. Manual (Structural) Features
#### These features must be calculated on the raw message ('message')

```
df['tokens_count'] = df['message'].apply(lambda x: len(nltk.word_tokenize(x)))
df['char_count'] = df['message'].apply(len)
```

### B. Vectorization and Combination (Training Step)

```
X_text = tfidf_vectorizer.fit_transform(df['clean_message'])
manual_features = df[['char_count', 'tokens_count']]
manual_features_sparse = csr_matrix(manual_features.values)

X_combined = hstack([X_text, manual_features_sparse])

```
#### Total columns should match (e.g., 5000 + 5 = 5005)

``` X_final_scaled = scaler.fit_transform(X_combined.toarray())```


## 4. PREDICTION PIPELINE (For Deployment)

```
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

## Example Usage:
## X_predict = preprocess_for_predict(new_email, tfidf_vectorizer, scaler)
## prediction = model.predict(X_predict)
```
