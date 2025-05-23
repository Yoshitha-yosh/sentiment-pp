import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
import pickle
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("sentimentdataset.csv", encoding='latin1')
df = df[['Text', 'Sentiment', 'Platform']].dropna()

# Refined sentiment mapping function
def map_sentiment(label):
    label = label.strip().lower()
    positive = {'acceptance', 'accomplishment', 'admiration', 'adoration', 'affection', 'amazement', 'amusement', 'appreciation', 'awe', 'blessed', 'calmness', 'celebration', 'compassion', 'confidence', 'contentment', 'creativity', 'delight', 'determination', 'ecstasy', 'elation', 'empowerment', 'enjoyment', 'enthusiasm', 'euphoria', 'excitement', 'freedom', 'friendship', 'fulfillment', 'grateful', 'gratitude', 'happiness', 'hope', 'inspiration', 'joy', 'kindness', 'love', 'optimism', 'peace', 'pride', 'radiance', 'relief', 'resilience', 'romance', 'satisfaction', 'serenity', 'success', 'surprise', 'tenderness', 'triumph', 'vibrancy', 'wonder'}
    negative = {'anger', 'anxiety','bitter', 'apprehensive', 'boredom', 'confusion', 'despair', 'disappointment', 'disgust', 'embarrassed', 'fear', 'frustrated', 'grief', 'hate', 'heartache', 'helplessness', 'jealousy', 'loneliness', 'melancholy','heating','terrible','misery', 'numbness', 'regret', 'resentment', 'sad', 'sadness', 'shame', 'sorrow', 'stress', 'suffering', 'terror', 'worried'}
    neutral = {'ambivalence', 'curiosity', 'indifference', 'neutral', 'uncertainty'}
    if label in positive:
        return 'positive'
    elif label in negative:
        return 'negative'
    elif label in neutral:
        return 'neutral'
    else:
        return None  # drop unknown labels

df['Sentiment'] = df['Sentiment'].apply(map_sentiment)
df = df.dropna(subset=['Sentiment'])

print("Simplified sentiment class distribution:")
print(df['Sentiment'].value_counts())

# Encode labels
sentiment_encoder = LabelEncoder()
df['Sentiment_enc'] = sentiment_encoder.fit_transform(df['Sentiment'])

platform_encoder = LabelEncoder()
df['Platform_enc'] = platform_encoder.fit_transform(df['Platform'])

# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Text'])
sequences = tokenizer.texts_to_sequences(df['Text'])
X = pad_sequences(sequences, maxlen=100)

y_sentiment = df['Sentiment_enc'].values
y_platform = df['Platform_enc'].values

# Stratified train-test split for sentiment and platform separately
X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
    X, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
)
X_train_plt, X_test_plt, y_train_plt, y_test_plt = train_test_split(
    X, y_platform, test_size=0.2, random_state=42, stratify=y_platform
)

# Build sentiment model
input_layer = Input(shape=(100,))
x = Embedding(5000, 64)(input_layer)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(LSTM(64, dropout=0.2))(x)
x = Dropout(0.3)(x)
sentiment_output = Dense(len(sentiment_encoder.classes_), activation='softmax')(x)

sentiment_model = Model(inputs=input_layer, outputs=sentiment_output)
sentiment_model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Class weights for sentiment
sentiment_class_weights = compute_class_weight('balanced', classes=np.unique(y_train_sent), y=y_train_sent)
sentiment_class_weight_dict = dict(enumerate(sentiment_class_weights))
print("Sentiment class weights:", sentiment_class_weight_dict)

# Train sentiment model
sentiment_model.fit(
    X_train_sent, y_train_sent,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    class_weight=sentiment_class_weight_dict,
    verbose=2
)

# Evaluate sentiment model
y_pred_sent = np.argmax(sentiment_model.predict(X_test_sent), axis=1)
print("Sentiment classification report:")
print(classification_report(y_test_sent, y_pred_sent, target_names=sentiment_encoder.classes_))

# Build platform model
input_layer_plt = Input(shape=(100,))
x_plt = Embedding(5000, 64)(input_layer_plt)
x_plt = SpatialDropout1D(0.2)(x_plt)
x_plt = Bidirectional(LSTM(64, dropout=0.2))(x_plt)
x_plt = Dropout(0.3)(x_plt)
platform_output = Dense(len(platform_encoder.classes_), activation='softmax')(x_plt)

platform_model = Model(inputs=input_layer_plt, outputs=platform_output)
platform_model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Class weights for platform
platform_class_weights = compute_class_weight('balanced', classes=np.unique(y_train_plt), y=y_train_plt)
platform_class_weight_dict = dict(enumerate(platform_class_weights))
print("Platform class weights:", platform_class_weight_dict)

# Train platform model
platform_model.fit(
    X_train_plt, y_train_plt,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    class_weight=platform_class_weight_dict,
    verbose=2
)

# Evaluate platform model
y_pred_plt = np.argmax(platform_model.predict(X_test_plt), axis=1)
print("Platform classification report:")
print(classification_report(y_test_plt, y_pred_plt, target_names=platform_encoder.classes_))

# Save models, tokenizer, and encoders
sentiment_model.save("sentiment_model.h5")
platform_model.save("platform_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("sentiment_encoder.pkl", "wb") as f:
    pickle.dump(sentiment_encoder, f)
with open("platform_encoder.pkl", "wb") as f:
    pickle.dump(platform_encoder, f)

print("Training complete. Models and encoders saved.")
