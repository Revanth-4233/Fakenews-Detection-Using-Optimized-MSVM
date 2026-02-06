"""
Multi-class Fake News Detection Model Training Script
======================================================
Uses LIAR dataset with 5 classes (merged from PolitiFact's 6 labels):
- pants-fire (0)
- false (1) - includes barely-true
- half-true (2)
- mostly-true (3)
- true (4)

Author: Auto-generated for Fake News Detection project
"""

import os
import sys
import urllib.request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# NLTK Setup
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Setup NLTK data path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(SCRIPT_DIR, 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir)

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Configuration
DATASET_DIR = os.path.join(SCRIPT_DIR, 'Dataset')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')

# LIAR dataset URLs (from UCSB NLP Lab)
LIAR_TRAIN_URL = "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/train2.tsv"
LIAR_TEST_URL = "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/test2.tsv"
LIAR_VAL_URL = "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/val2.tsv"

# Class labels (5 classes - merging barely-true into false)
CLASS_LABELS_ORIGINAL = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
CLASS_LABELS = ['pants-fire', 'false', 'half-true', 'mostly-true', 'true']  # 5 classes
CLASS_NAMES = ['Pants on Fire', 'False', 'Half True', 'Mostly True', 'True']

# Mapping from 6 labels to 5 labels (barely-true -> false)
LABEL_MAP = {
    'pants-fire': 'pants-fire',
    'false': 'false',
    'barely-true': 'false',  # Merged into false
    'half-true': 'half-true',
    'mostly-true': 'mostly-true',
    'true': 'true'
}


def download_liar_dataset():
    """Download LIAR dataset from GitHub"""
    print("\n" + "="*60)
    print("🌐 DOWNLOADING LIAR DATASET")
    print("="*60)
    
    train_path = os.path.join(DATASET_DIR, 'liar_train.tsv')
    test_path = os.path.join(DATASET_DIR, 'liar_test.tsv')
    val_path = os.path.join(DATASET_DIR, 'liar_val.tsv')
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Download files
    for url, path, name in [
        (LIAR_TRAIN_URL, train_path, "Training"),
        (LIAR_TEST_URL, test_path, "Test"),
        (LIAR_VAL_URL, val_path, "Validation")
    ]:
        if not os.path.exists(path):
            print(f"  📥 Downloading {name} set...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"     ✓ Saved to {os.path.basename(path)}")
            except Exception as e:
                print(f"     ✗ Failed: {e}")
                return None
        else:
            print(f"  ✓ {name} set already exists")
    
    return train_path, test_path, val_path


def load_liar_data(file_path):
    """Load and parse LIAR TSV file"""
    # LIAR-PLUS TSV columns
    columns = [
        'id', 'label', 'statement', 'subject', 'speaker',
        'speaker_job', 'state', 'party', 'barely_true_count',
        'false_count', 'half_true_count', 'mostly_true_count',
        'pants_fire_count', 'context', 'justification'
    ]
    
    try:
        # Try new pandas API first
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=columns, 
                             on_bad_lines='skip', encoding='utf-8')
        except TypeError:
            # Fall back to old pandas API
            df = pd.read_csv(file_path, sep='\t', header=None, names=columns, 
                             error_bad_lines=False, warn_bad_lines=False, encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None



def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    tokens = text.split()
    
    # Remove punctuation
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    
    # Keep only alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords
    tokens = [w for w in tokens if w not in stop_words]
    
    # Keep tokens with length > 1
    tokens = [word for word in tokens if len(word) > 1]
    
    # Apply stemming
    tokens = [ps.stem(token) for token in tokens]
    
    # Apply lemmatization
    try:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except:
        pass
    
    return ' '.join(tokens)


def train_multiclass_model():
    """Main training function for 5-class classification"""
    
    print("\n" + "="*60)
    print("MULTI-CLASS FAKE NEWS DETECTION MODEL TRAINING")
    print("="*60)
    print(f"\nUsing 5 classes:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}. {name}")
    
    # Step 1: Download dataset
    paths = download_liar_dataset()
    if paths is None:
        print("\n❌ Failed to download dataset. Exiting.")
        return False
    
    train_path, test_path, val_path = paths
    
    # Step 2: Load data
    print("\n" + "="*60)
    print("📂 LOADING DATASET")
    print("="*60)
    
    train_df = load_liar_data(train_path)
    test_df = load_liar_data(test_path)
    val_df = load_liar_data(val_path)
    
    if train_df is None or test_df is None or val_df is None:
        print("\n❌ Failed to load data files. Exiting.")
        return False
    
    # Combine all data
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    
    print(f"\n  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Combined training: {len(full_df)}")
    
    # Step 3: Preprocess
    print("\n" + "="*60)
    print("PREPROCESSING TEXT")
    print("="*60)
    
    # Filter valid labels and map to 5 classes
    full_df = full_df[full_df['label'].isin(CLASS_LABELS_ORIGINAL)]
    test_df = test_df[test_df['label'].isin(CLASS_LABELS_ORIGINAL)]
    
    # Map labels: merge barely-true into false
    full_df['label'] = full_df['label'].map(LABEL_MAP)
    test_df['label'] = test_df['label'].map(LABEL_MAP)
    
    print(f"\n  After filtering and mapping to 5 classes:")
    print(f"  Training: {len(full_df)} | Test: {len(test_df)}")
    
    # Clean text
    print("\n  🔄 Cleaning training text...")
    X_train_text = [clean_text(str(text)) for text in full_df['statement'].values]
    
    print("  🔄 Cleaning test text...")
    X_test_text = [clean_text(str(text)) for text in test_df['statement'].values]
    
    # Encode labels (0-5)
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_LABELS)
    
    y_train = label_encoder.transform(full_df['label'].values)
    y_test = label_encoder.transform(test_df['label'].values)
    
    print("\n  Label distribution in training (5 classes):")
    for label in CLASS_LABELS:
        count = (full_df['label'] == label).sum()
        print(f"    {label:15} : {count:5} samples")
    
    # Step 4: TF-IDF Vectorization (OPTIMIZED for higher accuracy)
    print("\n" + "="*60)
    print("TF-IDF VECTORIZATION (Optimized)")
    print("="*60)
    
    # Increased max_features and added trigrams for better representation
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=2, max_df=0.95)
    X_train_tfidf = tfidf.fit_transform(X_train_text).toarray()
    X_test_tfidf = tfidf.transform(X_test_text).toarray()
    
    print(f"\n  TF-IDF features: {X_train_tfidf.shape[1]}")
    
    # Step 5: Scaling
    print("\n" + "="*60)
    print("⚖️ FEATURE SCALING")
    print("="*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_tfidf)
    X_test_scaled = scaler.transform(X_test_tfidf)
    
    print(f"\n  Features scaled successfully")
    
    # Step 6: PCA Dimensionality Reduction (OPTIMIZED - keep more components)
    print("\n" + "="*60)
    print("PCA DIMENSIONALITY REDUCTION (Optimized)")
    print("="*60)
    
    # Increased components to retain more information
    n_components = min(500, X_train_scaled.shape[1], X_train_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"\n  Reduced to {n_components} components")
    print(f"  Explained variance: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    # Step 7: Firefly Feature Selection (OPTIMIZED - keep more features)
    print("\n" + "="*60)
    print("FIREFLY FEATURE SELECTION (Optimized)")
    print("="*60)
    
    # Increased selected features for better model performance
    n_selected = min(200, X_train_pca.shape[1])
    variances = np.var(X_train_pca, axis=0)
    selected_features = np.argsort(variances)[-n_selected:]
    selected_features = sorted(selected_features.tolist())
    
    X_train_final = X_train_pca[:, selected_features]
    X_test_final = X_test_pca[:, selected_features]
    
    print(f"\n  Selected {n_selected} features using variance-based selection")
    
    # Step 8: Train Multi-class SVM (OPTIMIZED hyperparameters)
    print("\n" + "="*60)
    print("TRAINING MULTI-CLASS SVM (Optimized)")
    print("="*60)
    
    svm_model = SVC(
        kernel='rbf', 
        C=100.0,  # Increased regularization for better fit
        gamma='auto',  # Changed to auto for better kernel width
        decision_function_shape='ovr',  # One-vs-Rest for multi-class
        class_weight='balanced',  # Handle imbalanced classes
        probability=True,  # Enable probability estimates
        cache_size=500  # More cache for faster training
    )
    
    print("\n  ⏳ Training in progress...")
    svm_model.fit(X_train_final, y_train)
    print("  ✓ Training completed!")
    
    # Step 9: Evaluate
    print("\n" + "="*60)
    print("📈 MODEL EVALUATION")
    print("="*60)
    
    # Training accuracy
    y_train_pred = svm_model.predict(X_train_final)
    train_acc = accuracy_score(y_train, y_train_pred) * 100
    
    # Test accuracy
    y_test_pred = svm_model.predict(X_test_final)
    test_acc = accuracy_score(y_test, y_test_pred) * 100
    
    print(f"\n  📊 Training Accuracy: {train_acc:.2f}%")
    print(f"  📊 Test Accuracy:     {test_acc:.2f}%")
    
    print("\n  📋 Classification Report (Test Set):")
    print("-" * 60)
    print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))
    
    print("\n  📊 Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Step 10: Save Models
    print("\n" + "="*60)
    print("💾 SAVING MODELS")
    print("="*60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save TF-IDF
    with open(os.path.join(MODEL_DIR, 'tfidf.pckl'), 'wb') as f:
        pickle.dump(tfidf, f)
    print("  ✓ TF-IDF vectorizer saved")
    
    # Save Scaler
    with open(os.path.join(MODEL_DIR, 'scaler.pckl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("  ✓ Scaler saved")
    
    # Save PCA
    with open(os.path.join(MODEL_DIR, 'pca.pckl'), 'wb') as f:
        pickle.dump(pca, f)
    print("  ✓ PCA saved")
    
    # Save Firefly selected features
    np.save(os.path.join(MODEL_DIR, 'firefly.npy'), selected_features)
    print("  ✓ Selected features saved")
    
    # Save SVM Model
    with open(os.path.join(MODEL_DIR, 'svm.pckl'), 'wb') as f:
        pickle.dump(svm_model, f)
    print("  ✓ SVM model saved")
    
    # Save Label Encoder
    with open(os.path.join(MODEL_DIR, 'label_encoder.pckl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    print("  ✓ Label encoder saved")
    
    # Save class names for reference
    np.save(os.path.join(MODEL_DIR, 'class_names.npy'), CLASS_NAMES)
    print("  ✓ Class names saved")
    
    # Save processed data for Django app
    np.save(os.path.join(MODEL_DIR, 'X.npy'), X_train_text[:500] if len(X_train_text) > 500 else X_train_text)
    np.save(os.path.join(MODEL_DIR, 'Y.npy'), y_train[:500] if len(y_train) > 500 else y_train)
    print("  ✓ Sample data saved")
    
    # Save train/test split for RunML
    np.save(os.path.join(MODEL_DIR, 'data.npy'), 
            [X_train_final[:400], X_test_final[:100], y_train[:400], y_test[:100]])
    print("  ✓ Train/test split saved")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n🎯 Final Test Accuracy: {test_acc:.2f}%")
    print(f"📁 Models saved to: {MODEL_DIR}")
    print("\n💡 You can now run the Django app and test predictions!")
    
    return True


if __name__ == "__main__":
    success = train_multiclass_model()
    sys.exit(0 if success else 1)
