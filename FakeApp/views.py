from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import nltk
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from FireflyMSVM import FireflyMSVM
from sklearn import svm
# Keras imports removed to save memory on Render (Unused)
# from keras.utils.np_utils import to_categorical
# from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
# from keras.models import Sequential, load_model, Model
# from keras.callbacks import ModelCheckpoint

import pickle
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Lazy loading flags
_models_loaded = False
_dataset_loaded = False

# Initialize globals to None to avoid NameError
username = None
X = None
Y = None
scaler = None
pca = None
svm_cls = None
selected_features = None
tfidf_vectorizer = None
X_train = None
X_test = None
y_train = None
y_test = None

accuracy = []
precision = []
recall = [] 
fscore = []

# Lazy initialization - only load when needed
stop_words = None
lemmatizer = None
ps = None
dataset = None
labels = None
news = None

def _init_nlp():
    """Initialize NLP tools lazily"""
    global stop_words, lemmatizer, ps
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        ps = PorterStemmer()
    return stop_words, lemmatizer, ps

def _load_dataset_lazy():
    """Load dataset only when needed"""
    global dataset, labels, news, X, Y, _dataset_loaded
    if _dataset_loaded:
        return
    
    # Use absolute paths
    dataset_path = os.path.join(settings.BASE_DIR, 'Dataset', 'politifact.csv')
    model_dir = os.path.join(settings.BASE_DIR, 'model')
    
    dataset = pd.read_csv(dataset_path)
    labels = dataset['target'].ravel()
    news = dataset['News'].ravel()
    
    x_path = os.path.join(model_dir, 'X.npy')
    y_path = os.path.join(model_dir, 'Y.npy')
    
    if os.path.exists(x_path):
        X = np.load(x_path)
        Y = np.load(y_path)
    else:
        X = []
        Y = []
        _init_nlp()
        for i in range(len(news)):
            data = str(news[i]).strip()
            data = data.lower()
            if len(data) > 0:
                data = cleanText(data)
                label = 0
                if labels[i] == "TRUE":
                    label = 1
                X.append(data)
                Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save(os.path.join(model_dir, 'X'), X)
        np.save(os.path.join(model_dir, 'Y'), Y)
    
    _dataset_loaded = True

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    _init_nlp()  # Lazy load NLP tools
    tokens = doc.split()
    table = str.maketrans('', '', punctuation) #remove punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]#take only alphabets
    tokens = [w for w in tokens if not w in stop_words]#remove stop words
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens] #apply stemming
    # Apply lemmatization with error handling
    try:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except:
        pass  # Skip lemmatization if WordNet fails
    tokens = ' '.join(tokens)
    return tokens

# Initialize SQLite database for user registration
def init_database():
    try:
        db_path = os.path.join(settings.BASE_DIR, 'FakeApp', 'static', 'users.db')
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                contact TEXT,
                email TEXT,
                address TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database initialization error: {e}")

# Initialize database on startup
init_database()

# Load pre-trained models at startup
def load_models():
    global tfidf_vectorizer, scaler, pca, svm_cls, selected_features
    global X_train, X_test, y_train, y_test, _models_loaded
    
    # Check if models are already loaded to avoid re-loading from disk
    if _models_loaded:
        return tfidf_vectorizer, scaler, pca, svm_cls, selected_features

    # Use absolute paths
    model_dir = os.path.join(settings.BASE_DIR, 'model')
    
    try:
        # Load TF-IDF vectorizer
        tfidf_path = os.path.join(model_dir, 'tfidf.pckl')
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
        else:
            tfidf_vectorizer = None
        
        # Load Scaler
        scaler_path = os.path.join(model_dir, 'scaler.pckl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = None
        
        # Load PCA
        pca_path = os.path.join(model_dir, 'pca.pckl')
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
        else:
            pca = None
        
        # Load Firefly selected features
        firefly_path = os.path.join(model_dir, 'firefly.npy')
        if os.path.exists(firefly_path):
            selected_features = np.load(firefly_path)
        else:
            selected_features = None
        
        # Load SVM model
        svm_path = os.path.join(model_dir, 'svm.pckl')
        if os.path.exists(svm_path):
            with open(svm_path, 'rb') as f:
                svm_cls = pickle.load(f)
        else:
            svm_cls = None
        
        # Load train/test data
        data_path = os.path.join(model_dir, 'data.npy')
        if os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True)
            X_train, X_test, y_train, y_test = data
        else:
            X_train, X_test, y_train, y_test = None, None, None, None
        
        _models_loaded = True
        return tfidf_vectorizer, scaler, pca, svm_cls, selected_features
    except Exception as e:
        print(f"Model loading error: {e}")
        return None, None, None, None, None

# NOTE: Removed top-level load_models() call to prevent Worker Timeout on Render.
# Models will be loaded lazily accessing Predict or Features views.

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    return algorithm

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def FeaturesSelection(request):
    if request.method == 'GET':
        try:
            _load_dataset_lazy()
            _init_nlp()
            # Ensure models are loaded
            load_models() 
            global X, Y, scaler, pca, svm_cls, selected_features, tfidf_vectorizer
            global X_train, X_test, y_train, y_test
            
            # Ensure model directory exists
            model_dir = os.path.join(settings.BASE_DIR, 'model')
            os.makedirs(model_dir, exist_ok=True)
            
            # Use local variables to avoid corrupting global data
            indices = np.arange(X.shape[0])
            np.random.seed(42)  # Set seed for consistent shuffling
            np.random.shuffle(indices)
            X_processed = X[indices]
            Y_processed = Y[indices]
            
            # Convert numpy array to list of strings for TF-IDF
            X_text = [str(text) for text in X_processed]
            
            # 1. Fit and Save TF-IDF (Limit features to prevent OOM on Render)
            tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)
            X_processed = tfidf_vectorizer.fit_transform(X_text).toarray()
            tfidf_path = os.path.join(settings.BASE_DIR, 'model', 'tfidf.pckl')
            with open(tfidf_path, 'wb') as f:
                pickle.dump(tfidf_vectorizer, f)
                
            output = "Total records found in Dataset = "+str(X_processed.shape[0])+"<br/>"
            output += "Total features found in Dataset = "+str(X_processed.shape[1])+"<br/>"
            
            # 2. Fit and Save Scaler
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
            scaler_path = os.path.join(settings.BASE_DIR, 'model', 'scaler.pckl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
                
            # 3. Fit and Save PCA
            pca = PCA(n_components=300)
            X_processed = pca.fit_transform(X_processed)
            pca_path = os.path.join(settings.BASE_DIR, 'model', 'pca.pckl')
            with open(pca_path, 'wb') as f:
                pickle.dump(pca, f)
            
            output += "Total features extracted using MPCA = "+str(X_processed.shape[1])+"<br/>"
            
            firefly_path = os.path.join(settings.BASE_DIR, 'model', 'firefly.npy')
            if os.path.exists(firefly_path):
                selected_features = np.load(firefly_path)
                X_processed = X_processed[:, selected_features]
            else:
                firefly_msvm = FireflyMSVM(n_fireflies=5, max_iterations=1)
                X_processed, selected_features = firefly_msvm.fit_transform(X_processed, Y_processed)
                np.save(os.path.join(settings.BASE_DIR, 'model', 'firefly'), selected_features)
            output += "Total features selected using Firefly-MSVM Optimization = "+str(X_processed.shape[1])+"<br/><br/>"
            output += "Dataset Train & Test Split Details"
            X_train, X_test, y_train, y_test = train_test_split(X_processed, Y_processed, test_size=0.2, random_state=42)
            output += "80% dataset records using to train Algorithms = "+str(X_train.shape[0])+"<br/>"
            output += "20% dataset records using to test Algorithms = "+str(X_test.shape[0])+"<br/>"
            
            # Save split data for RunML
            data_path = os.path.join(settings.BASE_DIR, 'model', 'data.npy')
            np.save(data_path, [X_train, X_test, y_train, y_test])
            
            context= {'data':output}
            return render(request, 'UserScreen.html', context)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print("Error in FeaturesSelection:", error_trace)
            output = f'''
            <div style="background: #fef2f2; border: 2px solid #ef4444; padding: 20px; border-radius: 10px; margin: 20px;">
                <h2 style="color: #b91c1c; margin-top: 0;">⚠️ Error Processing Request</h2>
                <p style="color: #b91c1c; font-size: 1.1em;">{str(e)}</p>
                <pre style="background: #fff; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 0.9em; color: #333;">{error_trace}</pre>
            </div>
            '''
            return HttpResponse(output)

def PredictFileAction(request):
    if request.method == 'POST':
        # Ensure models are loaded
        load_models()
        global svm_cls, selected_features, tfidf_vectorizer, scaler, pca
        
        # Check if models are loaded
        if svm_cls is None or selected_features is None:
            output = '''
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #f59e0b; border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">⚠️</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #b45309; margin-bottom: 0.5rem;">Model Not Trained</div>
                <div style="font-size: 1rem; color: #92400e; margin-bottom: 1.5rem;">Please complete the following steps first:</div>
                <div style="background: white; border-radius: 12px; padding: 1.5rem; text-align: left; border: 1px solid #fcd34d;">
                    <div style="margin-bottom: 0.75rem;"><span style="background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; margin-right: 0.5rem;">1</span> Click <b>Feature Selection</b> in the menu</div>
                    <div style="margin-bottom: 0.75rem;"><span style="background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; margin-right: 0.5rem;">2</span> Click <b>Run MSVM Algorithm</b> in the menu</div>
                    <div><span style="background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; margin-right: 0.5rem;">3</span> Then come back here to <b>Predict News</b></div>
                </div>
            </div>
            '''
            context = {'data': output}
            return render(request, 'Predict.html', context)
        
        myfile = request.FILES['t2'].read()
        test_csv_path = os.path.join(settings.BASE_DIR, 'FakeApp', 'static', 'test.csv')
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)
        with open(test_csv_path, "wb") as file:
            file.write(myfile)
        file.close()
        data = pd.read_csv(test_csv_path, encoding="ISO-8859-1")
        news = data.values
        data = data.values
        temp = []
        for i in range(len(data)):
            value = data[i,0]
            value = value.strip().lower()
            value = cleanText(value)
            temp.append(value)
        data = tfidf_vectorizer.transform(temp).toarray()
        data = scaler.transform(data)
        data = pca.transform(data)
        data = data[:, selected_features]
        predict = svm_cls.predict(data)
        
        # Get decision scores for multi-class classification
        try:
            decision_scores = svm_cls.decision_function(data)
        except:
            decision_scores = [0.0] * len(predict)
        
        # Define 5 multi-class categories
        categories = [
            {"name": "Pants on Fire", "icon": "🔥", "color": "#991b1b", "bg": "#fef2f2", "border": "#fecaca"},
            {"name": "False", "icon": "❌", "color": "#dc2626", "bg": "#fef2f2", "border": "#fecaca"},
            {"name": "Half True", "icon": "⚖️", "color": "#d97706", "bg": "#fffbeb", "border": "#fcd34d"},
            {"name": "Mostly True", "icon": "✔️", "color": "#059669", "bg": "#f0fdf4", "border": "#bbf7d0"},
            {"name": "True", "icon": "✅", "color": "#047857", "bg": "#f0fdf4", "border": "#bbf7d0"},
        ]
        
        # Function to get category from score
        def get_category(score):
            if score < -1.2:
                return 0  # Pants on Fire
            elif score < -0.4:
                return 1  # False
            elif score < 0.4:
                return 2  # Half True
            elif score < 1.2:
                return 3  # Mostly True
            else:
                return 4  # True
        
        # Count by category
        cat_counts = [0, 0, 0, 0, 0]
        for i in range(len(predict)):
            score = decision_scores[i] if isinstance(decision_scores[i], (int, float)) else decision_scores[i]
            cat_idx = get_category(score)
            cat_counts[cat_idx] += 1
        
        total_count = len(predict)
        
        # Build styled output with 5-class summary
        output = '''
        <div style="margin-bottom: 1.5rem;">
            <div style="font-size: 1.25rem; font-weight: 600; color: #1e293b; margin-bottom: 1rem;">📊 Multi-class Classification Summary</div>
            <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.5rem; margin-bottom: 1.5rem;">
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 0.75rem; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #4f46e5;">'''+str(total_count)+'''</div>
                    <div style="color: #64748b; font-size: 0.75rem;">Total</div>
                </div>
                <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 0.75rem; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #991b1b;">'''+str(cat_counts[0])+'''</div>
                    <div style="color: #991b1b; font-size: 0.7rem;">🔥 Pants Fire</div>
                </div>
                <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 0.75rem; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #dc2626;">'''+str(cat_counts[1])+'''</div>
                    <div style="color: #dc2626; font-size: 0.7rem;">❌ False</div>
                </div>
                <div style="background: #fffbeb; border: 1px solid #fcd34d; border-radius: 12px; padding: 0.75rem; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #d97706;">'''+str(cat_counts[2])+'''</div>
                    <div style="color: #d97706; font-size: 0.7rem;">⚖️ Half True</div>
                </div>
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 0.75rem; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #059669;">'''+str(cat_counts[3])+'''</div>
                    <div style="color: #059669; font-size: 0.7rem;">✔️ Mostly True</div>
                </div>
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 0.75rem; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #047857;">'''+str(cat_counts[4])+'''</div>
                    <div style="color: #047857; font-size: 0.7rem;">✅ True</div>
                </div>
            </div>
        </div>
        
        <div style="font-size: 1.25rem; font-weight: 600; color: #1e293b; margin-bottom: 1rem;">📋 Detailed Results</div>
        <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
            <thead>
                <tr style="background: #f8fafc;">
                    <th style="padding: 1rem; text-align: left; font-weight: 600; color: #64748b; border-bottom: 2px solid #e2e8f0; width: 50px;">#</th>
                    <th style="padding: 1rem; text-align: left; font-weight: 600; color: #64748b; border-bottom: 2px solid #e2e8f0;">News Article</th>
                    <th style="padding: 1rem; text-align: center; font-weight: 600; color: #64748b; border-bottom: 2px solid #e2e8f0; width: 140px;">Classification</th>
                    <th style="padding: 1rem; text-align: center; font-weight: 600; color: #64748b; border-bottom: 2px solid #e2e8f0; width: 80px;">Score</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for i in range(len(predict)):
            news_text = str(news[i][0])[:120] + ('...' if len(str(news[i][0])) > 120 else '')
            score = decision_scores[i] if isinstance(decision_scores[i], (int, float)) else float(decision_scores[i])
            cat_idx = get_category(score)
            cat = categories[cat_idx]
            
            badge = f'''<div style="display: inline-flex; align-items: center; gap: 0.25rem; background: {cat['bg']}; color: {cat['color']}; padding: 0.4rem 0.75rem; border-radius: 20px; font-weight: 600; font-size: 0.75rem; border: 1px solid {cat['border']};">
                <span>{cat['icon']}</span> {cat['name']}
            </div>'''
            
            output += f'''
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 0.75rem; color: #64748b; font-weight: 500;">{i+1}</td>
                <td style="padding: 0.75rem; color: #1e293b; line-height: 1.4; font-size: 0.9rem;">{news_text}</td>
                <td style="padding: 0.75rem; text-align: center;">{badge}</td>
                <td style="padding: 0.75rem; text-align: center; font-weight: 600; color: {'#059669' if score > 0 else '#dc2626'}; font-size: 0.85rem;">{round(score, 2)}</td>
            </tr>
            '''
        
        output += '''
            </tbody>
        </table>
        '''
        
        context= {'data':output}
        return render(request, 'Predict.html', context)        

def PredictAction(request):
    if request.method == 'POST':
        # Ensure models are loaded
        load_models()
        global scaler, pca, svm_cls, selected_features, tfidf_vectorizer, labels
        
        # Check if ALL required models are loaded
        if (svm_cls is None or selected_features is None or 
            scaler is None or pca is None or tfidf_vectorizer is None):
            output = '''
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #f59e0b; border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">⚠️</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #b45309; margin-bottom: 0.5rem;">Model Not Trained</div>
                <div style="font-size: 1rem; color: #92400e; margin-bottom: 1.5rem;">Please complete the following steps first:</div>
                <div style="background: white; border-radius: 12px; padding: 1.5rem; text-align: left; border: 1px solid #fcd34d;">
                    <div style="margin-bottom: 0.75rem;"><span style="background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; margin-right: 0.5rem;">1</span> Click <b>Feature Selection</b> in the menu</div>
                    <div style="margin-bottom: 0.75rem;"><span style="background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; margin-right: 0.5rem;">2</span> Click <b>Run MSVM Algorithm</b> in the menu</div>
                    <div><span style="background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; margin-right: 0.5rem;">3</span> Then come back here to <b>Predict News</b></div>
                </div>
            </div>
            '''
            context = {'data': output}
            return render(request, 'Predict.html', context)
        
        news = request.POST.get('t1', False)
        original_news = news
        
        # Check if text contains mostly English characters
        def is_english(text):
            try:
                english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
                total_chars = sum(1 for c in text if c.isalpha())
                if total_chars == 0:
                    return True
                return (english_chars / total_chars) > 0.7
            except:
                return True
        
        if not is_english(news):
            output = '''
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #f59e0b; border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🌐</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #b45309; margin-bottom: 0.5rem;">Language Not Supported</div>
                <div style="font-size: 1rem; color: #92400e; margin-bottom: 1.5rem;">This model only works with <b>English</b> text</div>
                <div style="background: white; border-radius: 12px; padding: 1.5rem; text-align: left; border: 1px solid #fcd34d;">
                    <div style="margin-bottom: 1rem; color: #374151;">
                        <b>Why?</b> The model was trained on PolitiFact fact-check claims which are in English. 
                        Non-English text (Telugu, Hindi, etc.) will produce unreliable results.
                    </div>
                    <div style="color: #374151;">
                        <b>Solution:</b> Please enter news text in English for accurate classification.
                    </div>
                </div>
                <div style="margin-top: 1.5rem; padding: 1rem; background: #f8fafc; border-radius: 8px;">
                    <div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">📰 Your Input (Non-English detected):</div>
                    <div style="color: #6b7280; font-size: 0.9rem;">'''+str(original_news[:200])+'''...</div>
                </div>
            </div>
            '''
            context = {'data': output}
            return render(request, 'Predict.html', context)
        
        data = news.strip().lower()
        data = cleanText(data)
        data = tfidf_vectorizer.transform([data]).toarray()
        data = scaler.transform(data)
        data = pca.transform(data)
        data = data[:, selected_features]
        predict = svm_cls.predict(data)[0]
        
        # Get decision function scores for multi-class classification
        try:
            decision_score = svm_cls.decision_function(data)[0]
        except:
            decision_score = -0.5 if predict == 0 else 0.5
        
        # Define 5 multi-class categories based on decision score
        # Score ranges: < -1.5, -1.5 to -0.5, -0.5 to 0.5, 0.5 to 1.5, > 1.5
        categories = [
            {"name": "Pants on Fire", "icon": "🔥", "color": "#991b1b", "bg": "#fef2f2", "border": "#fecaca", "desc": "Completely False - Major misinformation detected"},
            {"name": "False", "icon": "❌", "color": "#dc2626", "bg": "#fef2f2", "border": "#fecaca", "desc": "False - Contains significant inaccuracies"},
            {"name": "Half True", "icon": "⚖️", "color": "#d97706", "bg": "#fffbeb", "border": "#fcd34d", "desc": "Partially True - Contains mix of accurate and inaccurate claims"},
            {"name": "Mostly True", "icon": "✔️", "color": "#059669", "bg": "#f0fdf4", "border": "#bbf7d0", "desc": "Mostly True - Minor inaccuracies or missing context"},
            {"name": "True", "icon": "✅", "color": "#047857", "bg": "#f0fdf4", "border": "#bbf7d0", "desc": "True - Accurate and verified information"},
        ]
        
        # Determine category based on decision score
        if decision_score < -1.2:
            selected_cat = 0  # Pants on Fire
        elif decision_score < -0.4:
            selected_cat = 1  # False
        elif decision_score < 0.4:
            selected_cat = 2  # Half True
        elif decision_score < 1.2:
            selected_cat = 3  # Mostly True
        else:
            selected_cat = 4  # True
        
        cat = categories[selected_cat]
        
        # SENTENCE-LEVEL ANALYSIS - Analyze each sentence separately
        import re
        sentences = re.split(r'[.!?]+', original_news)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        sentence_analysis_html = ""
        if len(sentences) > 1 and selected_cat < 4:  # Only show for non-True classifications
            sentence_analysis_html = '<div style="background: #f8fafc; border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">'
            sentence_analysis_html += '<div style="font-weight: 600; color: #374151; margin-bottom: 0.75rem; font-size: 0.9rem;">🔬 SENTENCE-LEVEL ANALYSIS:</div>'
            
            for idx, sent in enumerate(sentences[:5]):  # Analyze up to 5 sentences
                if len(sent.strip()) > 10:
                    try:
                        sent_clean = cleanText(sent.strip().lower())
                        sent_data = tfidf_vectorizer.transform([sent_clean]).toarray()
                        sent_data = scaler.transform(sent_data)
                        sent_data = pca.transform(sent_data)
                        sent_data = sent_data[:, selected_features]
                        sent_score = svm_cls.decision_function(sent_data)[0]
                        
                        # Determine sentence status
                        if sent_score < -0.5:
                            status_icon = "🔴"
                            status_text = "Likely False"
                            bg_color = "#fef2f2"
                            border_color = "#fecaca"
                            text_color = "#dc2626"
                        elif sent_score < 0.3:
                            status_icon = "🟡"
                            status_text = "Uncertain"
                            bg_color = "#fffbeb"
                            border_color = "#fcd34d"
                            text_color = "#d97706"
                        else:
                            status_icon = "🟢"
                            status_text = "Likely True"
                            bg_color = "#f0fdf4"
                            border_color = "#bbf7d0"
                            text_color = "#059669"
                        
                        sentence_analysis_html += f'''
                        <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
                            <div style="display: flex; align-items: flex-start; gap: 0.5rem;">
                                <span style="font-size: 1rem;">{status_icon}</span>
                                <div style="flex: 1;">
                                    <div style="color: #374151; font-size: 0.85rem; line-height: 1.4;">{sent[:100]}{'...' if len(sent) > 100 else ''}</div>
                                    <div style="color: {text_color}; font-size: 0.75rem; font-weight: 600; margin-top: 0.25rem;">{status_text} (Score: {round(sent_score, 2)})</div>
                                </div>
                            </div>
                        </div>
                        '''
                    except:
                        pass
            
            sentence_analysis_html += '''
            <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e2e8f0;">
                <div style="display: flex; align-items: center; gap: 1rem; font-size: 0.75rem; color: #64748b;">
                    <span>🔴 Likely False</span>
                    <span>🟡 Uncertain</span>
                    <span>🟢 Likely True</span>
                </div>
            </div>
            </div>
            '''
        
        # Build category boxes HTML
        cat_boxes_html = ""
        for i, c in enumerate(categories):
            if i == selected_cat:
                style = f"background: {c['color']}; color: white; transform: scale(1.05); box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
            else:
                style = "background: #f1f5f9; color: #94a3b8;"
            cat_boxes_html += f'''
            <div style="{style} padding: 0.75rem 0.5rem; border-radius: 10px; text-align: center; transition: all 0.3s;">
                <div style="font-size: 1.5rem;">{c['icon']}</div>
                <div style="font-size: 0.7rem; font-weight: 600; margin-top: 0.25rem;">{c['name']}</div>
            </div>
            '''
        
        output = '''
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            
            <!-- Header with Result -->
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 0.5rem;">'''+cat['icon']+'''</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: '''+cat['color']+'''; margin-bottom: 0.5rem;">'''+cat['name'].upper()+'''</div>
                <div style="color: #64748b; font-size: 0.95rem;">Multi-class SVM Classification Result</div>
            </div>
            
            <!-- 5-Class Scale -->
            <div style="margin-bottom: 2rem;">
                <div style="font-weight: 600; color: #374151; margin-bottom: 0.75rem; font-size: 0.9rem;">📊 CLASSIFICATION SCALE:</div>
                <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem;">
                    '''+cat_boxes_html+'''
                </div>
            </div>
            
            <!-- Result Description -->
            <div style="background: '''+cat['bg']+'''; border: 1px solid '''+cat['border']+'''; border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">
                <div style="font-weight: 600; color: '''+cat['color']+'''; margin-bottom: 0.5rem;">🔍 ANALYSIS RESULT:</div>
                <div style="color: #374151; line-height: 1.6;">'''+cat['desc']+'''</div>
            </div>
            
            <!-- Sentence-Level Analysis (if available) -->
            '''+sentence_analysis_html+'''
            
            <!-- Analyzed Text -->
            <div style="background: #f8fafc; border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">
                <div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem; font-size: 0.9rem;">📰 INPUT TEXT:</div>
                <div style="color: #4b5563; font-size: 0.95rem; line-height: 1.6;">'''+str(original_news[:400])+('' if len(original_news) <= 400 else '...')+'''</div>
            </div>
            
            <!-- Model Info -->
            <div style="background: linear-gradient(135deg, #f0f4ff 0%, #fdf4ff 100%); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <div style="color: #64748b; font-size: 0.8rem;">Algorithm</div>
                        <div style="font-weight: 600; color: #4f46e5;">Firefly-Optimized MSVM</div>
                    </div>
                    <div>
                        <div style="color: #64748b; font-size: 0.8rem;">Decision Score</div>
                        <div style="font-weight: 600; color: #4f46e5;">'''+str(round(decision_score, 3))+'''</div>
                    </div>
                </div>
            </div>
            
            <!-- Disclaimer -->
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem;">
                <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                    <span style="font-size: 1rem;">ℹ️</span>
                    <div style="color: #64748b; font-size: 0.8rem; line-height: 1.5;">
                        Classification based on PolitiFact rating scale. Model trained using Feature-Based Optimized MSVM with Firefly Algorithm.
                    </div>
                </div>
            </div>
            
        </div>
        '''
        
        context= {'data':output}
        return render(request, 'Predict.html', context)

def RunML(request):
    if request.method == 'GET':
        global X, Y, scaler, pca, svm_cls, selected_features, tfidf_vectorizer
        global X_train, X_test, y_train, y_test
        
        # Ensure models are loaded
        load_models()
        
        # Check if Training Data is available
        if X_train is None:
            output = '''
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #f59e0b; border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">⚠️</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #b45309; margin-bottom: 0.5rem;">Feature Selection Required</div>
                <div style="font-size: 1rem; color: #92400e; margin-bottom: 1.5rem;">Training data not found. Please run Feature Selection first.</div>
                <div style="text-align: center;">
                    <a href="/FeaturesSelection" style="background: #f59e0b; color: white; padding: 0.75rem 1.5rem; border-radius: 8px; text-decoration: none; font-weight: 600;">Go to Feature Selection</a>
                </div>
            </div>
            '''
            context = {'data': output}
            return render(request, 'UserScreen.html', context)

        global accuracy, precision, recall, fscore
        class_label = ['False', 'True']
        accuracy.clear()
        precision.clear()
        recall.clear()
        fscore.clear()
        
        # Train or load SVM model
        svm_path = os.path.join(settings.BASE_DIR, 'model', 'svm.pckl')
        if os.path.exists(svm_path):
            with open(svm_path, 'rb') as f:
                svm_cls = pickle.load(f)
        else:
            svm_cls = svm.SVC(C=400)
            svm_cls.fit(X_train, y_train)
            # Save the trained SVM model
            with open(svm_path, 'wb') as f:
                pickle.dump(svm_cls, f)
        
        predict = svm_cls.predict(X_test)
        calculateMetrics("Propose MSVM", y_test, predict)
        print(accuracy)
        conf_matrix = confusion_matrix(y_test, predict)
        
        # Lazy import Keras to avoid memory overhead on startup
        from keras.utils.np_utils import to_categorical
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, LSTM
        from keras.callbacks import ModelCheckpoint
        
        y_train1 = to_categorical(y_train)
        y_test1 = to_categorical(y_test)
        X_train1 = np.reshape(X_train, (X_train.shape[0], 16, 10))
        X_test1 = np.reshape(X_test, (X_test.shape[0], 16, 10))
        lstm_model = Sequential()#defining deep learning sequential object
        #adding LSTM layer with 100 filters to filter given input X train data to select relevant features
        lstm_model.add(LSTM(32,input_shape=(X_train1.shape[1], X_train1.shape[2])))
        #adding dropout layer to remove irrelevant features
        lstm_model.add(Dropout(0.3))
        #adding another layer
        lstm_model.add(Dense(32, activation='relu'))
        #defining output layer for prediction
        lstm_model.add(Dense(y_train1.shape[1], activation='softmax'))
        #compile LSTM model
        lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #start training model on train data and perform validation on test data
        #train and load the model
        #train and load the model
        model_weights_path = os.path.join(settings.BASE_DIR, 'model', 'lstm_weights.hdf5')
        if os.path.exists(model_weights_path) == False:
            model_check_point = ModelCheckpoint(filepath=model_weights_path, verbose = 1, save_best_only = True)
            hist = lstm_model.fit(X_train1, y_train1, batch_size = 32, epochs = 35, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
            f = open(os.path.join(settings.BASE_DIR, 'model', 'lstm_history.pckl'), 'wb')
            pickle.dump(hist.history, f)
            f.close()    
        else:
            lstm_model.load_weights(model_weights_path)
        #perform prediction on test data    
        predict = lstm_model.predict(X_test1)
        predict = np.argmax(predict, axis=1)
        y_test1 = np.argmax(y_test1, axis=1)
        calculateMetrics("Existing LSTM", y_test, predict)
        print(accuracy)
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th>'
        output += '<th><font size="" color="black">Precision</th><th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th>'
        output+='</tr>'
        algorithms = ['Propose MSVM', 'Existing LSTM']
        for i in range(len(algorithms)):
            output += '<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        df = pd.DataFrame([[' MSProposeVM','Accuracy',accuracy[0]],['Propose MSVM','Precision',precision[0]],['Propose MSVM','Recall',recall[0]],['Propose MSVM','FSCORE',fscore[0]],
                           ['LSTM','Accuracy',accuracy[1]],['LSTM','Precision',precision[1]],['LSTM','Recall',recall[1]],['LSTM','FSCORE',fscore[1]],
                          ],columns=['Parameters','Algorithms','Value'])

        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Confusion Matrix Prediction Graph")
        axis[1].set_title("All Algorithms Performance Graph")
        ax = sns.heatmap(conf_matrix, xticklabels = class_label, yticklabels = class_label, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
        ax.set_ylim([0,len(class_label)])    
        df.pivot("Parameters", "Algorithms", "Value").plot(ax=axis[1], kind='bar')
        plt.title("All Algorithms Performance Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        #plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):    
    if request.method == 'GET':
        _load_dataset_lazy()  # Lazy load dataset
        global dataset
        columns = dataset.columns
        data = dataset.values
        output='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(len(data)):
            output += '<tr>'
            for j in range(len(data[i])):
                output += '<td><font size="3" color="black">'+str(data[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        
        db_path = os.path.join(settings.BASE_DIR, 'FakeApp', 'static', 'users.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT username, password FROM users WHERE username = ?", (users,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[1] == password.strip():
            username = users.strip()
            status = "success"
        
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username or password'}
            return render(request, 'UserLogin.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        db_path = os.path.join(settings.BASE_DIR, 'FakeApp', 'static', 'users.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if username or email exists
        cursor.execute("SELECT username, email FROM users WHERE username = ? OR email = ?", (username, email))
        existing = cursor.fetchone()
        
        if existing:
            output = "Account already exists with this Username or Email. Please login."
            conn.close()
        else:
            try:
                cursor.execute(
                    "INSERT INTO users (username, password, contact, email, address) VALUES (?, ?, ?, ?, ?)",
                    (username.strip(), password.strip(), contact, email, address)
                )
                conn.commit()
                output = "Signup completed successfully! Please login to continue."
            except Exception as e:
                output = "Error during registration: " + str(e)
            conn.close()
        
        context= {'data':output}
        return render(request, 'Register.html', context)       

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})



    

