
import pandas as pd
import numpy as np
import os
import sys

# Mock settings
BASE_DIR = os.getcwd()

def cleanText(doc):
    # Mock cleanText for now, just split and join
    return " ".join(doc.split())

def test_csv_loading():
    print("Testing CSV loading...")
    
    # simulate file upload by just reading the existing file
    myfile_path = os.path.join(BASE_DIR, 'test_news.csv')
    if not os.path.exists(myfile_path):
        print(f"File not found: {myfile_path}")
        return

    # In views.py:
    # myfile = request.FILES['t2'].read()
    # test_csv_path = os.path.join(settings.BASE_DIR, 'FakeApp', 'static', 'test.csv')
    # ... write ...
    # data = pd.read_csv(test_csv_path, encoding="ISO-8859-1")
    
    try:
        data = pd.read_csv(myfile_path, encoding="ISO-8859-1")
        print(f"Successfully read CSV with shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        
        news = data.values
        # data = data.values # redundant in view
        
        print(f"News array shape: {news.shape}")
        
        temp = []
        for i in range(len(news)):
            try:
                # views.py line 591: value = data[i,0] (data became data.values)
                # re-simulating exact variable names from view
                current_data_values = news # data.values in view
                value = current_data_values[i,0]
                
                print(f"Row {i} raw value: {str(value)[:50]}...")
                
                value = str(value).strip().lower()
                # value = cleanText(value) # Skip NLP for now
                temp.append(value)
            except Exception as e:
                print(f"Error processing row {i}: {e}")
                
        print(f"Processed {len(temp)} items.")
        
    except Exception as e:
        print(f"Error reading CSV: {e}")

if __name__ == "__main__":
    test_csv_loading()
