import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
#from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("C:\\Users\\ayush\\OneDrive\\Desktop\\recommandatio system\\kaggle-preprocessed.csv")

df['features']= (df['Dataset_name'].astype(str) + ' ' +
                 df['Type_of_file'].astype(str) + ' ' +
                 df['Author_name'].astype(str))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

vectorizer= TfidfVectorizer()
X= vectorizer.fit_transform(df['features'])

model = NearestNeighbors(n_neighbors=5, metric='cosine')
model.fit(X)

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(df, 'dataset.pkl')