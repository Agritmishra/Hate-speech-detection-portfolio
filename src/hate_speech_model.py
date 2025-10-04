import re, string, os, pickle, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

STOPWORDS = set(['a','an','the','and','or','if','in','on','at','to','for','with','of','is','it','this','that','these','those','as','by','from','be','been','are','was','were','so','but','not','no','we','you','he','she','they','i'])
def stem(w):
    for suf in ('ing','ly','ed','s'):
        if w.endswith(suf) and len(w)>len(suf)+2:
            return w[:-len(suf)]
    return w
def clean_text(text):
    text=str(text).lower()
    text=re.sub(r'http\\S+|www\\.\\S+','',text)
    text=re.sub(r'[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub(r'\\n',' ',text)
    text=re.sub(r'\\w*\\d\\w*','',text)
    tokens=[w for w in text.split() if w not in STOPWORDS]
    tokens=[stem(w) for w in tokens]
    return ' '.join(tokens)
def load_data(p): 
    import pandas as pd
    df=pd.read_csv(p)
    df=df[['tweet','class']].dropna()
    df['label']=df['class'].map({0:'Hate Speech',1:'Offensive Language',2:'No Hate and Offensive'})
    return df[['tweet','label']]
def prepare_features(df, vectorizer=None):
    texts=df['tweet'].astype(str).apply(clean_text).tolist()
    if vectorizer is None:
        vectorizer=TfidfVectorizer(max_features=2000,ngram_range=(1,2))
        X=vectorizer.fit_transform(texts)
    else:
        X=vectorizer.transform(texts)
    return X, vectorizer
def train_model(csv_path, save_dir):
    os.makedirs(save_dir,exist_ok=True)
    df=load_data(csv_path)
    X,v=prepare_features(df,None)
    y=df['label'].values
    # try stratify, fallback if dataset too small
    try:
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    except Exception:
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
    clf=LogisticRegression(max_iter=1000)
    clf.fit(X_train,y_train)
    preds=clf.predict(X_test)
    acc=accuracy_score(y_test,preds)
    rpt=classification_report(y_test,preds,digits=4)
    cm=confusion_matrix(y_test,preds)
    with open(os.path.join(save_dir,'model.pkl'),'wb') as f:
        pickle.dump(clf,f)
    with open(os.path.join(save_dir,'vectorizer.pkl'),'wb') as f:
        pickle.dump(v,f)
    metrics={'accuracy':float(acc),'report':rpt,'confusion_matrix':cm.tolist()}
    with open(os.path.join(save_dir,'metrics.json'),'w') as f:
        json.dump(metrics,f,indent=2)
    return acc, rpt, cm
def predict(texts, model_path, vectorizer_path):
    with open(model_path,'rb') as f:
        clf=pickle.load(f)
    with open(vectorizer_path,'rb') as f:
        v=pickle.load(f)
    texts=[clean_text(t) for t in texts]
    X=v.transform(texts)
    return clf.predict(X)
