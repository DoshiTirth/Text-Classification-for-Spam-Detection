from flask import Flask, render_template, request,redirect
import joblib,os
import re,string,numpy as np
import nltk,pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

Fake_news = Flask(__name__)


import joblib
os.system("cls")
StandardScaler_=joblib.load(r'MLF_MODELS/StandardScaler.pkl')
print(f" * Sucessfully Loaded StandardScaler : {StandardScaler_}")
TFID_=joblib.load(r'MLF_MODELS/TFID.pkl')
print(f" * Sucessfully Loaded TFID : {TFID_}")
Hyper__=joblib.load(r'MLF_MODELS/Model.pkl')
print(f" * Sucessfully Loaded HyperTuned Model : {Hyper__}")

def Cleaning(Text:str):

    check= True if isinstance(Text,str) else False
    if check:
        Word=WordNetLemmatizer()
        stop=stopwords.words('english')
        text="".join([punt for punt in Text if punt not in string.punctuation]).lower()
        Splitter=re.split(r"\W+",text)
        word=" ".join([Word.lemmatize(stemming) for stemming in Splitter if stemming not in stop ])
        New_=pd.DataFrame({"clean_":[word]})
        perc=sum([1 for punt in Text if punt in string.punctuation])
        New_['Percentage_']=round(perc/(len(text)-text.count(' ')),2)*100
        New_['Lenght_']=len(Text)-Text.count(' ')
        transform=TFID_.transform(New_['clean_']).toarray()
        new_=pd.DataFrame(np.hstack([New_[['Lenght_','Percentage_']],transform]))
        tranform=StandardScaler_.transform(new_)
        mapp={1:"SPAM",0:"HAM"}

        return Hyper__.predict(tranform)
    







@Fake_news.route('/',methods=['GET'])
def home():
    return "GROUP MEMBERS ON THIS PROJECT\n1.Bruce-Arhin Shadrach, 20061815\n2.Tirth Doshi, 200609650\n3.Chandrika Ghale, 200575692\n4.Derick Appiah, 200584981"
    
    
@Fake_news.route('/Model_prediction',methods=["POST"])
def predict():
    Data=request.get_json()
    text=Data.get('txt',{})
    if text and isinstance(text,str):
        pred=Cleaning(text)
        if pred[0] == 1:
            res="Prediction of the Text :  Spam Detected 📰"
        else:
            res="Prediction of the Text : No Spam Detected📰 "
    else:
        return "Input is empty",404        
    return res
     
    
if __name__ =="__main__":
    Fake_news.run('0.0.0.0',8500)    
    