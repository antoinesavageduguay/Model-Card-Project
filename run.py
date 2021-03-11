import sys
import pandas as pd
import io
import requests
import os  
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import RFE
from pdflatex import PDFLaTeX
from sklearn.metrics import confusion_matrix
    
X_train = pd.DataFrame()
X_test = pd.DataFrame()
y_train = pd.DataFrame()
y_test = pd.DataFrame()
X = pd.DataFrame()
y = pd.DataFrame()
logreg = None

def preprocess(df):
    df.rename(columns={' education':'education',' fnlwgt':'fnlwgt',' education-num':'education-num',' workclass': 'workclass',' capital-gain':'capital-gain', ' capital-loss': 'capital-loss', ' native-country': 'native-country',' hours-per-week': 'hours-per-week',' marital-status': 'marital-status', ' income':'income',' occupation':'occupation', ' race': 'race', ' sex':'sex', ' relationship': 'relationship'}, inplace=True)
    df['native-country'] = df['native-country'].replace('?',np.nan)
    df['workclass'] = df['workclass'].replace('?',np.nan)
    df['occupation'] = df['occupation'].replace('?',np.nan)
    df.dropna(how='any',inplace=True)
    #making dummy variable
    df['income'] = df['income'].map({' <=50K': 0, ' >50K': 1})
    df['marital-status'] = df['marital-status'].map({' Separated': 0, ' Never-married': 1, 
                                                             ' Married-AF-spouse': 2, ' Divorced': 3, ' Married-civ-spouse': 4, 
                                                             ' Married-spouse-absent': 5, ' Widowed': 6}).astype(int)
    df['workclass'] = df['workclass'].map({' ?': 0, ' Federal-gov': 1, 
                                                             ' Local-gov': 2, ' Never-worked': 3, ' Private': 4, 
                                                             ' Self-emp-inc': 5, ' Self-emp-not-inc': 6,' State-gov':7,' Without-pay':8}).astype(int)
    df['education'] = df['education'].map({' 10th': 0, ' 11th': 1, 
                                                             ' 12th': 2, ' 1st-4th': 3, ' 5th-6th': 4, 
                                                             ' 7th-8th': 5, ' 9th': 6,' State-gov':7,' Assoc-acdm':8,' Assoc-voc':9,' Bachelors':10,' Doctorate':11,' HS-grad':12,' Masters':13,' Preschool':14,' Prof-school':15,' Some-college':16  })
    df['occupation'] = df['occupation'].map({' ?': 0, ' Adm-clerical': 1, 
                                                             ' Armed-Forces': 2, ' Craft-repair': 3, ' Exec-managerial': 4, 
                                                             ' Farming-fishing': 5, ' Handlers-cleaners': 6,' Machine-op-inspct':7,' Other-service':8,' Priv-house-serv':9,' Prof-specialty':10,' Protective-serv':11,' Sales':12,' Masters':13,' Tech-support':14,' Transport-moving':15})
    df['relationship'] = df['relationship'].map({' Husband': 0, ' Not-in-family': 1, 
                                                             ' Other-relative': 2, ' Own-child': 3, ' Unmarried': 4, 
                                                             ' Wife': 5})
    df['race'] = df['race'].map({' Amer-Indian-Eskimo': 0, ' Asian-Pac-Islander': 1, 
                                                             ' Black': 2, ' Other': 3, ' White': 4})
    df['sex'] = df['sex'].map({' Female': 0, ' Male': 1})
    df.drop(['native-country'], axis=1, inplace=True)

def train_test_split(df, train_size_per = 0.8,test_size_per = 0.2, random_seed = 7):
    ind = np.arange(len(df))
    np.random.shuffle(ind)
    np.random.seed(random_seed)
    ind_train = ind[0:int(np.floor(len(df) * train_size_per))]
    ind_test = ind[int(np.floor(len(df) * train_size_per)):,]
    train_df = df.iloc[ind_train]
    test_df = df.iloc[ind_test]
    return train_df, test_df

    
def fetch():
    # fetching the data
    df = pd.read_csv('income_evaluation.csv')
    #preprocessing of the data
    preprocess(df)
    split = train_test_split(df)
    features = [i for i in df.columns.values.tolist() if i not in ['income']]
    global X
    global y
    global X_train
    global X_test
    global y_train
    global y_test
    X = df[features]
    y = df['income']
    X_train = split[0][features]
    X_test = split[1][features]
    y_train = split[0]['income']
    y_test = split[1]['income'] 




def train():
    global X
    global y
    global X_train
    global X_test
    global y_train
    global y_test
    global logreg
    fetch()
    if(X.empty):
        print('please fetch the data before training the algorithm')
    features = [i for i in df.columns.values.tolist() if i not in ['income']]
    selected_features = []
    logreg = LogisticRegression()
    rfe = RFE(logreg)
    rfe = rfe.fit(X_train, y_train)
    rfe_results = rfe.support_
    for j in range(len(rfe_results)):
        if(rfe_results[j]):
            selected_features.append(features[j])
    X_train = X_train[selected_features]
    X_test = X_test [selected_features]
    print('selected features: ', selected_features)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    


# Compute the evaluation metrics and figures
def evaluate():
        global X_train
        global X_test
        global y_train
        global y_test
        global logreg
        if(logreg is None):
         print('please train the model before evaluating the results')
        y_pred = logreg.predict(X_test)
        confusion_matrix_glob = confusion_matrix(y_test, y_pred)
        
        # some data manipulation to do the 1st graph
        test_df = pd.DataFrame()
        test_df['y_test'] = y_test
        test_df['y_pred'] = y_pred
        test_df['num'] = list(range(0,len(y_test)))  
        sex_sr = []
        for i in set(X_test['sex']):
            num_test = test_df.loc[X_test[X_test['sex'] == i].index.values]
            sex_sr.append((confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[0][0]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1])/(confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][0]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[0][1]+(confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[0][0]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1])))
        race_sr = []
        for i in set(X_test['race']):
            num_test = test_df.loc[X_test[X_test['race'] == i].index.values]
            race_sr.append((confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[0][0]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1])/(confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][0]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[0][1]+(confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[0][0]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1])))
        education_sr = []
        for i in set(X_test['education-num']):
            num_test = test_df.loc[X_test[X_test['education-num'] == i].index.values]
            if(len(confusion_matrix(num_test['y_test'],np.array(num_test['y_pred'])))>1):
                education_sr.append((confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[0][0]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1])/(confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][0]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[0][1]+(confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[0][0]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1])))
            else:
                education_sr.append(1)
        graph_sr = sex_sr+(race_sr)+(education_sr)
        
        # plot the 1st graph
        height = graph_sr
        plt.figure(figsize = (40, 5))
        bars = ('Male','Female','Amer-Indian','Asian-Pac','Black','Other','White','Ed-1','Ed-2','Ed-3','Ed-4','Ed-5','Ed-6','Ed-7',
           'Ed-8','Ed-9','Ed-10','Ed-11','Ed-12','Ed-13','Ed-14','Ed-15','Ed-16')
        x_pos = np.arange(len(bars))
        plt.bar(x_pos, height, color = (0.5,0.1,0.5,0.6))
        plt.title('Accuracy by group')
        plt.xlabel('categories')
        plt.ylabel('values')
        plt.xticks(x_pos, bars)
        plt.savefig('accuracy.png')
        
        # some data manipulation to do the 2nd graph
        test_df = pd.DataFrame()
        test_df['y_test'] = y_test
        test_df['y_pred'] = y_pred
        test_df['num'] = list(range(0,len(y_test)))  
        sex_sr = []
        for i in set(X_test['sex']):
            num_test = test_df.loc[X_test[X_test['sex'] == i].index.values]
            sex_sr.append((confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1])/(confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][0]))
        race_sr = []
        for i in set(X_test['race']):
            num_test = test_df.loc[X_test[X_test['race'] == i].index.values]
            race_sr.append((confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1])/(confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][0]))
        education_sr = []
        for i in set(X_test['education-num']):
            num_test = test_df.loc[X_test[X_test['education-num'] == i].index.values]
            if(len(confusion_matrix(num_test['y_test'],np.array(num_test['y_pred'])))>1):
                education_sr.append((confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1])/(confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][1]+confusion_matrix(num_test['y_test'],np.array(num_test['y_pred']))[1][0]))
            else:
                education_sr.append(1)
        graph_sr = sex_sr+(race_sr)+(education_sr)
        
        # plot the 1st graph
        height = graph_sr
        plt.figure(figsize = (40, 5))
        bars = ('Male','Female','Amer-Indian','Asian-Pac','Black','Other','White','Ed-1','Ed-2','Ed-3','Ed-4','Ed-5','Ed-6','Ed-7',
           'Ed-8','Ed-9','Ed-10','Ed-11','Ed-12','Ed-13','Ed-14','Ed-15','Ed-16')
        x_pos = np.arange(len(bars))
        plt.bar(x_pos, height, color = (0.5,0.1,0.5,0.6))
        plt.title('False Positive')
        plt.xlabel('categories')
        plt.ylabel('values')
        plt.xticks(x_pos, bars)
        plt.savefig('false_pos.png')
        confusion_matrix_glob = confusion_matrix(y_test, y_pred)
        print(confusion_matrix_glob)
        
# Compile the PDF documents
def build_paper():
  os.system("pdflatex card.tex")


 


###############################
# No need to modify past here #
###############################

supported_functions = {'fetch': fetch,
                       'train': train,
                       'evaluate': evaluate,
                       'build_paper': build_paper}

# If there is no command-line argument, return an error
if len(sys.argv) < 2:
  print("""
    You need to pass in a command-line argument.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
  """)
  sys.exit(1)

# Extract the first command-line argument, ignoring any others
arg = sys.argv[1]

# Run the corresponding function
if arg in supported_functions:
  supported_functions[arg]()
else:
  raise ValueError("""
    '{}' not among the allowed functions.
    Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
    """.format(arg))