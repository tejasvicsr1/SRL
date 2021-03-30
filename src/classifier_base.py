import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def label_encode(y):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return(y,list(le.classes_),le)

def classification_report(y_val, predicted):
    report =  metrics.classification_report(y_val, predicted, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    return(df_report)


if __name__ == "__main__":
    df = pd.read_csv("../data/processed/interim.txt")
    print(df.columns)

    y,y_classes,le = label_encode(df['srl'])
    print(y_classes)


    skip_col = ['Label', 'chunk', 'postposition', 'head-POS', 'dependency-head', 'dependency', 'srl', 'predicate']
    x = df.drop(skip_col,axis=1)
    print(x.shape,y.shape)
    print(x.head(5))


    X_train, X_test, y_train, y_test = X_train, X_val, y_train, y_val = train_test_split(x, y,test_size=0.33,random_state=123)
    print(X_train.shape,X_val.shape,y_train.shape,y_val.shape)

    logreg = LogisticRegression()
    #class_weight='balanced'

    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)
    
    print(y_pred)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    
    # Todo - get classes ids
    y_pred = le.inverse_transform(y_pred)
    y_test  = le.inverse_transform(y_test)
    
    df_report = classification_report(y_test,y_pred)
    df_report.to_csv("../data/results/classifier_logistic_report.csv")

    