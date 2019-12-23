import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, label=None):
    """
    The ROC curve, modified from
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91

    tpr = true positive rate
    fpr = false positive rate
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')

def plot_feature_importance(feature_imp, feature_imp_idx):
    """
    Plot the feature importance graph
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10,8))

    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, cmap,
                            normalize=False,
                            title=None):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    References: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    from sklearn import metrics
    from sklearn.utils.multiclass import unique_labels
    import matplotlib.pyplot as plt
    import numpy as np

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    title = 'Accuracy Score: {0}'.format(metrics.accuracy_score(y_true, y_pred))
    plt.title(title, size = 15)
    return ax

def freqTable(lines_list, cols_list, show_total, perc_type=None):
    '''
    lines_list: df column list for exhibition in lines of frequency table. Ex: [df["col1"], df["col2"]]
    cols_list: df column list for exhibition in columns of frequency table. Ex: [df["col1"], df["col2"]]
    show_total: Show totals? True | False
    perc_type: "columns" for column percentage; "index" for row percentage or None for absolute value

    References: https://pbpython.com/pandas-crosstab.html
    '''

    import pandas as pd

    if (perc_type==None):
        freq = pd.crosstab(
            index=lines_list
            ,columns=cols_list
            ,margins=show_total
            ,margins_name="Total"
        )
    else:
        freq = pd.crosstab(
            index=lines_list
            ,columns=cols_list
            ,margins=show_total
            ,margins_name="Total"
            ,normalize=perc_type
        )

    return freq

def percMissing(df):
    '''
    Verify Missing columns

    Input: DataFrame
    '''

    import pandas as pd

    # Create Series object
    s = pd.Series()
    # Check each column
    for col in df.columns:
        # Fill series object (index is the column name and the value is the % of missing rows)
        # the count() funcion does not return missing
        s.at[col] = ((len(df) - df[col].count()) / len(df)) * 100
        # Create dataframe with results
        df_missing = pd.DataFrame({'col':s.index, 'perc_missing':s.values})

    return df_missing.sort_values(by=['perc_missing'], ascending=False)

def getEmailDomain(email):
    '''
    Extracts email domain

    Input: email string
    '''

    return email.split('@')[1].lower()

def getPiece1EmailDomain(domain):
    '''
    Extracts piece 1 from email domain
    Example: teste@domain.com.br will return '.com' string

    Input: email domain string
    '''

    if (len(domain.split('.')) == 2):
        return domain.split('.')[-1]
    elif (len(domain.split('.')) >= 3):
        return domain.split('.')[-2]
    else:
        return 'missing'

def getPiece2EmailDomain(domain):
    '''
    Extracts piece 2 from email domain
    Example: teste@domain.com.br will return '.br' string

    Input: email domain string
    '''

    if (len(domain.split('.')) >= 3):
        return domain.split('.')[-1]
    else:
        return 'missing'

def getEmailUser(email):
    '''
    Extracts email user from email
    Example: teste@domain.com.br will return 'teste' string

    Input: email string
    '''

    return email.split('@')[0].lower()

def percentageNumberInStr(string):
    '''
    Calculates the percentage of numbers contained in a string

    Input: string
    '''

    count = 0
    if len(string) > 0:
        for char in string:
            if char in ['0','1','2','3','4','5','6','7','8','9']:
                count += 1
        return count/len(string)
    else:
        return 0

def oversampleSMOTE(X, y):
    '''
    Resample a dataset using SMOTE oversample

    Input:
        X = dataframe with x variables (explanatory variables)
        y = dataframe with y variable (variable to predict)

    Output:
        df[0] = X dataframe resampled
        df[1] = y dataframe resampled
    '''

    from imblearn.over_sampling import SMOTE
    import pandas as pd

    sm = SMOTE(random_state=123)
    X_resampled, y_resampled = sm.fit_resample(X, y.ravel())

    # Get column names
    X_cols = X.columns.values
    y_cols = [y.name]

    return pd.DataFrame(X_resampled, columns=X_cols) , pd.DataFrame(y_resampled, columns=y_cols)

def buildDummyVariables(df, category_vars):
    '''
    Return a DF with categoric variables dummized

    Input:
        df = dataframe that contains the variables to be dummized
        category_vars = list of variables to be dummized

    Output:
        df = X dataframe with the dummy variables, excluding the original categoric variables
    '''

    import pandas as pd

    df_new = df.copy()

    # Build new DF with dummy variables
    for var in category_vars:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(df[var], prefix=var)
        df_new = df_new.join(cat_list)
    
    # build a list of columns to keep, excluding the original category variables
    df_vars = df_new.columns.values.tolist()
    to_keep = [i for i in df_vars if i not in category_vars]
    
    # return new df with new dummt columns, excluding the original categoric variables
    return df_new[to_keep]