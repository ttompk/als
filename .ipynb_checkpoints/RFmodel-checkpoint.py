# RFmodel.py
# runs RF regression model on model data
# outputs the model coefficients and analysis.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score


class run_rf():
    '''
    run a random forest model - classifier or regressor
    performs train/test split
    input:
        df = df
        features = list of strings. list of features to include in model (should NOT include target)
        target = string. column name of target variable
        limit_on = string. column name of values to subset data by
        exclude_columns = string. column name of feature to leave out of analysis.
    output:
        model outputs
    '''
    
    def __init__(self, df, features, class_regress, target, limit_on=None, exclude_columns=None, split=True):
        self.df = df
        self.features = features
        self.target = target
        self.class_regress = class_regress
        self.limit_on = limit_on
        self.exclude_columns = exclude_columns
        self.split=split
        
        self.model_data = None
        self.X_train=None
        self.X_test=None 
        self.y_train=None
        self.y_test=None
        self.y=None
        self.X=None
        self.model=None
        self.y_predicted=None
        self.resid_tbl=None
        
        #autorun functions
        self.prep_data()
        self.rf_model_fit()
        self.model_predictions()
        self.residuals_table()
        
        
    def prep_data(self):
        ### prepare input
        all_columns = self.features + [self.target]
        model_data = self.df[all_columns].copy()
        model_data.dropna(inplace=True)
        model_data.reset_index(drop=True, inplace=True)
    
        # limit data to only subjects with value = target
        if self.limit_on != None:
            limited_model_data = model_data[ model_data[self.limit_on]==1].copy()

            # assign target, features
            self.y = limited_model_data[self.target]
            self.X = limited_model_data[self.features]
            if exclude_columns!=None:
                self.X.drop(self.exclude_columns, axis=1, inplace=True)
        else:
            # assign target, features
            self.y = model_data[self.target]
            self.X = model_data[self.features]
            if exclude_columns!=None:
                self.X.drop(self.exclude_columns, axis=1, inplace=True)
    
        # split data
        if self.split == True:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                                                self.y,
                                                                                test_size=0.25)
        else:
            self.X_train=self.X
            self.y_train=self.y
    
    def rf_model_fit(self):
        ### Fit the rf model
        if self.class_regress == 'classifier':
            # run classiffier RF model
            self.model = RandomForestClassifier(n_estimators=10)
            self.model.fit(self.X_train, self.y_train)
        else:
            self.model = RandomForestRegressor(n_estimators=10)
            self.model.fit(self.X_train, self.y_train)
    

    def model_predictions(self):
        ### Make predictions
        # predicted values
        if self.class_regress == 'classiffier':
            # classiffier
            self.y_predicted = self.model.predict_proba(self.X_test)

        else:
            # regression
            self.y_predicted = self.model.predict(self.X_test)
        print("Accuracy/R2 Score: {}".format(self.model.score(self.X_test, self.y_test.values)))
    
    
    def residuals_table(self):
        # table of residuals
        if self.class_regress == 'regressor':
            self.resid_tbl = pd.DataFrame(self.y_test.values, columns=["test"])
            self.resid_tbl['predicted']=self.y_predicted
            self.resid_tbl['residual']=self.y_test.values - self.y_predicted
    
    
    def print_residuals(self):
        print("** Comparison Data **")
        print("Number of training observations: {}".format(len(self.X_train)))
        print("Number of test observations: {}".format(self.resid_tbl.shape[0]))
        print(self.resid_tbl.head())
        print("")
    
    def crossval_score(self):
        # score - R2 for regress, accuracy for class
        if self.class_regress == 'classifier':
            self.cvscores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
            print("Cross Val Scores: {}".format(self.scores))
            print("Mean Cross Val Score: {}".format(np.mean(self.scores)))
            
    def print_conf_table(self):
        # print confusion table from function
        if self.class_regress == 'classifier':
            confusion_table(self.model, self.X_test, self.y_test)
        else:
            print("This is not a classification experiment.")

    def print_f1_score(self):
        #print f1 score
        if self.class_regress == 'classifier':
            f_1_score = f1_score(self.y_test, self.model.predict(self.X_test))
            print("")
            print("** F1 Score **")
            print("F1 = 2 * (precision * recall) / (precision + recall)")
            print("F1 Score: {}%".format(round(f_1_score*100,2)))

    def plot_roc(self):
        # plot ROC from function
        if self.class_regress == 'classifier':
            plot_roc(self.model, self.X_test, self.y_test, model_type='forest', target=self.target)

    def plot_predictions(self):
        if self.class_regress != 'classifier':
            if self.limit_on==None:
                title = ""
            else:
                title = "Limit On: " + self.limit_on
            plot_predicted_regression(self.resid_tbl['predicted'], 
                                      self.resid_tbl['test'], 
                                      suptitle= "Target: "+target, 
                                      title=title)
    def plot_parallel(self):
        plot_model_parallel(self.y_test, self.y_predicted, units='units')
    
    
# end class


def feature_import(model, X_test, y_test):
    '''
    determine relative feature importance in model using permuation importance
    input:
        model = sklearn model (ex. model = sklearn.regressor() )
        X_test = numpy array. contains features (not target)
        y_test = numpy array. contains target only
    '''
    # permutation importances returns a df with feature, importance columns
    colnames = X_test.columns
    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.DataFrame(y_test)
    imp = rfpimp.importances(model, X_test_df, y_test_df, colnames)
    viz = rfpimp.plot_importances(imp)
    print("Permutation Feature Importance")
    viz.view()
    return imp
    #end function
    
    
def plot_predicted_regression(predicted, actual, suptitle='Regression', title=""):
    plt.plot(predicted, actual, '*')
    plt.suptitle(suptitle)
    plt.title(title)
    plt.ylabel("actual")
    plt.xlabel("predicted")
    plt.xlim(0, 2500)
    plt.ylim(0, 2500)
    plt.plot([0,2500], [0,2500])
    plt.show()
    #end function


def plot_model_parallel(y_test, y_predicted, units='units'):
    data = pd.DataFrame(y_test.values, columns=["test"])
    data['predicted']=y_predicted
    data['result']='result'
    parallel_coordinates(data, 'result',colormap=plt.get_cmap("Set2"), alpha=0.6)
    plt.title("Results of Model")
    plt.ylabel(units)
    plt.show()
    #end function


def plot_roc(model, X_test, y_test, model_type='forest', target='Target'):
    '''
    plot a ROC curve for a model
    '''
    # note: kNN does not plot an roc curve using a decision function...not 
    n_classes = len(model_list)
    y_dec = model.predict_proba(X_test)
    if y_dec.ndim == 1:
        y_dec = np.array(y_dec).reshape(-1,1)
    if y_test.ndim == 1:
        y_test = np.array(y_test).reshape(-1,1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure()
    lw = 2

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_dec[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color='darkorange',
            lw=lw, label='ROC curve (area = {:.2f})'.format(roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if model_type == 'adaboost':
        plt.title('ROC: {}  | {}, LR={}, N_Est={} |'.format(model_type, model.get_params()['algorithm'], model.get_params()['learning_rate'], model.get_params()['n_estimators']))
    elif model_type == 'gradient':
        plt.title('ROC: {}  | {}, LR={}, N_Est={} |'.format(model_type, model.get_params()['criterion'], model.get_params()['learning_rate'], model.get_params()['n_estimators']))
    elif model_type == 'forest':
        plt.title('ROC: {}  |  Citerion={}, N_Est={} |'.format(model_type, model.get_params()['criterion'], model.get_params()['n_estimators']))
    else:
        plt.title('ROC: {}  |  |'.format(model_type))
    plt.suptitle(target)
    plt.legend(loc="lower right")
    plt.show()
    #end function

