# classifier_model.py

import pandas as pd 
import numpy as np 
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc, recall_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import rfpimp 
from pdpbox import pdp, get_dataset, info_plots

'''
from bokeh.io import output_file, show
from bokeh.layouts import widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
'''

class Classifier_Model():
    '''
    classifier_model - train a classifier model
    Inputs data, splits into training/test, fits model, saves results
        input:
            model_type: string - 'random', 'adaboost', 'gradient', 'knn', 'logistic'
            X:          nparray. required. factors/features
            y:          nparray. required. target variable - should be [0, 1]
            params:     list. list of model-specific parameters for training. (see below)
            cv:         optional. int. If >0, run cross validation with cv folds and output the mean score
        output:
            train_pred: 1d array of n predictions, for every point in the training set
            resid:      1d array of n residuals, for every point in the training set
            norm_resid: 1d array of n residuals, normalized
            test_pred:  if X_test is passed in, 1d array of predictions for every point in the test set
            rmse:       float root mean squared error
            cv_score:   float. if cv is passed in, mean cv score
            intercept:  float. 
            coef:       d size array of coefficients
            r2:         r-squared
            r2_adj:     adjusted R-squared

        params:     random   = 
                    adaboost = {'base_estimator': None , 'n_estimators':50 , 
                                'learning_rate':1 , 'random_state':1234 }
                    gradient = {'learning_rate': 0.1  , 'n_estimators':100 , 
                                'min_samples_split':2  , 'min_samples_leaf':1 ,  
                                'max_depth':3, 'random_state':1234 }
    '''

    def __init__(self, model_type, X, y, params):
        ### attributes
        self.model_type = model_type
        self.X = X
        self.y = y
        self.params = params
        self.colnames = X.columns

        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.predicted = None
        self.probs = None
        self.resid = None
        self.resid_norm = None
        self.rmse = None
        self.rmse_log = None
        self.coef = None
        self.intercept = None
        self.cv_score = None
        self.a_max = None
        self.r2 = None
        self.r2_adj = None
        self.score, self.cv_scores = None, None 
        self.cv = None

        # check that the X and y data are shaped properly (at least 2 dim). 
        if self.X.ndim == 1:
            self.X = np.array(X).reshape(-1,1)
        #if self.y.ndim == 1:
            #self.y = np.array(y).reshape(-1,1)

        ### auto-run methods
        self.splitter()

        # train the specified regression
        if self.model_type == 'adaboost':
            self.adaboost_pred()
        elif self.model_type == 'random':
            self.randomf_pred()
        elif self.model_type == 'gradient':
            self.gradient_pred()
        elif self.model_type == 'knn':
            self.knn_pred()
        elif self.model_type == 'svm':
            self.svm_pred()
        elif self.model_type == 'logistic':
            self.logistic_pred()
        else:
            print("There is something wrong with your model selection. ")

        ### save the results
        self._fit()
        self.predictions()
        # - end function -
    
        
        
    def splitter(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)


    def randomf_pred(self):
        """
        random forest classifier
        input:
            model:
            X_train: nxd array
            y_train: n-size 1D array
            X_test: nxd array. Optional
            cv: int optional. If > 0, run cross validation with cv folds and output the mean score
        """
        # grid search cv
        self.model = RandomForestClassifier(**self.params)
        # - end function -
        
    
    def adaboost_pred(self):
        """
        adaboost classifier
        input:
            model:
            X_train: nxd array
            y_train: n-size 1D array
            X_test: nxd array. Optional
            cv: int optional. If > 0, run cross validation with cv folds and output the mean score
        """
        # grid search cv
        self.model = AdaBoostClassifier(**self.params)
        # - end function -


    def gradient_pred(self):
        """
        gradientboost classifier
        input:
            model: 
            X_train: nxd array
            y_train: n-size 1D array
            X_test: nxd array. Optional
            cv: int optional. If > 0, run cross validation with cv folds and output the mean score
        """
        self.model = GradientBoostingClassifier(**self.params)
        # - end function -


    def knn_pred(self):
        """
        knn classifier
        input:
            model:
            X_train: nxd array
            y_train: n-size 1D array
            cv: int optional. If > 0, run cross validation with cv folds and output the mean score
        """
        self.model = KNeighborsClassifier(**self.params)
        # - end function -


    def svm_pred(self):
        """
        svm classifier
        input:
            model:
            X_train: nxd array
            y_train: n-size 1D array
            X_test: nxd array. Optional
        """
        self.model = LogisticRegression(**self.params)
        # - end function -


    def logistic_pred(self):
        """
        logistic classifier
        input:
            model:
            X_train: nxd array
            y_train: n-size 1D array
            X_test: nxd array. Optional
            cv: int optional. If > 0, run cross validation with cv folds and output the mean score
        """
        self.model = LogisticRegression(**self.params)
        # - end function -


    def _fit(self):
        self.model.fit(self.X_train,self.y_train)


    def cv_analysis(self, cv_params):
        #cv_params = {'cv':5, 'score_types':{'prec_macro':'precision_macro', 'rec_mac':'recall_macro'}}
        self.cv = cv_params['cv']
        score_params = cv_params['score_types']
        scores = cross_validate(estimator=self.model, X=self.X_train, y=self.y_train, scoring=score_params, cv=cv_params['cv'], return_train_score=True)
        self.cv_scores = sorted(scores.keys())
        for k, val in cv_params['score_types']:
            print("CV Score: {} : {}".format(cv_params['score_types'][k], self.cv_scores))


    def predictions(self):
        '''
        determines the model predictions
        '''
        self.y_pred = self.model.predict(self.X_test)
        self.y_train = self.y_train.flatten()
        self.predicted = self.model.predict(self.X_train).flatten()  #weighted mean prediction of the classifiers
        self.probs = self.model.predict_proba(self.X_train)
        self.y_dec = self.model.decision_function(self.X_test)  #printing wants a 1d array?
        self.score = self.model.score(self.X_test, self.y_test)
        # end function


    def confusion_table(self):
        # Compute confusion matrix - uses confusion matrix from y_true and y_predicted
        print("Confusion Matrix")
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        header = [np.array(['Actual','Actual']), np.array(['True','False'])] 
        indexer = [np.array(['Predicted','Predicted']), np.array(['True','False'])] 
        df = pd.DataFrame([[tp,fp], [fn, tn]], columns = header, index = indexer)
        print(df)
        #return ct


    def summary_table(self):
        '''
        print the summary values for the model. use bokeh??
        '''
        st = "future table"
        print("Mean Accuracy: {}".format(self.model.score(self.X_test, self.y_test)))
        params = self.model.get_params()
        print()
        for p,val in params.items():
            print("{}: {}".format(p, val))
        print()


    def bokeh_page(self):
        '''
        #bokeh table - doesn't work
        '''
        from bokeh.io import output_file, show
        from bokeh.layouts import widgetbox
        from bokeh.models.widgets import Select

        def print_val_handler(attr, old, new):
            #print("Previous label: " + old)
            print("Updated label: " + new)

        output_file("select.html")
        dropdown = Select(title="Model:", value="logisitic", options=["logisitic", "svm", "knn", "gradient", "adaboost"])
        dropdown.on_change('value', print_val_handler)
        #dropdown.on_click(print_val_handler)
        show(widgetbox(dropdown))

    
    def feature_import(self):
        '''
        determine relative feature importance in model using permuation importance
        '''
        # permutation importances returns a df with feature, importance columns
        X_test_df = pd.DataFrame(self.X_test)
        y_test_df = pd.DataFrame(self.y_test)
        imp = rfpimp.importances(self.model, X_test_df, y_test_df, self.colnames)
        viz = rfpimp.plot_importances(imp)
        viz.view()

        # Compute permutation feature importances for scikit-learn models using
        # k-fold cross-validation (default k=3).
        if self.cv is not None:
            cv_imp = rfpimp.cv_importances(self.model, self.X_train, self.y_train, k=self.cv)


    def partial_plots(self):
        '''
        plot partial dependence plots 
        '''
        names = self.X.columns


    def ice_pred_plot(self, feature, feature_name='Feature'):
        '''
        partial dependence plot - pdpbox version
        https://towardsdatascience.com/introducing-pdpbox-2aa820afd312
        '''
        fig, axes, summary_df = info_plots.actual_plot(
            model=self.model, X=self.X_train, feature=feature, feature_name=feature_name)
        plt.plot()

    def plot_betas(self):
        #plot the magnitude of coefficients
        predictors = self.X_train.columns
        coef = Series(self.model.coef_, predictors).sort_values()
        coef.plot(kind='bar', title='Modal Coefficients')



    def plot_roc(self):
        '''
        plot a ROC curve for a model or lists of models
        '''
        # note: kNN does not plot an roc curve using a decision function...not 
        model_list = [1]
        n_classes = len(model_list)
        if self.y_dec.ndim == 1:
            self.y_dec = np.array(self.y_dec).reshape(-1,1)
        if self.y_test.ndim == 1:
            self.y_test = np.array(self.y_test).reshape(-1,1)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        plt.figure()
        lw = 2

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], self.y_dec[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color='darkorange',
                lw=lw, label='ROC curve (area = {:.2f})'.format(roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if self.model_type == 'adaboost':
            plt.title('ROC: {}  | {}, LR={}, N_Est={} |'.format(self.model_type, self.model.get_params()['algorithm'], self.model.get_params()['learning_rate'], self.model.get_params()['n_estimators']))
        elif self.model_type == 'gradient':
            plt.title('ROC: {}  | {}, LR={}, N_Est={} |'.format(self.model_type, self.model.get_params()['criterion'], self.model.get_params()['learning_rate'], self.model.get_params()['n_estimators']))
        else:
            plt.title('ROC: {}  |  |'.format(self.model_type))

        plt.legend(loc="lower right")
        plt.show()

