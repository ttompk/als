# regress_model.py

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

class Linear_Model():
    '''
    Linear_model class
    Inputs data, splits into training/test, fits model, saves results
        input:
            model_type: reqd. 'linear', 'lasso', 'ridge', 'random'
            X:          reqd. numpy array or pandas df.
            y:          reqd. numpy array. target variable
            split:      optional. boolean. default = True. Splits the X and y variables 
                        to 'train' and 'test' sets 
            test_size:  optional. float. default = 0.25. Train/test split size (test split proportion)
            alpha:      optional. list. default = [1]. List of alphas to use in the ridge and 
                        lasso regressions
            cv:         optional. int. If >1, run cross validation with cv folds and output the 
                        mean score
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
            
            
        params 
            gboost = {'loss': ’ls’, 'learning_rate': 0.1, 'n_estimators'= 100, 
                      'subsample'=1.0, 'criterion'=’friedman_mse’, 'min_samples_split'=2,
                      'min_samples_leaf'=1, 'min_weight_fraction_leaf'=0.0, 
                      'max_depth'=3, 'min_impurity_decrease'=0.0, 
                      'min_impurity_split'=None, 'init'=None, 'random_state'=None,
                      'max_features'=None, 'alpha'=0.9, 'verbose'=0, 
                      'max_leaf_nodes'=None, 'warm_start'=False, 'presort'=’auto’,
                      'validation_fraction'=0.1, 'n_iter_no_change'=None, 'tol'=0.0001' }
            random = {'n_estimators'=’warn’, criterion=’mse’, max_depth=None, 
                      'min_samples_split'=2, 'min_samples_leaf'=1, 
                      'min_weight_fraction_leaf'=0.0, 'max_features'=’auto’, 
                      'max_leaf_nodes'=None, 'min_impurity_decrease'=0.0, 
                      'min_impurity_split'=None, 'bootstrap'=True, 'oob_score'=False, 
                      'n_jobs'=None, 'random_state'=None, 'verbose'=0, 'warm_start'=False }
    '''

    def __init__(self, model_type, X, y, params, split=True, test_size=0.25, alpha = [1], cv = 5):
        # attributes
        self.model_type = model_type
        self.X = X
        self.y = y
        self.params = params
        self.alpha = alpha
        self.cv = cv
        self.test_size = test_size
        self.model = None
        self.X_train = X   # set to X in case data not split for training 
        self.X_test = None
        self.y_train = y   # set to y in case data not split for training
        self.y_test = None
        self.predicted = None
        self.test_pred = pd.DataFrame
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
        self.score = None

        # check that the X and y data are shaped properly.
        if self.X.ndim == 1:
            self.X = np.array(X).reshape(-1,1)
        if self.y.ndim == 1:
            self.y = np.array(y).reshape(-1,1)

        ## auto-run methods
        # split data
        if split == True:
            self.splitter()
        
        # run the specified regression
        if self.model_type == 'lasso':
            self.lasso_pred()
        elif self.model_type == 'ridge':
            self.ridge_pred()
        elif self.model_type == "linear":
            self.linear_pred()
        elif self.model_type == "random":
            self.random_pred()
        else:
            print("There is something wrong with your model selection. Chose 'linear', 'lasso', or 'ridge'")
        
        # save the results
        self.results()
        # end function
    

    def splitter(self):
        '''
        split X and y into train/test
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( self.X, self.y, test_size=self.test_size, random_state=1234)
    

    def linear_pred(self):
        """
        linear regression
        input:
            model: fits sklearn's LinearRegression()
            X_train: nxd array
            y_train: n-size 1D array
            X_test: nxd array. Optional
            cv: int optional. If >0, run cross validation with cv folds and output the mean score
        """
        # enter cross-validation function.
        self.model = LinearRegression()
        self.model.fit(self.X_train,self.y_train)
        # end function


    def ridge_pred(self):
        """
        ridge regression.
        get optimal alpha then get predictions and metrics using that alpha.
        input:
            X_train:   nxd array
            y_train:   n-size 1D array
            alphas:    list of floats, alphas to test.
            X_test:    nxd array. Optional
            cv:        int optional. If >1, run cross validation with cv folds and 
                       output the mean score
        """
        temp_cv_scores = np.ones(len(self.alpha))
        for idx, a in enumerate(self.alpha):
            temp_model = Ridge(alpha = a)
            #temp_model.fit(self.X_train, self.y_train)
            temp_cv_scores[idx] = (cross_val_score(temp_model, 
                                                   self.X_train, 
                                                   self.y_train, 
                                                   cv = self.cv)).mean()
        self.a_max = self.alpha[temp_cv_scores.argmax()]
        self.model = Ridge(alpha = self.a_max)
        self.model.fit(self.X_train, self.y_train)
        # end function


    def lasso_pred(self):
        '''
        lasso regression
        get optimal alpha then get predictions and metrics using that alpha
        '''
        temp_cv_scores = np.ones(len(self.alpha))
        for idx, a in enumerate(self.alpha):
            temp_model = Lasso(alpha = a)
            temp_model.fit(self.X_train, self.y_train)
            temp_cv_scores[idx] = (cross_val_score(temp_model, 
                                                   self.X_train, 
                                                   self.y_train, 
                                                   cv = self.cv)).mean()
        self.a_max = self.alpha[temp_cv_scores.argmax()]
        self.model = Lasso(alpha = self.a_max)
        self.model.fit(self.X_train, self.y_train)
        # end function
        
        
    def random_pred(self):
        """
        linear regression
        input:
            model: fits sklearn's RandomForestRegressor()
            X_train: nxd array
            y_train: n-size 1D array
            X_test: nxd array. Optional
            cv: int optional. If >0, run cross validation with cv folds and output the mean score
        """
        # enter cross-validation function.
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(self.X_train,self.y_train)
        # end function

        
    def gboost_pred(self):
        """
        linear regression with gradient boost
        input:
            model: fits sklearn's GradientBoostingRegressor()
            X_train: nxd array
            y_train: n-size 1D array
            X_test: nxd array. Optional
            cv: int optional. If >0, run cross validation with cv folds and output the mean score
        """
        # enter cross-validation function.
        self.model = GradientBoostingRegressor(**self.params)
        self.model.fit(self.X_train,self.y_train)
        # end function
        

    def test_predict(self, X_test=pd.DataFrame()):
        if len(X_test.index) == 0:
            self.test_pred = self.model.predict(self.X_test)
        else:
            self.test_pred = self.model.predict(X_test)
        return self.test_pred
        

    def results(self):
        '''
        determines the model fit metrics and saves them to the class object
        '''
        self.y_train = self.y_train.flatten()
        self.predicted = self.model.predict(self.X_train).flatten()
        self.resid = self.predicted - self.y_train
        self.resid_norm = (self.resid - self.resid.mean()) / self.resid.std()
        self.rmse = np.sqrt(mean_squared_error(self.y_train, self.predicted))
        self.rmse_log = np.sqrt(((( np.log(self.predicted+1)-np.log(self.y_train + 1) )**2).sum() ) / len(self.y_train))
        self.coef = self.model.coef_
        self.intercept = self.model.intercept_
        self.r2 = r2_score(self.y_train, self.predicted)
        self.r2_adj = 1 - (1-self.r2)*(len(self.y_train)-1)/(len(self.y_train)-self.X_train.shape[1]-1)
        self.score = self.model.score(self.X_test, self.y_test)

        if self.X_test is not None:
            self.test_pred = self.model.predict(self.X_test)

        if self.cv > 1:
            self.cv_score = (cross_val_score(self.model, self.X_train, self.y_train, cv = self.cv)).mean()
        # end function