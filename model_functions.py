# model_functions.py
# support functions for ALS prediction model
# 
# library dependencies
from sklearn.metrics import confusion_matrix
import rfpimp


def plot_model_parallel(y_test, y_predicted, units='units'):
    '''
    parallel plot of actual and predicted values. A line connects each value pair.
    '''
    data = pd.DataFrame(y_test.values, columns=["test"])
    data['predicted']=y_predicted
    data['result']='result'
    parallel_coordinates(data, 'result',colormap=plt.get_cmap("Set1"), alpha=0.6)
    plt.title("Predicted Parallel Plot",  fontsize=18)
    plt.xlabel("Actual/Predicted)", fontsize=18)
    plt.ylabel(units, fontsize=18)
    plt.show()
    

def plot_predicted_regression(predicted, actual, suptitle='Regression', title=""):
    '''
    scatter plot of actual vs predicted. Note x and y limits.
    '''
    plt.plot(predicted, actual, '*')
    plt.suptitle(suptitle, fontsize=18)
    plt.title(title)
    plt.ylabel("actual", fontsize=18)
    plt.xlabel("predicted", fontsize=18)
    plt.xlim(0, 2500)
    plt.ylim(0, 2500)
    plt.plot([0,2500], [0,2500])
    plt.show()


def plot_roc(model, X_test, y_test, model_type='forest', target='Target'):
    '''
    plot a ROC curve for a model or lists of models
    '''
    # note: kNN does not plot an roc curve using a decision function...not 
    model_list = [1]
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
        plt.title('ROC: {}  | {}, LR={}, N_Est={} |'.format(model_type, model.get_params()['algorithm'],
                                                            model.get_params()['learning_rate'], 
                                                            model.get_params()['n_estimators']))
    elif model_type == 'gradient':
        plt.title('ROC: {}  | {}, LR={}, N_Est={} |'.format(model_type, model.get_params()['criterion'],
                                                            model.get_params()['learning_rate'], 
                                                            model.get_params()['n_estimators']))
    elif model_type == 'forest':
        plt.title('ROC: {}  |  Citerion={}, N_Est={} |'.format(model_type, model.get_params()['criterion'],
                                                               model.get_params()['n_estimators']))
    else:
        plt.title('ROC: {}  |  |'.format(model_type))
    plt.suptitle(target)
    plt.legend(loc="lower right")
    plt.show()

    
def confusion_table(model, X_test, y_test):
    '''
    creates a confusion matrix table.
    dependency: from sklearn.metrics import confusion_matrix
    input:
        model = an instantiated/trained sklearn model
        X_test = np array. values to make a prediction
        y_test = np array. labels for test set
    '''
    # Compute confusion matrix - uses confusion matrix from y_true and y_predicted
    print("** Confusion Matrix **")
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    header = [np.array(['Actual','Actual']), np.array(['True','False'])] 
    indexer = [np.array(['Predicted','Predicted']), np.array(['True','False'])] 
    df = pd.DataFrame([[tp,fp], [fn, tn]], columns = header, index = indexer)
    print(df)
    #return df   # df can be returned if needed
# end function


def feature_import(model, X_test, y_test):
    '''
    determine relative feature importance in model using permuation importance.
    dependency: import rfpimp  
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


def run_rf(df, features, class_regress, target, limit_on=None, exclude_columns=None, split=True):
    '''
    takes in a pandas df, trains a random forest model - classifier or regressor, makes predictions,
    produces plots, returns residuals
    dependency: 
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
        from sklearn.metrics import f1_score
        # this file:    confusion_table(), plot_roc(), plot_predicted_regression(), 
                        plot_model_parallel(), feature_import()
    input:
        df = df
        features = list of strings. list of features to include in model (should contain the target)
        class_regress = string. 'classifier' or 'regressor'
        target = string. column name of target variable
        limit_on = string. column values should be boolean. Data should be limited to this value.
        exclude_columns = leave this feature out of analysis.
        split = True/False. True = perform split/train/test.
    output:
        plots:  classifier = confusion table, roc curve
                regression = actual vs prediction, parallel plot
                both = feature importances
    returns: residuals, x test array, y test array
    '''
    ### prepare input
    all_features = features + [target]
    model_data = df[all_features].copy()
    model_data.dropna(inplace=True)
    model_data.reset_index(drop=True, inplace=True)
    
    # limit data to only subjects with value = target
    if limit_on != None:
        limited_model_data = model_data[ model_data[limit_on]==1].copy()

        # assign target, features
        y = limited_model_data[target]
        X = limited_model_data.drop(target, axis=1)
        if exclude_columns!=None:
            X.drop(exclude_columns, axis=1, inplace=True)
    else:
        # assign target, features
        y = model_data[target]
        X = model_data.drop(target, axis=1)
        if exclude_columns!=None:
            X.drop(exclude_columns, axis=1, inplace=True)
    
    # split data
    if split==True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.25, 
                                                        random_state=1234)
    else:
        X_train=X
        y_train=y
    
    ### Fit model
    if class_regress == 'classifier':
        # run classiffier RF model
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X_train,y_train)
    else:
        model = RandomForestRegressor(n_estimators=10)
        model.fit(X_train,y_train)
    

    ### Output
    # predicted values
    if class_regress == 'classiffier':
        # classiffier
        y_predicted = model.predict_proba(X_test)
       
    else:
        # regression
        y_predicted = model.predict(X_test)
    
    # table of residuals
    compare_data = pd.DataFrame(y_test.values, columns=["test"])
    compare_data['predicted']=y_predicted
    compare_data['residual']=y_test.values - y_predicted
    
    print("** Comparison Data **")
    print("Number of training observations: {}".format(len(X_train)))
    print("Number of test observations: {}".format(compare_data.shape[0]))
    print(compare_data.head())
    print("")
    
    # score - R2 for regress, accuracy for class
    print("Accuracy/R2 Score: {}".format(model.score(X_test, y_test.values)))
    if class_regress == 'classifier':
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print("Cross Val Scores: {}".format(scores))
        print("Mean Cross Val Score: {}".format(np.mean(scores)))

    # print confusion table
    if class_regress == 'classifier':
        confusion_table(model, X_test, y_test)

        #print f1 score
        f_1_score = f1_score(y_test, model.predict(X_test))
        print("")
        print("** F1 Score **")
        print("F1 = 2 * (precision * recall) / (precision + recall)")
        print("F1 Score: {}%".format(round(f_1_score*100,2)))

        # plot ROC
        plot_roc(model, X_test, y_test, model_type='forest', target=target)

    else:
        if limit_on==None:
            title = ""
        else:
            title = "Limit On: " + limit_on
        plot_predicted_regression(compare_data['predicted'], 
                                  compare_data['test'], 
                                  suptitle= "Target: "+target, 
                                  title=title)
        # plot parallels
        plot_model_parallel(y_test, y_predicted, units='units')
    
    # plot feature importances
    feature_import(model, X_test, y_test)
    
    return compare_data, X_test, y_test
# end function


def run_gboost(df, features, class_regress, target, limit_on, exclude_columns=None, params=None):
    '''
    run a gboost model - classifier or regressor
    input:
        df = df
        features = list of strings. list of features to include in model (should contain the target)
        target = string. column name of target variable
        limit_on = string. column values should be boolean
        exclude_columns = leave this feature out of analysis.
    output:
        model outputs
    
    regressor defauts:
    params = { 'loss':'ls', 'learning_rate':0.1, 'n_estimators':100, 
                'subsample':1.0, 'criterion':'friedman_mse', 'min_samples_split':2, 
                'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_depth':3, 
                'min_impurity_decrease':0.0, 'min_impurity_split':None, 'init':None, 
                'random_state':None, 'max_features'=None, 'alpha':0.9, 'verbose':0, 
                'max_leaf_nodes':None, 'warm_start':False, 'presort':'auto', 
                'validation_fraction':0.1, 'n_iter_no_change':None, 'tol':0.0001 }
    '''
    ### prepare input
    model_data = df[features].copy()
    model_data.dropna(inplace=True)
    model_data.reset_index(drop=True, inplace=True)
    
    # limit data to only subjects with value = target
    if limit_on != None:
        limited_model_data = model_data[ model_data[limit_on]==1].copy()

        # assign target, features
        y = limited_model_data[target]
        X = limited_model_data.drop(target, axis=1)
        if exclude_columns!=None:
            X.drop(exclude_columns, axis=1, inplace=True)
    else:
        # assign target, features
        y = model_data[target]
        X = model_data.drop(target, axis=1)
        if exclude_columns!=None:
            X.drop(exclude_columns, axis=1, inplace=True)
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.25, 
                                                        random_state=1234)
    
    ### Fit model
    if class_regress == 'classifier':
        # run classiffier RF model
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X_train,y_train)
    else:
        model = GradientBoostingRegressor(**params)
        model.fit(X_train,y_train)
    
    ### Output
    # predicted values
    if class_regress == 'classiffier':
        # classiffier
        y_predicted = model.predict_proba(X_test)
       
    else:
        # regression
        y_predicted = model.predict(X_test)
    
    compare_data = pd.DataFrame(y_test.values, columns=["test"])
    compare_data['predicted']=y_predicted
    compare_data['result']='result'
    
    print("** Comparison Data **")
    print("Number of training observations: {}".format(len(X_train)))
    print("Number of test observations: {}".format(compare_data.shape[0]))
    print(compare_data.head())
    print("")
    
    # score - R2 for regress, accuracy for class
    print("Accuracy/R2 Score: {}".format(model.score(X_test, y_test.values)))
    
    if class_regress == 'classifier':
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print("Cross Val Scores: {}".format(scores))
        print("Mean Cross Val Score: {}".format(np.mean(scores)))

    # print confusion table
    if class_regress == 'classifier':
        confusion_table(model, X_test, y_test)

        #print f1 score
        f_1_score = f1_score(y_test, model.predict(X_test))
        print("")
        print("** F1 Score **")
        print("F1 = 2 * (precision * recall) / (precision + recall)")
        print("F1 Score: {}%".format(round(f_1_score*100,2)))

        # plot ROC
        plot_roc(model, X_test, y_test, model_type='forest', target=target)

    else:
        if limit_on==None:
            title = ""
        else:
            title = "Limit On: " + limit_on
        
        plot_predicted_regression(compare_data['predicted'], 
                                  compare_data['test'], 
                                  suptitle= "Target: "+target, 
                                  title=title)
        # plot parallels
        plot_model_parallel(y_test, y_predicted, units='units')
    
    # plot feature importances
    feature_import(model, X_test, y_test)
#end function

    
def run_cv_gboost(df, features, class_regress, target, limit_on, exclude_columns, param_grid):
    '''
    takes in a pandas df, performs grid search CV on  Gradianet boosting regressor, returns
    best parameters.
    dependency: 
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import f1_score
        # this file:    confusion_table(), plot_roc(), plot_predicted_regression(), 
                        plot_model_parallel(), feature_import()
    input:
        df = df
        features = list of strings. list of features to include in model (should contain the target)
        class_regress = string. 'classifier' or 'regressor'
        target = string. column name of target variable
        limit_on = string. column values should be boolean. Data should be limited to this value.
        exclude_columns = leave this feature out of analysis.
        split = True/False. True = perform split/train/test.
    output:  none
    returns: best parameters of grid search
    '''
    
    ### prepare input
    model_data = df[features].copy()
    model_data.dropna(inplace=True)
    model_data.reset_index(drop=True, inplace=True)
    
    # limit data to only subjects with value = target
    if limit_on != None:
        limited_model_data = model_data[ model_data[limit_on]==1].copy()

        # assign target, features
        y = limited_model_data[target]
        X = limited_model_data.drop(target, axis=1)
        if exclude_columns!=None:
            X.drop(exclude_columns, axis=1, inplace=True)
    else:
        # assign target, features
        y = model_data[target]
        X = model_data.drop(target, axis=1)
        if exclude_columns!=None:
            X.drop(exclude_columns, axis=1, inplace=True)
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.25, 
                                                        random_state=1234)
    
    model = GradientBoostingRegressor()
    grid = GridSearchCV(estimator = model, param_grid = param_grid, 
                      cv = 3, n_jobs = 4, verbose = 2)
    grid.fit(X_train, y_train)
    
    print(grid.best_params_)
#end function

