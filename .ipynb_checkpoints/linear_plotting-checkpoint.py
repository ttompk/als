import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels as sm
import numpy as np


def linearmodel_plots(LMresults, sample_n=None, height=3, width=6):
    ''' 
    This funtion calls all the linear model plotting functions at once. Only works with the 
    results from the LinearModel class.
    ---
    Input
        LMresults:  a fitted instance of LinearModel
    '''
    
    # hopefully plots
    pred_hist_plot(LMresults.predicted, sample_n)
    lm_plot(LMresults.predicted, LMresults.y_train, sample_n)
    resid_plot(LMresults.predicted, LMresults.resid, sample_n)
    resid_plot_norm(LMresults.predicted, LMresults.resid_norm, sample_n)
    normal_resid_distr(LMresults.resid_norm, sample_n)
    hetero_plot(LMresults.predicted, LMresults.resid_norm, sample_n)
    #leverage_plot(LMresults.resid_normal, _dummy_)
    # end function
    

def lm_plot(predicted, y_train, sample_n=None, height = 3, width = 6):
    '''
    plot predicted vs actual
    '''
    plt.style.use('seaborn')
    
    # if sampling value turned on then limit data in the output
    if sample_n != None:
        if len(predicted) > sample_n: 
            i = np.random.choice(len(predicted), sample_n, replace=False)
            predicted = predicted[i]
            y_train = y_train[i]
            
    # 
    plot_lm = plt.figure(1)
    plot_lm.set_figheight(height)
    plot_lm.set_figwidth(width)
    plt.scatter(predicted, y_train, alpha = 0.5)
    x = np.linspace(min(predicted), max(predicted), 100)
    #plt.plot(x, b0 + b1*x, color = 'r')   # - this only works with 1D X values.
    #plt.plot(x,x, color='red')
    
    # labels
    plot_lm.axes[0].set_title('Predicted vs Actual')
    plot_lm.axes[0].set_xlabel('Predicted Values')
    plot_lm.axes[0].set_ylabel('Actual Values')
    plt.show()
    # end function
    
    
def resid_plot(predicted, resid, sample_n=None, height=2, width=4):    
    '''
    This plot shows if residuals have non-linear patterns. There could be a non-linear 
    relationship between predictor variables and an outcome variable and the pattern 
    could show up in this plot if the model doesn’t capture the non-linear relationship. 
    If you find equally spread residuals around a horizontal line without distinct patterns, 
    that is a good indication you don’t have non-linear relationships.

    A good model's data are simulated in a way that meets the regression assumptions very well, 
    while a bad model's data are not.
    '''
    #plt.style.use('seaborn')
    
    # if sampling value turned on then limit data in the output
    if sample_n != None:
        if len(predicted) > sample_n: 
            i = np.random.choice(len(predicted), sample_n, replace=False)
            predicted = predicted[i]
            resid = resid[i]
    
    ## residual plot - x = predicted vs y = observed - actual
    plot_lm_r = plt.figure(1)
    plot_lm_r.set_figheight(height)
    plot_lm_r.set_figwidth(width)
    plot_lm_r.axes[0] = sns.residplot(predicted, resid, lowess=True, scatter_kws={'alpha':0.5}, line_kws={'color': 'red', 'lw': 2})
    plot_lm_r.axes[0].set_title('Predicted vs Residuals')
    plot_lm_r.axes[0].set_xlabel('Predicted Values')
    plot_lm_r.axes[0].set_ylabel('Model Residuals (obs - pred)')
    plt.show()
    # end function
    

def resid_plot_norm(predicted, norm_resid, sample_n=None, height=2, width=4):    
    '''
    This plot shows if normalized residuals have non-linear patterns. There could be a non-linear 
    relationship between predictor variables and an outcome variable and the pattern 
    could show up in this plot if the model doesn’t capture the non-linear relationship. 
    If you find equally spread residuals around a horizontal line without distinct patterns, 
    that is a good indication you don’t have non-linear relationships.

    A good model's data are simulated in a way that meets the regression assumptions very well, 
    while bad model's data are not.
    '''
    plt.style.use('seaborn')
    
    # if sampling value turned on then limit data in the output
    if sample_n != None:
        if len(predicted) > sample_n: 
            i = np.random.choice(len(predicted), sample_n, replace=False)
            predicted = predicted[i]
            norm_resid = norm_resid[i]
    
    ## residual plot - x = predicted vs y = observed - actual
    plot_lm_r = plt.figure(1)
    plot_lm_r.set_figheight(height)
    plot_lm_r.set_figwidth(width)
    plot_lm_r.axes[0] = sns.residplot(predicted, norm_resid, lowess=True, scatter_kws={'alpha':0.5}, line_kws={'color': 'red', 'lw': 2})
    plot_lm_r.axes[0].set_title('Predicted vs Normalized Residuals')
    plot_lm_r.axes[0].set_xlabel('Predicted Values')
    plot_lm_r.axes[0].set_ylabel('SalesPrice Model Residuals (obs - pred)')
    plt.show()
    # end function
    

def normal_resid_distr(norm_resid, sample_n=None, height=2, width=4):
    '''
    This plot shows if residuals are normally distributed. 
    Do residuals follow a straight line well or do they deviate severely? 
    It’s good if residuals are lined well on the straight dashed line.
    '''
    plt.style.use('seaborn')
    
    # if sampling value turned on then limit data in the output
    if sample_n != None:
        if len(norm_resid) > sample_n: 
            i = np.random.choice(len(norm_resid), sample_n, replace=False)
            norm_resid = norm_resid[i]
    
    ## residual normal quantile plot
    norm_q_plot = ProbPlot(norm_resid)
    plot_lm_q = norm_q_plot.qqplot(line='45', alpha=0.2, color='#4C72B0', lw=1)
    plot_lm_q.set_figheight(height)
    plot_lm_q.set_figwidth(width)
    # labels
    plot_lm_q.axes[0].set_title('Normal Quantile')
    plot_lm_q.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_q.axes[0].set_ylabel('Normalized Residuals')
    plt.show()
    # end function


def hetero_plot(predicted, norm_resid, sample_n=None, height=2, width=4):
    '''
    This plot shows if residuals are spread equally along the ranges of predictors. This 
    is how you can check the assumption of equal variance (homoscedasticity). It’s good 
    if you see a horizontal line with equally (randomly) spread points.
    '''
    plt.style.use('seaborn')
    
    # if sampling value turned on then limit data in the output
    if sample_n != None:
        if len(predicted) > sample_n: 
            i = np.random.choice(len(predicted), sample_n, replace=False)
            predicted = predicted[i]
            norm_resid = norm_resid[i]
    
    ## assess heteroscedasticity using absolute squarerooted normalized residuals
    # heteroscedasticity
    norm_sqrt_abs_resid_values = np.sqrt(np.abs(norm_resid))
    plot_lm_h = plt.figure(3)
    plot_lm_h.set_figheight(height)
    plot_lm_h.set_figwidth(width)
    plt.scatter(predicted, np.sqrt(np.abs(norm_resid)), alpha = 0.2)
    sns.regplot(predicted, np.sqrt(np.abs(norm_resid)),
                    scatter = False, ci = False, lowess = True,
                    line_kws ={'color': 'red', 'lw':2, 'alpha': 0.8})
    plot_lm_h.axes[0].set_title('Heteroscedasticity')
    plot_lm_h.axes[0].set_xlabel('Predicted Values')
    plot_lm_h.axes[0].set_ylabel('sqrt abs normalized residuals')
    plt.show()
    # end function
    

def leverage_plot(norm_resid, _dummy_, height=2, width=4):
    '''
    this plot not operational:
     - need values for this from statsmodels:   model.get_influence().hat_matrix_diag 
    '''
    plt.style.use('seaborn')
    
    # if sampling value turned on then limit data in the output
    if sample_n != None:
        if len(norm_resid) > sample_n: 
            i = np.random.choice(len(norm_resid), sample_n, replace=False)
            #predicted = predicted[i]
            norm_resid = norm_resid[i]
    
    ## leverage - outlier influence
    plot_lm_o = plt.figure(4)
    plot_lm_o.set_figheight(height)
    plot_lm_o.set_figwidth(width)
    #plt.scatter(model.get_influence().hat_matrix_diag, norm_resid, alpha = 0.2)
    #sns.regplot(model.get_influence().hat_matrix_diag, norm_resid, scatter=False, ci=False, lowess=True,line_kws ={'color': 'red', 'lw':2, 'alpha': 0.8})
    #plot_lm_o.axes[0].set_xlim(0,0.20)
    #plot_lm_o.axes[0].set_ylim(-3, 5)
    plot_lm_o.axes[0].set_title('Outlier Leverage')
    plot_lm_o.axes[0].set_xlabel('Leverage')
    plot_lm_o.axes[0].set_ylabel('Normalized Residuals')
    plt.show()
    # end function
    
    
def pred_hist_plot(predicted, sample_n=None, height=2, width=4):
    
    # if sampling value turned on then limit data in the output
    if sample_n != None:
        if len(predicted) > sample_n: 
            i = np.random.choice(len(predicted), sample_n, replace=False)
            predicted = predicted[i]
    
    plt.hist(predicted, bins=30, rwidth=0.9)
    plt.xlabel("predicted y")
    plt.ylabel("counts")
    plt.show()
    # end function