# alsfrs_functions.py
# ALS project
# functions: 
# linreg_scalers(), polyreg_scalers(), slope_mini_table(), plot_question_long(),
# plot_question_long_circles(), plot_line_over_actuals()
# 


def linreg_scalers(df, subject_col, y_col, x_col, y_label):
    '''
    creates linear regression slopes, intercepts for every subject in a df.
    input:
        df = df
        subject_col = name of subject id column
        y_col = name of first y data column
        x_col = name of column with x values
        y_label = name of y feature (for table output)
    output:  none
    return:  regression outputs in dataframe
    '''
    slopes = defaultdict()
    
    # list of every subject in table 
    subjects_list = list(np.unique(df[subject_col]))
    
    # for each subject
    for subj in subjects_list:
        data = df.loc[ df[subject_col]==subj].copy()
        y=data[y_col]
        x=data[x_col]

        # if only one record then skip
        if len(data)==1:
            continue
        else: # if missing values in x or y then skip
            if ~(y.isnull().any())  or (x.isnull().any()):
                y=np.array(y)
                x=np.array(x)
                
                # linregress is part of scipy.stats. 
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                slopes[subj] = (y_label, slope, intercept)

    # format as dataframe
    full_result = pd.DataFrame(slopes).transpose().reset_index()
    
    # relabel columns
    s_label = "slope_" + y_label 
    i_label = "intercept_" + y_label
    full_result.rename(index=str, columns={"index": "subject_id",
                                           0: "test", 
                                           1: s_label, 
                                           2: i_label }
                       , inplace=True)
    
    # make small table
    mini_table = slope_mini_table(full_result.copy(), i_label)
    
    # return table
    return full_result, mini_table
#end function


def polyreg_scalers(df, subject_col, y_col, x_col, y_label):
    '''
    creates polynomial linear regression slopes, intercepts for every subject in a df.
    input:
        df = df
        subject_col = name of subject id column
        y_col = name of dependent data column
        x_col = name of independent column (features)
        y_label = name of y feature (for table output)
    output:
        regression outputs in dataframe
    '''
    slopes = defaultdict()
    
    # list of every subject in table 
    subjects_list = list(np.unique(df[subject_col]))
    
    # for each subject
    for subj in subjects_list:
        data = df.loc[ df[subject_col]==subj].copy()
        y=data[y_col]
        x=data[x_col]

        # if only one record then skip
        if len(data)==1:
            continue
        else: # if missing values in x or y then skip
            if ~(y.isnull().any()) and  ~(x.isnull().any()):
                y=np.array(y)  #.reshape(-1,1)
                x=np.array(x).reshape(-1,1)
                
                polynomial_features= PolynomialFeatures(degree=2)
                x_poly = polynomial_features.fit_transform(x)
                
                model = LinearRegression()
                model.fit(x_poly, y)
                
                # linregress is part of scipy.stats. 
                #slope, intercept, r_value, p_value, std_err = linregress(x_poly, y)
                slopes[subj] = (model.coef_)
                

    # format as dataframe
    full_result = pd.DataFrame(slopes).transpose().reset_index()
    
    # relabel columns
    c1_label = "first_coef_" + y_label 
    c2_label = "second_coef_" + y_label
    c3_label = "third_coef_" + y_label
    full_result.rename(index=str, columns={"index": "subject_id", 
                                           0: c1_label ,
                                           1: c2_label, 
                                           2: c3_label }, inplace=True)
    
    # make small table
    mini_table = poly_slope_mini_table(full_result.copy())
    
    # return table
    return full_result, mini_table
# end function


def slope_mini_table(df, i_label):
    df.drop(['test', i_label], axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def poly_slope_mini_table(df):
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def plot_question_long(df, y_col, x_col, subject_list, ylim=None):
    '''
    plot multiple lines on one plot
    input:
        df = df
        y_col = name of feature to plot
        x_col = name of time variable (usually) to plot
        subject_list = list of subject to inlude in plot
    output:  plot
    return:  none
    '''
    plt.figure(figsize=(12,6))
    for subject in subject_list:
        this_df = df[ df['subject_id']==subject]
        plt.plot(this_df[x_col], this_df[y_col])
    plt.title("ALSFRS Total Score", fontsize=18)
    plt.ylabel("Score", fontsize=18)
    plt.xlabel("Days since Onset", fontsize=18)
    plt.xlim(left=0)
    plt.ylim(0, 42)
    plt.show()
    
def plot_question_long_circles(df, y_col, x_col, subject_list, ylim=None):
    '''
    plot multiple lines on one plot
    input:
        df = df
        y_col = name of feature to plot
        x_col = name of time variable (usually) to plot
        subject_list = list of subject to inlude in plot
    '''
    plt.figure(figsize=(12,6))
    for subject in subject_list:
        this_df = df[ df['subject_id']==subject]
        circ_df = this_df[ this_df['day_since_onset']== this_df['day_since_onset'].min()]
        plt.plot(this_df[x_col], this_df[y_col], linewidth= 2)
        plt.plot(circ_df[x_col], circ_df[y_col], color="black", marker='o')
    plt.title("ALSFRS Total Score", fontsize=18)
    plt.ylabel("Score", fontsize=18)
    plt.xlabel("Days since Onset", fontsize=18)
    plt.xlim(left=0)
    #if ylim!=None:
    plt.ylim(0, 42)
    plt.show()
    

def plot_line_over_actuals(df_points, df_slope, subject_col, y_col_points, y_col_slope, y_col_inter, x_col, y_label):
    '''
    plots a linear regression line over repeated measures
    input: 
        df_points = dataframe of all points
        df_slope = dataframe of slope, intercept values for each subject
        subject_col = string, column name of subjects
        y_col_points = string, column name of y-axis points in df_points
        y_col_slope = string, column name of slope values in df_slope
        y_col_inter = string, column name of y-intercept values in df_slope
        x_col = string, column name for x values in df_points
        y_label = string, pass through a label name
    output: plot with linear regression lines for each subject and line plots for repeated measures
    returns: none
    '''
    subject_list = df_slope[subject_col].unique()
    df_slope.dropna(inplace=True)
    df_slope.reset_index(inplace=True)
    for subj in subject_list:
        points = df_points[ df_points[subject_col]==subj]
        line = df_slope[ df_slope[subject_col]==subj]
        points_data = np.all(~np.isnan(points[y_col].values))
        if points_data and line.shape[0]!=0:
            plt.figure(figsize=(8,4))
            plt.plot(points[x_col].values, points[y_col_points].values, label="actual")
            plt.plot(points[x_col].values,
                     line[y_col_inter].values + line[y_col_slope].values * points[x_col].values, 
                     '-', label="predicted") 
            plt.title("{} : {}".format(subj, y_label))
            plt.ylabel(y_label, fontsize=18)
            plt.xlabel(x_col, fontsize=18)
            #plt.ylim(8,56)
            plt.show()

            