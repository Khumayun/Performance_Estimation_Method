from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import semopy
from sklearn.cross_decomposition import CCA
# from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import TheilSenRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.stats import linregress
from scipy import stats
import pylab
from sklearn.impute import SimpleImputer


def preprocess(path_to_file, verbal=False):
    df = pd.read_csv(path_to_file)
    df = df.drop(['repo_fullname'],axis=1)
    try:
        df = df.drop(['id'],axis=1)
    except Exception as e:
        pass
    df.isnull().sum().sum()
    df.dtypes.value_counts()
    # df[df.columns[df.isnull().any()]]
    # df.shape
    col_type = list(df.dtypes)
    fig = df.isna().sum().reset_index(name="n").plot.bar(x='index', y='n', rot=45, figsize=(20, 20)).get_figure()
    fig.savefig('results/zero_val.png')

    new_df = df.dropna(how='all', axis=1)
    cols = new_df.columns[new_df.isnull().any()].to_list()
    # cols
    # new_df.shape
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(new_df[cols])
    X_trans = imp.transform(new_df[cols])
    X_trans = pd.DataFrame(X_trans, columns=cols)
    # X_trans.shape
    new_df.drop(cols, inplace=True, axis=1)
    imp_df = pd.concat([new_df, X_trans], axis=1)

    ## plot normally distribyted repo_age_days feature
    ax = imp_df['repo_age_days'].plot(kind='hist', title='Plot of repo_age_days feature')
    fig = ax.get_figure()
    if verbal:
        fig.savefig('results/repo_age_days_plot.png')
    return imp_df


def full_analysis(path_to_file):
    df = preprocess(path_to_file, verbal=True)
    # factor analysis
    f_max, X_train, X_test, y_train, y_test = factor_analysis_and_tsne(df)
    # cca
    cca(df)
    # sem
    sem(df)
    # correlation
    p, sort_p = correlation(df, 0.80, 'pearson')
    correlation(df, 0.80, 'spearman')
    # pvalues
    p_values(df, sort_p)
    # mlt table
    # fig, ax = render_mpl_table(df, header_columns=0, col_width=3.0, )
    # fig.savefig("results/table_mlp.png")
    # ks statistics
    ks_qq(df)

def ks_qq(imp_df):
    data_ = imp_df['repo_age_days']
    data_norm = np.random.normal(np.mean(data_), np.std(data_), len(imp_df))
    values, base = np.histogram(data_)
    values_norm, base_norm = np.histogram(data_norm)
    cumulative = np.cumsum(values)
    cumulative_norm = np.cumsum(values_norm)
    plt.plot(base[:-1], cumulative, c='blue', label='Emperical CDF')
    plt.plot(base_norm[:-1], cumulative_norm, c='green', label='Normal CDF')
    plt.legend(loc="upper left")
    plt.savefig('results/ks.png', bbox_inches='tight')
    plt.clf()

    # measurements = np.random.normal(loc = 20, scale = 5, size=100)
    stats.probplot(data_, dist="norm", plot=pylab)
    pylab.savefig("results/q-q plot.png")


def render_mpl_table(data, col_width=3.0, row_height=0.225, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax
def p_values(df, sort_spearman):
    # print for overleaf table
    pv = p_val(df, sort_spearman, method='spearman')
    # print(*pv[:6], sep='\n')
    df = pd.DataFrame(pv, columns=['Feature 1', 'Feature 2', 'Pearson Coefficient', 'Associated p-value'])
    with open('results/coefficients_and_pval.txt', 'w') as f:
        for x in zip(df['Feature 1'], df['Feature 2'], df['Pearson Coefficient'], df['Associated p-value']):
            # print(f"{x[0]} & {x[1]} & {x[2]:.2f} & {x[3]:.2f}")
            f.writelines(f"{x[0]} & {x[1]} & {x[2]:.2f} & {x[3]:.2f}\n")
    f.close()


def p_val(imp_df, corr_ls, method='spearman'):
  corr_ls1 = []
  for i in range(len(corr_ls)):
    a , b = imp_df[corr_ls[i][0]], imp_df[corr_ls[i][1]]
    if method == 'pearson':
      val = linregress(a,b)
      corr_ls1.append((*corr_ls[i],val.pvalue))
    elif method == 'spearman':
      val = stats.spearmanr(a,b)
      corr_ls1.append((*corr_ls[i],val.pvalue))
  return corr_ls1

def get_unique_features(ls):
  col = set()
  for i in range(len(ls)):
    f1,f2,_ = ls[i]
    if f1 not in col:
      col.add(f1)
    if f2 not in col:
      col.add(f2)
  #print(col)
  return len(col),col

def correlation(dataset, threshold, method):
    col_corr = set() # Set of all the names of deleted columns
    corr_ls = []
    corr_matrix = dataset.corr(method=method)

    with open('results/correlations'+method+'.png', 'w') as f:
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] # getting the name of column
                    col_corr.add(colname)
                    corr_ls.append((dataset.columns[i],dataset.columns[j],corr_matrix.iloc[i, j]))
                    f.writelines(str((dataset.columns[i],dataset.columns[j],corr_matrix.iloc[i, j])))

    get_unique_features(corr_ls)
    sort_pear = sorted(corr_ls, key=lambda a: a[2], reverse=True)
    _, pear_cols = get_unique_features(sort_pear[:3])
    sns.pairplot(dataset[list(pear_cols)])
    plt.savefig('results/pairplot_correlations'+method+'.png')
    plt.clf()
    return corr_ls, sort_pear


def factor_analysis_and_tsne(imp_df):
    trial_df = imp_df.select_dtypes(exclude=['object'])
    trial_df = trial_df.loc[:, (trial_df == 0).mean() < .5]  # drops columns with >= 50% data having zeroes
    chi_square_value, p_value = calculate_bartlett_sphericity(trial_df)
    print(f'Chi-square: {chi_square_value},'
          f'p-value: {p_value} ')
    kmo_all, kmo_model = calculate_kmo(trial_df)
    print(f'Kmo_all: {kmo_all},'
          f'Kmo_model: {kmo_model}')
    fa = FactorAnalyzer()
    trial_df = trial_df - trial_df.mean()
    fa.fit(trial_df)
    eigen_values, vectors = fa.get_eigenvalues()
    print(f'Eigen values: {eigen_values}')

    # Plot and save eigen scree
    plt.scatter(range(1, trial_df.shape[1] + 1), eigen_values)
    plt.plot(range(1, trial_df.shape[1] + 1), eigen_values)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.savefig('results/eigenscree.png')
    plt.clf()
    # plt.show()

    fa = FactorAnalyzer(6, rotation="varimax")
    fa.fit(trial_df)
    factor_df = pd.DataFrame(fa.loadings_, columns=["F1","F2",'F3','F4','F5','F6'],index=trial_df.columns.to_list())
    #factor_df.set_index(trial_df.columns.to_list())
    print(f'Factors: {factor_df}')
    max_fac = []
    for col in factor_df:
        max_fac.append(factor_df[col].idxmax())

    # split
    X, y = trial_df[max_fac], imp_df['wf_count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    ##### TSNE #####

    X_cop = X.copy()

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
    tsne_results = tsne.fit_transform(X_cop)

    X_cop['tnse1'] = tsne_results[:, 0]
    X_cop['tnse2'] = tsne_results[:, 1]

    # fig, axes = plt.subplots(len(train_c.columns), figsize=(4, 200))
    # for i in range(len(train_c.columns)):
    # col = r
    c = len(set(imp_df['repo_languages']))
    # print(col)
    sns.scatterplot(
        x="tnse1", y="tnse2",
        hue=imp_df['repo_languages'],
        palette=sns.color_palette("husl", c),
        data=X_cop,
        legend=False,
        alpha=0.3
    )
    plt.savefig('results/tsne_for_repo_age_days.png')
    plt.clf()

    return factor_df.max(axis=0), X_train, X_test, y_train, y_test

def cca(imp_df):
    # Instantiate the Canonical Correlation Analysis with 2 components
    my_cca = CCA(n_components=2)
    # Split The data
    F1 = imp_df[['contributors_count', 'contributors_top_avg_commits', 'contributors_top_avg_participation_week',
                 'contributors_top_avg_additions', 'contributors_top_avg_deletions']]
    F2 = imp_df[['forks_count', 'forks_avg_per_day', 'forks_avg_max_per_day']]
    F3 = imp_df[['stars_count', 'stars_avg_per_day_real', 'stars_max_per_day']]
    F4 = imp_df[['commits_days_since_last', 'commits_total_lines_added',
                 'commits_total_lines_removed', 'commits_avg_added',
                 'commits_avg_removed', 'commits_avg_files_changed',
                 'commits_avg_message_length', 'commits_avg_per_day_real',
                 'commits_max_per_day']]
    F5 = imp_df[['repo_topics', 'repo_branches', 'repo_age_days', 'repo_workflows',
                 'repo_languages', 'repo_milestones', 'repo_watchers',
                 'repo_deployments', 'repo_readme_length', 'repo_network_members']]
    F6 = imp_df[['issues_total_comments', 'issues_count', 'issues_open', 'issues_labels',
                 'issues_avg_labels', 'issues_avg_closing_time',
                 'issues_avg_comment_time', 'issues_avg_comments',
                 'issues_avg_comment_length', 'issues_avg_title_length',
                 'issues_avg_body_length', 'issues_avg_per_day',
                 'issues_avg_per_day_real']]
    # Fit the model
    my_cca.fit(F1, F5)

    xrot = my_cca.x_rotations_
    yrot = my_cca.y_rotations_

    # Put them together in a numpy matrix
    xyrot = np.vstack((xrot,yrot))

    nvariables = xyrot.shape[0]
    # print(nvariables)
    plt.figure(figsize=(15, 15))
    plt.xlim((-1,1))
    plt.ylim((-1,1))

    # Plot an arrow and a text label for each variable
    for var_i in range(nvariables):
      x = xyrot[var_i,0]
      y = xyrot[var_i,1]

      plt.arrow(0,0,x,y)
      plt.text(x,y,imp_df.columns[var_i], color='red' if var_i >= 5 else 'blue')

    plt.savefig('results/cca_rotations.png')
    plt.clf()

def sem(imp_df):
    # Specify the model relations using the same syntax given before
    model_spec = """
      # measurement model
        CodeQuality =~  commits_count + contributors_count + pulls_count
        Maintainability =~ repo_languages + repo_branches
        Motivation =~ commits_avg_per_day_real + commits_max_per_day + releases_count
      # regressions
        ProjPerformace =~  stars_count + wf_count + issues_total_comments 
        ProjPerformace ~ CodeQuality + Motivation + Maintainability
    """

    # Instantiate the model
    model = semopy.Model(model_spec)

    # Fit the model using the data
    model.fit(imp_df)

    # Save the results using the inspect method
    res = model.inspect()
    res.to_csv('results/sem.csv')
    # return res

def regressions(X_train, X_test, y_train, y_test):

    ########### TheilSenRegressor ##############
    senreg = TheilSenRegressor(random_state=0).fit(X_train, y_train)
    sscore = senreg.score(X_train, y_train)
    y_pred = senreg.predict(X_test)
    mean_squared_error(y_test, y_pred)
    scoefs = senreg.coef_

    ########### RandomClassifier ##############
    clf = RandomForestClassifier()
    params = {'n_estimators': [30, 50, 100],
              'max_features': ['sqrt', 'log2', 10]}
    gsv = GridSearchCV(clf, params, cv=3,
                       n_jobs=-1, scoring='f1')
    gsv.fit(X_train, y_train)
    with open('results/gsv_X_train.txt', 'w') as f:
        f.writelines(classification_report(y_train, gsv.best_estimator_.predict(X_train)))
    f.close()
    with open('results/gsv_X_test.txt', 'w') as f:
        f.writelines(classification_report(y_test, gsv.best_estimator_.predict(X_test)))
    f.close()

    # Make predictions for the test set
    y_predtest = gsv.best_estimator_.predict(X_test)
    accuracy_score(y_test, y_predtest)
    # View confusion matrix for test data and predictions
    # confusion_matrix(y_test, y_predtest)
    plot_confusion_matrix(gsv, X_test, y_test)
    plt.show()
    plt.savefig('gsv_X_test_confusionMatrix.png')
    plt.clf()

    ########### KMeans ##############
    k_candidates = range(1, 8)
    inertias = []
    for k in k_candidates:
        k_means = KMeans(random_state=42, n_clusters=k, init='k-means++')
        k_means.fit(X_train)
        inertias.append(k_means.inertia_)
    plt.plot(k_candidates, inertias)
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.savefig('results/kmeans')
    plt.clf()

    kmeans = KMeans(n_clusters=6, init='k-means++', random_state=0).fit(X_train)
    labels = kmeans.labels_

    scores = f"""
        silhouette_score {silhouette_score(X_train, labels)},
        calinski_harabasz_score{calinski_harabasz_score(X_train, labels)},
        davies_bouldin_score{davies_bouldin_score(X_train, labels)}"""
    with open('results/Kmeans_scores.txt', 'w') as f:
        f.writelines(scores)
    f.close()

    # clusters
    X_cop = X_train.copy()
    X_cop['clusters'] = pd.Series(labels, index=X_cop.index)
    variances = []
    for i in range(6):
        temp = X_cop[X_cop['clusters'] == i]
        temp = temp.iloc[:, 0:6].var().tolist()
        print(temp)
        variances.append(temp)
    variances.pop(1)
    variances.pop(1)
    max_fac = ['stars_count',
     'commits_count',
     'contributors_top_avg_deletions',
     'commits_avg_per_day_real',
     'issues_open',
     'commits_total_lines_added']
    var_df = pd.DataFrame(variances, columns=max_fac, index=['Cluster-0', 'Cluster-3', 'Cluster-4', 'Cluster-5'])
    var_df.to_csv('results/clusters.csv')

