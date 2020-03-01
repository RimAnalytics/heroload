"""
This test script loads a toy dataset and models it with a few
shallow learning algorithms.

Requirements:
    pandas
    sklearn
    

"""


# Load pandas and sklearn modules
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
                             GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Ignore FutureWarning in sklearn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



def main():
    # Load full dataframe and get training / test sets
    full_df = pd.read_csv("./test_team_dataset.csv", parse_dates=["game_date"])
    full_df = full_df[full_df["game_date"] > pd.Timestamp("2017-10-01")].reset_index(drop=True)

    # Encode string types as ints
    le = LabelEncoder()

    location_arr = ['home', 'away']
    le.fit(location_arr)
    full_df['location'] = le.transform(full_df['location'])

    outcome_arr = ['win', 'loss']
    le.fit(outcome_arr)
    full_df['outcome'] = le.transform(full_df['outcome'])


    # Get training and test sets
    feat_list = full_df.columns
    feat_list = feat_list.drop(['game_date', 'team', 'opponent', 'opponent_score', 'outcome'])

    train_df = full_df[full_df['game_date'] < pd.Timestamp('2019-10-01')].drop('game_date', axis=1)
    test_df = full_df[full_df['game_date'] > pd.Timestamp('2019-10-01')].drop('game_date', axis=1)

    train_feat_df = train_df[feat_list]
    test_feat_df = test_df[feat_list]

    train_label_df = train_df['outcome']
    test_label_df = test_df['outcome']


    # Run classifiers and print results

    for clf in [KNeighborsClassifier(),\
                SVC(), LinearSVC(), NuSVC(),\
                DecisionTreeClassifier(), DecisionTreeRegressor(),\
                RandomForestClassifier(), AdaBoostClassifier(), \
                    GradientBoostingClassifier(),\
                GaussianNB(),\
                LinearDiscriminantAnalysis(),\
                QuadraticDiscriminantAnalysis(),\
                LogisticRegression()]:

        clf.fit(train_feat_df, train_label_df)
        label_predict = clf.predict(test_feat_df)

        accuracy = accuracy_score(test_label_df, label_predict)

        print("%30s" % clf.__class__.__name__, \
               " accuracy:", "%1.5f" % accuracy)


if __name__ == "__main__":
    main()
