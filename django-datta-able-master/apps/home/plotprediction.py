# %%
"""
# **Arba Minch Tourism Prediction Challenge**
The dataset describes 6476 rows of up-to-date information on tourist expenditure collected by the National Bureau of Statistics (NBS) in Arba Minch Ethiopia.The dataset was collected to gain a better understanding of the status of the tourism sector and provide an instrument that will enable sector growth.

The objective of this hackathon is to develop a machine learning model to predict what a tourist will spend when visiting Arba Minch.The model can be used by different tour operators to automatically help tourists across the world estimate their expenditure before visiting Arba Minch.

Below is the third winning solution to the hackathon.
"""

# %%
# Importing the necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.utils import resample
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# loading our datasets
train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")
sub_df = pd.read_csv("SampleSubmission.csv")
descp = pd.read_csv("VariableDefinitions.csv")

# %%
"""
# Exploratory Data Analysis(EDA)
Here we try to understand our dataset in order to obtain insights from it so as to aid us while feature engineering our dataset
"""

# %%
# checking the first five rows of our train dataset
train_df.head(100)


# %%
train_df.count()


# %%
# same case for our test dataset
test_df.head()


# %%
test_df.count()


# %%
# viewing the size of both our dataset
print(train_df.shape)
print(test_df.shape)

# %%
# checking if there are any missing values in our dataset
print(train_df.isnull().sum())

# %%
# same case for our test dataset
print(test_df.isnull().sum())

# %%
train_df.info()

# %%
test_df.info()

# %%
# combining both our train and test dataset to have one that we can clean both datasets
data = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
data.columns.tolist()

# %%
# checking for missing values in our new dataset
data.isnull().sum()

# %%
# Handling the missing values by filling 'None' throughout all instances
data.travel_with.fillna('None', inplace=True)
data.most_impressing.fillna('None', inplace=True)
# Handling the missing values by filling through the mean value of the entire column
data.total_female.fillna(data.total_female.mean(), inplace=True)
data.total_male.fillna(data.total_male.mean(), inplace=True)

# %%
data.isnull().sum()

# %%
"""
# Feature Engineering
Here we try to come up with new features from existing ones in order to improve on the training process of our dataset

"""

# %%
"""
in our first feature we try to point out wether the tourist travelled into Tanzania durring peak periods of the year such as holiday months. durring this times, tourism services are relatively expensive due to high demand.

The column on purpose of visit aids us in doing so. 
"""

# %%
peak_period = []

for row in data.purpose:
    if row == 'Leisure and Holidays':
        peak_period.append(True)
    else:
        peak_period.append(False)

data['peak_period'] = peak_period

# %%
"""
In our second feature, we try and isolate tourists from local African countries and those from other continents.
In the tourism sector, Prices for local tourists and those for international differ with great margin
"""

# %%
# we obtain a list on all countries to aid us in isolating african countries
data['country'].unique()

# %%
is_African = []
african = ['SOUTH AFRICA', 'NIGERIA', 'RWANDA', 'MOZAMBIQUE', 'KENYA', 'ALGERIA', 'EGYPT', 'MALAWI',
           'UGANDA', 'ZIMBABWE', 'ZAMBIA', 'CONGO', 'MAURITIUS', 'DRC', 'TUNISIA', 'ETHIOPIA', 'BURUNDI',
           'GHANA', 'NIGER', 'COMORO', 'ANGOLA', 'SUDAN', 'NAMIBIA', 'LESOTHO', 'IVORY COAST', 'MADAGASCAR',
           'DJIBOUT', 'MORROCO', 'BOTSWANA', 'LIBERIA', 'GUINEA', 'SOMALI']

for country in data['country']:
    if country in african:
        is_African.append(True)
    else:
        is_African.append(False)

data['is_african'] = is_African

# %%
# Here we get two more features of the total number of people and the total number of nights spent
data["total_persons"] = data["total_female"] + data["total_male"]

data["total_nights_spent"] = data["night_Arba_minch"] + data["night_Gamo_Gofa"]
# data["cost_per_person"] = data['total_cost']/data['total_persons']


# %%
"""
**Label Encoding**

We now do some encoding techniques where by we perform label encoding first.
Label encoding involves converting labels into numeric form so as to have it into a machine-readable form.
"""

# %%
le = LabelEncoder()
data['age_group'] = le.fit_transform(data['age_group'])
data['package_transport_int'] = le.fit_transform(data['package_transport_int'])
data['package_accomodation'] = le.fit_transform(data['package_accomodation'])
data['package_food'] = le.fit_transform(data['package_food'])
data['package_transport_tz'] = le.fit_transform(data['package_transport_tz'])
data['package_sightseeing'] = le.fit_transform(data['package_sightseeing'])
data['package_guided_tour'] = le.fit_transform(data['package_guided_tour'])
data['package_insurance'] = le.fit_transform(data['package_insurance'])
data['first_trip_tz'] = le.fit_transform(data['first_trip_tz'])
data['country'] = le.fit_transform(data['country'])
data['peak_period'] = le.fit_transform(data['peak_period'])
data['is_african'] = le.fit_transform(data['is_african'])


# %%
columns_to_transform = ['tour_arrangement', 'travel_with', 'purpose',
                        'main_activity', 'info_source', 'most_impressing', 'payment_mode']
data = pd.get_dummies(data, columns=columns_to_transform, drop_first=True)

# %%
data.head(5)

# %%


# %%
data.to_csv("encoded_data.csv")


# %%
data.info()

# %%
# convert float dtypes to int
data["total_female"] = data['total_female'].astype('int')
data["total_male"] = data['total_male'].astype('int')
data["night_Arba_minch"] = data['night_Arba_minch'].astype('int')
data["night_Gamo_Gofa"] = data['night_Gamo_Gofa'].astype('int')

# %%
# separate data into train and test
train_df = data[data.total_cost.notnull()].reset_index(drop=True)
test_df = data[data.total_cost.isna()].reset_index(drop=True)

# %%
print(train_df.shape)
print(test_df.shape)

# %%
"""
# Modelling
Here is where we create a model that we'll train on with our training set so as to aid us in making our desired predictions.
we notice that we our hackathon is a linear regression problem hence we'll need to use a regression algorithm to solve it.
we used a catboost Regressor model to train and give our prediction.

"""

# %%

feat_cols = train_df.drop(["ID", "total_cost"], 1)
cols = feat_cols.columns
target = train_df["total_cost"]

# %%

# %%
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(
    train_df[cols], target, test_size=0.25, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %%
"""
checking if there are duplicated files
"""

# %%
data.duplicated().sum()


# %%
plt.figure(figsize=(20, 10), facecolor='w')
sns.boxplot(data=data)
plt.show()


# %%
"""
The above graph shows we have a stable data. Except for Total Cost but It is meant to be that way so we will not remove it
"""

# %%
data['total_cost'].max()


# %%
data['total_cost'].min()


# %%


# %%
cor = data.corr()
plt.figure(figsize=(20, 10), facecolor='w')
sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns, annot=True)
plt.title("Correlation among all the Variables of the Dataset", size=20)
cor


# %%
"""
This graph shows the relation between all columns to eachother
"""

# %%
data.info()

# %%
categorical_features = ['total_cost', 'age_group',
                        'night_Arba_minch', 'night_Gamo_Gofa', 'total_persons']

for feature in categorical_features:
    print(feature, ':')
    print(data[feature].value_counts())
    print("-----------------")


# %%
num_plots = len(categorical_features)
total_cols = 2
total_rows = num_plots//total_cols + 1
fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                        figsize=(7*total_cols, 7*total_rows), facecolor='w', constrained_layout=True)
for i, var in enumerate(categorical_features):
    row = i//total_cols
    pos = i % total_cols
    plot = sns.countplot(x=var, data=data, ax=axs[row][pos])


# %%
"""

"""

# %%
numeric_features = ['total_cost', 'age_group',
                    'night_Arba_minch', 'night_Gamo_Gofa', 'total_persons']
for feature in numeric_features:
    plt.figure(figsize=(18, 10), facecolor='w')
    sns.distplot(data[feature])
    plt.title('{} Distribution'.format(feature), fontsize=20)
    plt.show()


# %%
num_plots = len(numeric_features)
total_cols = 2
total_rows = num_plots//total_cols + 1
color = ['m', 'g', 'b', 'r', 'y', 'v', 'o']
fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                        figsize=(7*total_cols, 7*total_rows), facecolor='w', constrained_layout=True)
for i, var in enumerate(numeric_features):
    row = i//total_cols
    pos = i % total_cols
    plot = sns.violinplot(y=var, data=data, ax=axs[row][pos], linewidth=2)


# %%
graph_1 = data.groupby("is_african", as_index=False).total_cost.mean()


# %%
plt.figure(figsize=(12, 8), facecolor='w')
sns.regplot(x=graph_1["is_african"], y=graph_1["total_cost"])
plt.title("Graph showing cash spending of Africans.", size=20)
plt.xlabel("is_african", size=20)
plt.ylabel("total_cost", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# %%
"""
Graph shows Africans don't spend that much money
"""

# %%
graph_2 = data.groupby("total_male", as_index=False).age_group.sum()


# %%

plt.figure(figsize=(12, 8), facecolor='w')
sns.barplot(x=graph_2["total_male"], y=graph_2["age_group"])
plt.title(
    "males and their age group", size=20)
plt.ylabel("Gender\n male", size=20)
plt.xlabel("age_group", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# %%
graph_3 = data.groupby("total_female", as_index=False).age_group.sum()


# %%
plt.figure(figsize=(12, 8), facecolor='w')
sns.barplot(x=graph_3["total_female"], y=graph_3["age_group"])
plt.title(
    "Females and their age group", size=20)
plt.ylabel("Gender\n female ", size=20)
plt.xlabel("age_group", size=20)
plt.xticks(size=12)
plt.yticks(size=12)


# %%
data.info()

# %%
"""
Predictive Modeling
We use the following different machine learning models for the dataset:

1. Logistic Regressor
2. K-Nearest Neighbour Classifier
3. Random Forest Classifier
4. Decision Tree Classifier
5. Gradient Boosting Classifier
"""

# %%
data.isnull().sum()


# %%
data = pd.read_csv('cleaned_data.csv')


# %%
# sns.lmplot('total_female', 'total_male',
#            data=data,
#            hue="total_cost",
#            col="night_Arba_minch", row="night_Gamo_Gofa")
# plt.show()


# %%
# data.drop('SelectKBest', axis=1, inplace=True)
data = data.fillna(0)
X = data.iloc[:, 0:47]
y = data.iloc[:, -1]
print("X - ", X.shape, "\ny - ", y.shape)


# %%
# Data Feature Selection
# Data Splitting
# Data Scaling
# Data Modeling
# Hyperparameter Tuning
# Ensembling


# %%

# Apply SelectKBest and extract top 10 features
best = SelectKBest(score_func=chi2, k=10)


# %%
fit = best.fit(X, y)

data_scores = pd.DataFrame(fit.scores_)
data_columns = pd.DataFrame(X.columns)


# %%

# Join the two dataframes
scores = pd.concat([data_columns, data_scores], axis=1)
scores.columns = ['Feature', 'Score']
print(scores.nlargest(20, 'Score'))


# %%
data = data[['total_cost', 'main_activity_Diving and Sport Fishing', 'night_Arba_minch', 'package_insurance',
             'total_nights_spent', 'night_Gamo_Gofa', 'main_activity_Bird watching', 'info_source_Trade fair', 'purpose_Scientific and Academic',
             'purpose_Volunteering', 'is_african', 'main_activity_Hunting tourism', 'country', 'total_female', 'main_activity_Wildlife tourism',
             'package_guided_tour', 'main_activity_Cultural tourism', 'package_food']]
data.head()


# %%
y = data['main_activity_Cultural tourism']
X = data.drop(['main_activity_Cultural tourism'], axis=1)
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.4, random_state=1)


# %%
scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# %%
m1 = 'LogisticRegression'
lr = LogisticRegression(random_state=1, max_iter=1000)
model = lr.fit(train_x, train_y)
lr_predict = lr.predict(test_x)
lr_conf_matrix = confusion_matrix(test_y, lr_predict)
lr_acc_score = accuracy_score(test_y, lr_predict)
print("confussion matrix")
print(lr_conf_matrix)
print("\n")
print("Accuracy of Logistic Regression:", lr_acc_score*100, '\n')
print(classification_report(test_y, lr_predict))

# %%
m2 = 'KNeighborsClassifier'
knn = KNeighborsClassifier(n_neighbors=1)
model = knn.fit(train_x, train_y)
knn_predict = knn.predict(test_x)
knn_conf_matrix = confusion_matrix(test_y, knn_predict)
knn_acc_score = accuracy_score(test_y, knn_predict)
print("confussion matrix")
print(knn_conf_matrix)
print("\n")
print("Accuracy of k-NN Classification:", knn_acc_score*100, '\n')
print(classification_report(test_y, knn_predict))


# %%
m3 = 'Random Forest Classfier'
rf = RandomForestClassifier(n_estimators=200, random_state=0, max_depth=12)
rf.fit(train_x, train_y)
rf_predicted = rf.predict(test_x)
rf_conf_matrix = confusion_matrix(test_y, rf_predicted)
rf_acc_score = accuracy_score(test_y, rf_predicted)
print("confussion matrix")
print(rf_conf_matrix)
print("\n")
print("Accuracy of Random Forest:", rf_acc_score*100, '\n')
print(classification_report(test_y, rf_predicted))

# %%
m4 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=30)
dt.fit(train_x, train_y)
dt_predicted = dt.predict(test_x)
dt_conf_matrix = confusion_matrix(test_y, dt_predicted)
dt_acc_score = accuracy_score(test_y, dt_predicted)
print("confussion matrix")
print(dt_conf_matrix)
print("\n")
print("Accuracy of DecisionTreeClassifier:", dt_acc_score*100, '\n')
print(classification_report(test_y, dt_predicted))

# %%

m5 = 'Gradient Boosting Classifier'
gvc = GradientBoostingClassifier()
gvc.fit(train_x, train_y)
gvc_predicted = gvc.predict(test_x)
gvc_conf_matrix = confusion_matrix(test_y, gvc_predicted)
gvc_acc_score = accuracy_score(test_y, gvc_predicted)
print("confussion matrix")
print(gvc_conf_matrix)
print("\n")
print("Accuracy of Gradient Boosting Classifier:", gvc_acc_score*100, '\n')
print(classification_report(test_y, gvc_predicted))

# %%
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# %%
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=10,
                               cv=3,
                               verbose=2,
                               random_state=7,
                               n_jobs=-1)

# Fit the random search model
rf_random.fit(train_x, train_y)

# %%
rf_hyper = rf_random.best_estimator_
rf_hyper.fit(train_x, train_y)
print("Accuracy on training set is : {}".format(
    rf_hyper.score(train_x, train_y)))
print("Accuracy on validation set is : {}".format(
    rf_hyper.score(test_x, test_y)))
rf_predicted = rf_hyper.predict(test_x)
rf_acc_score = accuracy_score(test_y, rf_predicted)
print("Accuracy of Hyper-tuned Random Forest Classifier:", rf_acc_score*100, '\n')
print(classification_report(test_y, rf_predicted))


# %%
n_estimators = [int(i) for i in np.linspace(start=100, stop=1000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(i) for i in np.linspace(10, 100, num=10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# %%
gb = GradientBoostingClassifier(random_state=0)
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations
gb_random = RandomizedSearchCV(estimator=gb, param_distributions=random_grid,
                               n_iter=10, scoring='f1',
                               cv=3, verbose=2, random_state=0, n_jobs=-1,
                               return_train_score=True)

# Fit the random search model
gb_random.fit(train_x, train_y)

# %%
gb_hyper = gb_random.best_estimator_
gb_hyper.fit(train_x, train_y)
print("Accuracy on training set is : {}".format(
    gb_hyper.score(train_x, train_y)))
print("Accuracy on validation set is : {}".format(
    gb_hyper.score(test_x, test_y)))
gbc_predicted = gb_hyper.predict(test_x)
gbc_acc_score = accuracy_score(test_y, gbc_predicted)
print("Accuracy of Hyper-tuned Gradient Boosting Classifier:",
      gbc_acc_score*100, '\n')
print(classification_report(test_y, gbc_predicted))


# %%
lr_false_positive_rate, lr_true_positive_rate, lr_threshold = roc_curve(
    test_y, lr_predict)
knn_false_positive_rate, knn_true_positive_rate, knn_threshold = roc_curve(
    test_y, knn_predict)
rf_false_positive_rate, rf_true_positive_rate, rf_threshold = roc_curve(
    test_y, rf_predicted)
dt_false_positive_rate, dt_true_positive_rate, dt_threshold = roc_curve(
    test_y, dt_predicted)
gbc_false_positive_rate, gbc_true_positive_rate, gbc_threshold = roc_curve(
    test_y, gbc_predicted)


sns.set_style('whitegrid')
plt.figure(figsize=(15, 8), facecolor='w')
plt.title('Reciever Operating Characterstic Curve')
plt.plot(lr_false_positive_rate, lr_true_positive_rate,
         label='Logistic Regression')
plt.plot(knn_false_positive_rate, knn_true_positive_rate,
         label='K-Nearest Neighbor')
plt.plot(rf_false_positive_rate, rf_true_positive_rate, label='Random Forest')
plt.plot(dt_false_positive_rate, dt_true_positive_rate, label='Desion Tree')
plt.plot(gbc_false_positive_rate, gbc_true_positive_rate,
         label='Gradient Boosting Classifier')
plt.plot([0, 1], ls='--')
plt.plot([0, 0], [1, 0], c='.5')
plt.plot([1, 1], c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()


# %%
model_ev = pd.DataFrame({'Model': ['Logistic Regression', 'K-Nearest Neighbour', 'Random Forest',
                                   'Decision Tree', 'Gradient Boosting'], 'Accuracy': [lr_acc_score*100, knn_acc_score*100,
                                                                                       rf_acc_score*100, dt_acc_score*100, gbc_acc_score*100]})
model_ev


# %%
colors = ['red', 'green', 'blue', 'gold', 'silver']
plt.figure(figsize=(15, 8), facecolor='w')
plt.title("Barplot Representing Accuracy of different models")
plt.ylabel("Accuracy %")
plt.xlabel("Models")
plt.bar(model_ev['Model'], model_ev['Accuracy'], color=colors)
plt.show()
