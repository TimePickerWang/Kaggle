import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_data():
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')

    # 丢弃特征
    train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # 性别转换为数值
    train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    test_df['Sex'] = test_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # 将SibSp和Parch组合成一个名为FamilySize的合成特征，它表示每个成员的船上家庭成员总数
    train_df["FamSize"] = train_df["SibSp"] + train_df["Parch"]
    test_df["FamSize"] = test_df["SibSp"] + test_df["Parch"]

    # 新增一个特征FamTag,当FamSize>3时，另其为0，小于等于3时，另其为1
    train_df["FamTag"] = train_df.FamSize.apply(lambda x: 1 if x <= 3 else 0)
    test_df["FamTag"] = test_df.FamSize.apply(lambda x: 1 if x <= 3 else 0)

    # 丢弃掉SibSp、Parch和FamSize这3个特征
    train_df = train_df.drop(['SibSp', 'Parch', 'FamSize'], axis=1)
    test_df = test_df.drop(['SibSp', 'Parch', 'FamSize'], axis=1)

    # 填充缺失值
    train_df['Embarked'] = train_df['Embarked'].fillna('S')
    test_df['Embarked'] = test_df['Embarked'].fillna('S')

    # 将特征Embarked的值转为数值型
    train_df['Embarked'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(int)
    test_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(int)

    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
    test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
    test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

    # 年龄'Age'和船票费用'Fare'都是连续值，为了方便处理，我们将这两个特征分别划分为8个、4个区间，
    # 划分的区间为新的特征，分别是'AgeBand'和'FareBand'
    train_df['AgeBand'] = pd.cut(train_df['Age'], 8)
    test_df['AgeBand'] = pd.cut(test_df['Age'], 8)
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    test_df['FareBand'] = pd.qcut(test_df['Fare'], 4)

    # 将'Age'和'Fare'分别用不同的值代替，将连续值离散化，然后丢弃新增的2个特征
    combine = [train_df, test_df]
    for dataset in combine:
        # Age
        dataset.loc[dataset['Age'] <= 10, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 20), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 60) & (dataset['Age'] <= 70), 'Age'] = 6
        dataset.loc[dataset['Age'] > 70, 'Age'] = 7
        # Fare
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3

        dataset['Age'] = dataset['Age'].astype(int)
        dataset['Fare'] = dataset['Fare'].astype(int)
        dataset.drop(['AgeBand', 'FareBand'], axis=1, inplace=True)

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    test_Id = test_df["PassengerId"]
    X_test = test_df.drop("PassengerId", axis=1).copy()
    return X_train, Y_train, X_test, test_Id


X_train, Y_train, X_test, test_Id = get_data()

# a.Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_a = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)


# b.Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_b = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# c.Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_c = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# d.Random Forest
random_forest = RandomForestClassifier(n_estimators=50)
random_forest.fit(X_train, Y_train)
Y_pred_d = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# e.Gradient Tree Boosting
gbdt = GradientBoostingClassifier(n_estimators=100,max_depth=8)
gbdt.fit(X_train, Y_train)
Y_pred_e = gbdt.predict(X_test)
gbdt.score(X_train, Y_train)
acc_gbdt = round(gbdt.score(X_train, Y_train) * 100, 2)

# 按准确率排序
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines',
              'Decision Tree', 'Random Forest', 'Gradient Tree Boosting'],
    'Score': [acc_log, acc_svc, acc_decision_tree, acc_random_forest, acc_gbdt]})
acc = models.sort_values(by='Score', ascending=False)
print(acc)

# 将预测结果写导csv文件中
submission = pd.DataFrame({
        "PassengerId": test_Id,
        "Survived": Y_pred_e
    })
submission.to_csv('./data/submission_test.csv', index=False)
