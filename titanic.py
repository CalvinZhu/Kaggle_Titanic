#coding=utf-8
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"train_sample")
        plt.ylabel(u"score")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train_score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"cross_validation_score")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()
        # plt.savefig("learn_curve.jpg")

def training():
    data_train = pd.read_csv("train.csv")
    data_test = pd.read_csv("test.csv")
    # data_train.info()

    #Cabin数据缺失四分之三,因此将该项是否缺失作为特征条件
    data_train.loc[(data_train.Cabin.notnull()),'Cabin'] = 1
    data_train.loc[(data_train.Cabin.isnull()),'Cabin'] = 0

    #年龄缺失，通过分类的年龄平均数作为补充
    #按称谓计算各层平均年龄
    Mr_age_mean = (data_train[data_train.Name.str.contains('Mr.')]['Age'].mean())
    Mrs_age_mean = (data_train[data_train.Name.str.contains('Mrs.')]['Age'].mean())
    Miss_age_mean = (data_train[data_train.Name.str.contains('Miss.')]['Age'].mean())
    Master_age_mean = (data_train[data_train.Name.str.contains('Master.')]['Age'].mean())

    #以平均值填充缺失年龄
    data_train.loc[(data_train['Name'].str.contains('Dr.')) & data_train.Age.isnull(),'Age'] = Mr_age_mean
    data_train.loc[(data_train['Name'].str.contains('Mr.')) & data_train.Age.isnull(),'Age'] = Mr_age_mean
    data_train.loc[(data_train['Name'].str.contains('Mrs.')) & data_train.Age.isnull(),'Age'] = Mrs_age_mean
    data_train.loc[(data_train['Name'].str.contains('Miss.')) & data_train.Age.isnull(),'Age'] = Miss_age_mean
    data_train.loc[(data_train['Name'].str.contains('Master.')) & data_train.Age.isnull(),'Age'] = Master_age_mean

    #Age、Fare的连续值采用分段处理
    data_train['Fare'][data_train.Fare <= 7.91] = 0
    data_train['Fare'][(data_train.Fare > 7.91) & (data_train.Fare <= 14.454)] = 1
    data_train['Fare'][(data_train.Fare > 14.454) & (data_train.Fare <= 31)] = 2
    data_train['Fare'][data_train.Fare > 31] = 3

    data_train['Age'][data_train.Age <= 16] = 0
    data_train['Age'][(data_train.Age > 16) & (data_train.Age <= 32)] = 1
    data_train['Age'][(data_train.Age > 32) & (data_train.Age <= 48)] = 2
    data_train['Age'][(data_train.Age > 48) & (data_train.Age <= 64)] = 3
    data_train['Age'][data_train.Age > 64] = 4

    #特征数据离散化,one-hot编码，定性数据转成定量
    dummies_Age = pd.get_dummies(data_train['Age'], prefix='Age')
    dummies_Fare = pd.get_dummies(data_train['Fare'], prefix='Fare')
    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
    # dummies_SibSp = pd.get_dummies(data_train['SibSp'], prefix='SibSp')
    # dummies_Parch = pd.get_dummies(data_train['Parch'], prefix='Parch')

    # df = pd.concat([data_train, dummies_Age, dummies_Fare, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,
    #                 dummies_SibSp, dummies_Parch])
    # df.drop(['Age', 'Fare', 'Cabin', 'Embarked', 'Sex', 'Pclass', 'SibSp', 'Parch'])

    #挖掘隐藏特征
    #特定称谓
    data_train['Title'] = data_train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    data_train['Title'][data_train.Title == 'Capt.'] = 'Mr.'
    data_train['Title'][data_train.Title == 'Don.'] = 'Mr.'
    data_train['Title'][data_train.Title == 'Jonkheer.'] = 'Mr.'
    data_train['Title'][data_train.Title == 'Lady.'] = 'Mrs.'
    data_train['Title'][data_train.Title == 'Major.'] = 'Master.'
    data_train['Title'][data_train.Title == 'Mlle.'] = 'Mrs.'
    data_train['Title'][data_train.Title == 'Mme.'] = 'Mrs.'
    data_train['Title'][data_train.Title == 'Sir.'] = 'Mr.'
    data_train['Title'][data_train.Title == 'the'] = 'Mrs.'

    #是否独身
    data_train['isalone'] = np.nan
    data_train['isalone'][data_train.SibSp + data_train.Parch == 0] = 1
    data_train['isalone'][data_train.isalone.isnull()] = 0

    #是否母亲
    data_train['mother'] = np.nan
    data_train['mother'][(data_train.Parch > 0) & (data_train.Sex == 'female')] = 1
    data_train['mother'][data_train.isalone.isnull()] = 0

    #是否有家庭
    data_train['family'] = np.nan
    data_train['family'][(data_train.SibSp + data_train.Parch == 0)] = 0
    data_train['family'][(data_train.SibSp + data_train.Parch > 0) & (data_train.SibSp + data_train.Parch <=3)] = 1
    data_train['family'][data_train.family.isnull()] = 2

    #是否小孩
    data_train['person'] = np.nan
    data_train['person'][data_train.Age <= 16] = 0
    data_train['person'][(data_train.Age > 16) & (data_train.Sex == 'female')] = 1
    data_train['person'][(data_train.Age > 16) & (data_train.Sex == 'male')] = 2

    #是否相同票
    data_train['Ticket_same'] = np.nan
    data_train['Ticket_same'] = data_train['Ticket'].duplicated()
    data_train['Ticket_same'][data_train.Ticket_same == True] = 1
    data_train['Ticket_same'][data_train.Ticket_same == False] = 0

    #家庭成员数量
    data_train['family_size'] = data_train['Parch'] + data_train['SibSp']

    dummies_Title = pd.get_dummies(data_train['Title'], prefix='Title')
    dummies_isalone = pd.get_dummies(data_train['isalone'], prefix='isalone')
    dummies_mother = pd.get_dummies(data_train['mother'], prefix='mother')
    dummies_person = pd.get_dummies(data_train['person'], prefix='person')

    #建模
    #特征重要性评估，卡方监测，选出最重要的几个特征，如果出现欠拟合的情况，再在基于这几个重要特种组合特征分析
    # predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare", "family", "isalone", "Ticket_same", "Sex", "Title",
    #               "Embarked", "person", "mother", "Cabin"]
    # selector = SelectKBest(chi2, k=14)
    # a = selector.fit(data_train[predictors],data_train['Survived'])
    # print(np.array(a.scores_),'\n',a.get_support())
    # ax = sns.barplot(x=predictors, y=np.array(a.scores_),ci=0)
    # plt.show()

    #建模训练得到模型，逻辑回归
    # df = pd.concat([data_train, dummies_Age, dummies_Fare, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,
    #                 dummies_SibSp, dummies_Parch], axis=1)
    df = pd.concat([data_train, dummies_Age, dummies_Fare, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,
                    dummies_Title, dummies_isalone, dummies_mother, dummies_person], axis=1)
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*|family_.*|isalone_.*'
                              '|person_.*|Ticket_.*|mother_.*|Name_.*|family_size')

    train_np = train_df.as_matrix()
    y = train_np[:, 0]
    x = train_np[:, 1:]
    logic_clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    logic_clf.fit(x,y)

    #交叉验证
    all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*'
                               '|Title_.*|family_.*|isalone_.*|person_.*|Ticket_.*|mother_.*|Name_.*|family_size')
    print(all_data.shape)

    x = all_data.as_matrix()[:, 1:]
    y = all_data.as_matrix()[:, 0]
    logic_clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    print(cross_val_score(logic_clf, x, y, cv=5))
    # plot_learning_curve(clf, "learning curve", x, y)

    #模型组合，Voting
    logic_clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(40,20,10),max_iter=3000, random_state=1, activation='logistic')
    rf_test = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_features='sqrt',max_depth=10,
                                     min_samples_split=10,min_samples_leaf=2,n_jobs=50,random_state=42,verbose=1,
                                     min_weight_fraction_leaf=0.0,oob_score=True)
    clf = VotingClassifier(estimators=[('rf',logic_clf),('gbm',mlp_clf),('et',rf_test)],voting='soft',n_jobs=-1)
    clf.fit(x,y)
    return clf
    # bagging_clf = BaggingRegressor(logic_clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
    # bagging_clf.fit(x, y)
    # return bagging_clf

def feature_test():
    data_test = pd.read_csv("test.csv")
    #Cabin数据缺失四分之三,因此将该项是否缺失作为特征条件
    data_test.loc[(data_test.Cabin.notnull()),'Cabin'] = 1
    data_test.loc[(data_test.Cabin.isnull()),'Cabin'] = 0

    #年龄缺失，通过分类的年龄平均数作为补充
    #按称谓计算各层平均年龄
    Mr_age_mean = (data_test[data_test.Name.str.contains('Mr.')]['Age'].mean())
    Mrs_age_mean = (data_test[data_test.Name.str.contains('Mrs.')]['Age'].mean())
    Miss_age_mean = (data_test[data_test.Name.str.contains('Miss.')]['Age'].mean())
    Master_age_mean = (data_test[data_test.Name.str.contains('Master.')]['Age'].mean())

    #以平均值填充缺失年龄
    data_test.loc[(data_test['Name'].str.contains('Dr.')) & data_test.Age.isnull(),'Age'] = Mr_age_mean
    data_test.loc[(data_test['Name'].str.contains('Mr.')) & data_test.Age.isnull(),'Age'] = Mr_age_mean
    data_test.loc[(data_test['Name'].str.contains('Mrs.')) & data_test.Age.isnull(),'Age'] = Mrs_age_mean
    data_test.loc[(data_test['Name'].str.contains('Miss.')) & data_test.Age.isnull(),'Age'] = Miss_age_mean
    data_test.loc[(data_test['Name'].str.contains('Master.')) & data_test.Age.isnull(),'Age'] = Master_age_mean

    #Age、Fare的连续值采用分段处理
    data_test['Fare'][data_test.Fare <= 7.91] = 0
    data_test['Fare'][(data_test.Fare > 7.91) & (data_test.Fare <= 14.454)] = 1
    data_test['Fare'][(data_test.Fare > 14.454) & (data_test.Fare <= 31)] = 2
    data_test['Fare'][data_test.Fare > 31] = 3

    data_test['Age'][data_test.Age <= 16] = 0
    data_test['Age'][(data_test.Age > 16) & (data_test.Age <= 32)] = 1
    data_test['Age'][(data_test.Age > 32) & (data_test.Age <= 48)] = 2
    data_test['Age'][(data_test.Age > 48) & (data_test.Age <= 64)] = 3
    data_test['Age'][data_test.Age > 64] = 4

    #特征数据离散化,one-hot编码，定性数据转成定量
    dummies_Age = pd.get_dummies(data_test['Age'], prefix='Age')
    dummies_Fare = pd.get_dummies(data_test['Fare'], prefix='Fare')
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
    # dummies_SibSp = pd.get_dummies(data_test['SibSp'], prefix='SibSp')
    # dummies_Parch = pd.get_dummies(data_test['Parch'], prefix='Parch')

    #挖掘隐藏特征
    #特定称谓
    data_test['Title'] = data_test['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])


    #是否独身
    data_test['isalone'] = np.nan
    data_test['isalone'][data_test.SibSp + data_test.Parch == 0] = 1
    data_test['isalone'][data_test.isalone.isnull()] = 0

    #是否母亲
    data_test['mother'] = np.nan
    data_test['mother'][(data_test.Parch > 0) & (data_test.Sex == 'female')] = 1
    data_test['mother'][data_test.isalone.isnull()] = 0

    #是否有家庭
    data_test['family'] = np.nan
    data_test['family'][(data_test.SibSp + data_test.Parch == 0)] = 0
    data_test['family'][(data_test.SibSp + data_test.Parch > 0) & (data_test.SibSp + data_test.Parch <=3)] = 1
    data_test['family'][data_test.family.isnull()] = 2

    #是否小孩
    data_test['person'] = np.nan
    data_test['person'][data_test.Age <= 16] = 0
    data_test['person'][(data_test.Age > 16) & (data_test.Sex == 'female')] = 1
    data_test['person'][(data_test.Age > 16) & (data_test.Sex == 'male')] = 2

    #是否相同票
    data_test['Ticket_same'] = np.nan
    data_test['Ticket_same'] = data_test['Ticket'].duplicated()
    data_test['Ticket_same'][data_test.Ticket_same == True] = 1
    data_test['Ticket_same'][data_test.Ticket_same == False] = 0

    #家庭成员数量
    data_test['family_size'] = data_test['Parch'] + data_test['SibSp']

    dummies_Title = pd.get_dummies(data_test['Title'], prefix='Title')
    dummies_isalone = pd.get_dummies(data_test['isalone'], prefix='isalone')
    dummies_mother = pd.get_dummies(data_test['mother'], prefix='mother')
    dummies_person = pd.get_dummies(data_test['person'], prefix='person')

    # df = pd.concat([data_test, dummies_Age, dummies_Fare, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,
    #                 dummies_SibSp, dummies_Parch], axis=1)
    df = pd.concat([data_test, dummies_Age, dummies_Fare, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,
                    dummies_Title, dummies_isalone, dummies_mother, dummies_person], axis=1)
    test_df = df.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Title_.*|family_.*|isalone_.*'
                              '|person_.*|Ticket_.*|mother_.*|Name_.*|family_size')

    return test_df

if __name__ == '__main__':
    data_test = pd.read_csv("test.csv")
    train_clf = training()
    data_test_df = feature_test()
    predictions = train_clf.predict(data_test_df)
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv('logistic_regression_predictions.csv', index=False)
