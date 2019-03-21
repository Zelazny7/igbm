from constraints import Constrainer, Interval, Value, Missing
from sklearn.compose import ColumnTransformer
import seaborn as sns
import xgboost as xgb
import numpy as np
import matplotlib as plt

def partial_dependency(bst, X, y, tf, f_id = -1):

    """
    Calculate the dependency (or partial dependency) of a response variable on a predictor (or multiple predictors)
    1. Sample a grid of values of a predictor.
    2. For each value, replace every row of that predictor with this value, calculate the average prediction.
    """

    X_temp = X.copy()


    grid = np.linspace(np.percentile(X_temp.iloc[:, f_id], 0.1),
                       np.percentile(X_temp.iloc[:, f_id], 99.5),
                       50)

    grid = np.round(grid, 1)

    y_pred = np.zeros(len(grid))

    if f_id == -1:
        print ('Input error!')
        return
    else:
        for i, val in enumerate(grid):

            X_temp.iloc[:, f_id] = val
            x = tf.transform(X_temp)

            data = xgb.DMatrix(x)

            y_pred[i] = np.average(bst.predict(data))

    return grid, y_pred


if __name__ == '__main__':

    iris = sns.load_dataset("iris")

    c1 = Constrainer([
        Value(5, 1),
        Interval(0, 10, False, False, 1, 0)
    ])

    # print(c1)

    c2 = Constrainer([
        Interval(0, 10, False, False, -1, 0)
    ])

    #print(c1.fit_transform(iris.sepal_length.values.reshape(-1, 1)).shape)


    # pickle.dump(c1, open('test.dat', 'wb'))
    # c1 = pickle.load(open('test.dat', 'rb'))
    #
    # iris.columns
    #
    ct = ColumnTransformer(
        [("sepals", c1, ["sepal_length", "sepal_width"]),
         ("petals", c2, ["petal_length", "petal_width"])])

    ct = ColumnTransformer(
        [("sepals", c1, [0, 1]),
         ("petals", c2, [2, 3])])

    ct.fit(iris)

    def get_mono(ct: ColumnTransformer):
        mono = []
        for name, tf, cols in ct.transformers:
            for col in cols:
                mono += tf.mono
        return mono


    x = ct.fit_transform(iris)
    y = np.array(iris.species == "virginica", dtype=np.int)

    params = {
        "objective": "binary:logistic",
        "monotone_constraints": str(tuple(get_mono(ct))),
        "eta": 1,
        "max_depth": 2
    }

    dtrain = xgb.DMatrix(x, label=y)

    mod: xgb.Booster = xgb.train(params, dtrain, num_boost_round=100)

    # contribs = mod.predict(dtrain, pred_contribs=True)
    #
    # pdeps = []
    # import itertools
    # for cols in itertools.tee(range(0,8), 4):
    #     pdeps += [np.sum(contribs[:,list(cols)], axis=1)]


    #plt = sns.scatterplot(iris.iloc[:,0], pdeps[0])
    #plt.get_figure().savefig("test.png")

    grid, y_pred = partial_dependency(mod, iris, y, ct, f_id = 0)

    plt = sns.scatterplot(grid, y_pred)
    plt.get_figure().savefig("test.png")

    print(get_mono(ct))