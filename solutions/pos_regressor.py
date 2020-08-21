from base import Regressor
import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import KernelPCA
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

class PositionRegressor(Regressor):

    """ Implement solution for Part 1 below  """

    def __init__(self):

        ada = AdaBoostClassifier(ExtraTreeClassifier(max_depth=9, random_state=0), n_estimators=50, learning_rate=0.1, random_state=0)
        self.model = make_pipeline(StandardScaler(), KernelPCA(n_components=100, kernel='rbf', random_state=0), ada)

        self.agent_pos = dict()

    def train(self, data):
        data_size = data['obs'].shape

        x_train = data['obs'].flatten().reshape(data_size[0], 12288)
        x_train = x_train / 255.0
        pos = np.asarray([info['agent_pos'] for info in data['info']])

        j = 0
        for i in range(len(pos)):
            if tuple(pos[i]) not in self.agent_pos:
                self.agent_pos[tuple(pos[i])] = j
                j = j + 1

        y_train = np.asarray([self.agent_pos[tuple(pos[i])] for i in range(len(pos))])

        self.model.fit(x_train, y_train)

    def predict(self, Xs):
        
        data_size = Xs.shape
        x_test = Xs.flatten().reshape(data_size[0], 12288)
        x_test = x_test / 255.0

        pred = self.model.predict(x_test)

        res = []
        for i in range(len(pred)):
            flag = 0
            for k, v in self.agent_pos.items():
                if round(pred[i]) == v:
                    res.append(list(k))
                    flag = flag + 1
                if flag == 1:
                    break

            if flag == 0:
                res.append(list((0, 0)))

        return res
