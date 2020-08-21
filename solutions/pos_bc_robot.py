from base import RobotPolicy
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC

class POSBCRobot(RobotPolicy):
    """ Implement solution for Part 2 below """

    def __init__(self):

        self.model = make_pipeline(StandardScaler(),KernelPCA(n_components=2,kernel='rbf'),SVC())

    def train(self, data):
        x_train = data['obs']
        y_train = data['actions']

        self.model.fit(x_train, y_train.ravel())

    def get_actions(self, observations):

        actions = self.model.predict(observations)

        return actions
