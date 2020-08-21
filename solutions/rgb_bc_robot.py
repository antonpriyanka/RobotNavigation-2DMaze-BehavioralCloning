from base import RobotPolicy
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC

class RGBBCRobot(RobotPolicy):
    """ Implement solution for Part3 below """

    def __init__(self):

        self.model = make_pipeline(StandardScaler(), KernelPCA(n_components=20, kernel='rbf', random_state=0),SVC())

    def train(self, data):
        y_train = data['actions']

        data_size = data['obs'].shape
        x_train = data['obs'].flatten().reshape(data_size[0], 12288)
        x_train = x_train / 255.0

        self.model.fit(x_train, y_train.ravel())

    def get_actions(self, observations):
        x_test = observations / 255.0
        actions = self.model.predict(x_test)
        return actions
