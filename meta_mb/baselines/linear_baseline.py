from meta_mb.baselines.base import Baseline
from meta_mb.utils.serializable import Serializable
import numpy as np


class LinearBaseline(Baseline):
    """
    Abstract class providing the functionality for fitting a linear baseline
    Don't instantiate this class. Instead use LinearFeatureBaseline or LinearTimeBaseline
    """

    def __init__(self, reg_coeff=1e-5):
        super(LinearBaseline, self).__init__()
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def predict(self, path):
        """
        Abstract Class for the LinearFeatureBaseline and the LinearTimeBaseline
        Predicts the linear reward baselines estimates for a provided trajectory / path.
        If the baseline is not fitted - returns zero baseline

        Args:
           path (dict): dict of lists/numpy array containing trajectory / path information
                 such as "observations", "rewards", ...

        Returns:
             (np.ndarray): numpy array of the same length as paths["observations"] specifying the reward baseline

        """
        if self._coeffs is None:
            return np.zeros(len(path["observations"]))
<<<<<<< HEAD:hw5/baselines/linear_baseline.py
        '''  YOUR CODE HERE FOR PROBLEM 1B'''
        # hint baselines should be features dot coeffs

        prediction = np.dot(self._features(path),self._coeffs)
        '''  YOUR CODE ENDS'''
        return prediction
=======
        return self._features(path).dot(self._coeffs)
>>>>>>> parent of 053b127... first comit with 1a:meta_mb/baselines/linear_baseline.py

    def get_param_values(self, **tags):
        """
        Returns the parameter values of the baseline object

        Returns:
            numpy array of linear_regression coefficients

        """
        return self._coeffs

    def set_params(self, value, **tags):
        """
        Sets the parameter values of the baseline object

        Args:
            value: numpy array of linear_regression coefficients

        """
        self._coeffs = value

    def fit(self, paths, target_key='returns'):
        """
        Fits the linear baseline model with the provided paths via damped least squares

        Args:
            paths (list): list of paths
            target_key (str): path dictionary key of the target that shall be fitted (e.g. "returns")

        """
        assert all([target_key in path.keys() for path in paths])
<<<<<<< HEAD:hw5/baselines/linear_baseline.py
        """ YOUR CODE HERE FOR PROBLEM 1B """
        # hint: 1. convert your paths to concatenated features use the function self._features
        # 2. your target_key is returns, thats the target you want to fit, path[target_key] where path is an element of paths
        # 3. you need regularization coefficient to avoid NAN (provided)
        # 4. Use np.linalg.lstsq to find the best coeff for featmat.T.dot(featmat), featmat.T.dot(target), (provided)
        # 5. don't forget your reg_coeff with an identity matrix sould be add to featmat.T.dot(featmat) as well (provided)
        # 6. You can do this your own way!!

        featmat = np.concatenate([self._features(p) for p in paths])
        target = np.concatenate([path[target_key] for path in paths])
=======

        featmat = np.concatenate([self._features(path) for path in paths], axis=0)
        target = np.concatenate([path[target_key] for path in paths], axis=0)
>>>>>>> parent of 053b127... first comit with 1a:meta_mb/baselines/linear_baseline.py
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(target),
                rcond=-1
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def _features(self, path):
        raise NotImplementedError("this is an abstract class, use either LinearFeatureBaseline or LinearTimeBaseline")


class LinearFeatureBaseline(LinearBaseline):
    """
    Linear (polynomial) time-state dependent return baseline model
    (see. Duan et al. 2016, "Benchmarking Deep Reinforcement Learning for Continuous Control", ICML)

    Fits the following linear model

    reward = b0 + b1*obs + b2*obs^2 + b3*t + b4*t^2+  b5*t^3

    Args:
        reg_coeff: list of paths

    """
    def __init__(self, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def _features(self, path):
        obs = np.clip(path["observations"], -10, 10)
        path_length = len(path["observations"])
        time_step = np.arange(path_length).reshape(-1, 1) / 100.0
        return np.concatenate([obs, obs ** 2, time_step, time_step ** 2, time_step ** 3, np.ones((path_length, 1))],
                              axis=1)


class LinearTimeBaseline(LinearBaseline):
    """
    Linear (polynomial) time-dependent reward baseline model

    Fits the following linear model

    reward = b0 + b3*t + b4*t^2+  b5*t^3

    Args:
        reg_coeff: list of paths

    """

    def _features(self, path):
        path_length = len(path["observations"])
        time_step = np.arange(path_length).reshape(-1, 1) / 100.0
        return np.concatenate([time_step, time_step ** 2, time_step ** 3, np.ones((path_length, 1))],
                              axis=1)

