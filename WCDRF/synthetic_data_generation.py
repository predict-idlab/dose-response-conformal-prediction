
from scipy.stats import norm, t, beta, uniform
from scipy.special import erfinv, gamma
import numpy as np
import pandas as pd
import warnings

class UniformDist:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def pdf(self, X):
        return np.array((X >= self.lower)&(X <= self.upper)) * (1/(self.upper - self.lower))
    
    def interval(self, confidence):
        p = confidence/2.0
        return self.lower + (self.upper - self.lower) * (0.5 - p), self.lower + (self.upper - self.lower) * (0.5 + p)


class CombinedUniform:
    def __init__(self, binomial_ratio, threshold, Uniform1, Uniform2):
        self.binomial_ratio = binomial_ratio
        self.Uniform1 = Uniform1
        self.Uniform2 = Uniform2
        self.threshold = threshold

    def pdf(self, T):
        return np.array(np.array(T) < self.threshold) * self.binomial_ratio * self.Uniform1.pdf(T) + (1 - np.array(np.array(T) < self.threshold)) * (1 - self.binomial_ratio) * self.Uniform2.pdf(T)

        
    def interval(self, confidence):
        return_arr = self.binomial_ratio * np.array(self.Uniform1.interval(confidence)) + (1-self.binomial_ratio) * np.array(self.Uniform1.interval(confidence))
        return return_arr[0], return_arr[1]


class NormalDist:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def pdf(self, X):
        return 1/np.sqrt(2 * np.pi * (self.scale**2)) * np.exp(-((X - self.loc)**2) / (2 * self.scale**2))
    
    def interval(self, confidence):
        p = confidence/2.0
        return self.loc + self.scale * np.sqrt(2) * erfinv(2*(0.5-p) - 1), self.loc + self.scale * np.sqrt(2) * erfinv(2*(0.5+p) - 1)

class StudentTDist:
    def __init__(self, df, loc):
        self.df = df
        self.loc = loc

    def pdf(self, X):
        return gamma((self.df + 1) / 2) / (np.sqrt(np.pi * self.df) * gamma(self.df / 2)) * (1 + ((X - self.loc)**2) / self.df) ** (- (self.df + 1) / 2)
    
    def interval(self, confidence):
        return t(df=self.df,loc=self.loc).interval(confidence)


class BetaDist:
    def __init__(self, alpha, beta, loc, scale):
        self.alpha = alpha
        self.beta = beta
        self.loc = loc
        self.scale = scale

    def pdf(self, X):
        return_arr = (((X - self.loc) / self.scale)** (self.alpha - 1)) * ( (1 - ((X - self.loc) / self.scale) ) ** (self.beta - 1)) / (gamma(self.alpha) * gamma(self.beta) / gamma(self.alpha + self.beta)) / self.scale
        # return_arr = (((X - self.loc) )** (self.alpha - 1)) * ( (1 - ((X - self.loc) ) ) ** (self.beta - 1)) / (gamma(self.alpha) * gamma(self.beta) / gamma(self.alpha + self.beta))
        # return (X ** (self.alpha - 1)) * ( (1 - (X) ) ** (self.beta - 1)) / (gamma(self.alpha) * gamma(self.beta) / gamma(self.alpha + self.beta))

        return np.array(return_arr > 0) * return_arr
    
    def interval(self, confidence):
        return beta(self.alpha,self.beta,self.loc,self.scale).interval(confidence)




class synthetic_data_generator:
    def __init__(self, source: int, scenario: int, interventional_mode: int = 0, interventional_delta: float = 0):
        """
        Initialize the class with the specified source and scenario.

        Parameters
        ----------
        source : int
            The source type, which should be either 0 or 1. If an invalid value is provided,
            a warning will be issued and the source will be set to 0.
        
        scenario : int
            The scenario number. Valid scenario numbers depend on the source:
            - If source is 0, valid scenario numbers are 1, 2, 3, 4, 5, 6, 7, 8.
            - If source is 1, valid scenario numbers are 1, 2.
            If an invalid scenario number is provided, a warning will be issued and the 
            scenario will be set to 1.

        Attributes
        ----------
        source : int
            The validated source type, set to either 0 or 1.
        
        scenario : int
            The validated scenario number.
        
        Y_std : float
            The standard deviation value determined by the source:
            - 5 if source is 0
            - 0.1 if source is 1

        interventional_mode : int
            The mode that is used to generate interventional data when source = 1. By default is zero

        Raises
        ------
        UserWarning
            If the provided source is invalid, a warning is issued and the source is set to 0.
        
        UserWarning
            If the provided scenario is invalid, a warning is issued and the scenario is set to 1.

        Examples
        --------
        >>> obj = MyClass(0, 3)
        >>> obj.source
        0
        >>> obj.scenario
        3
        >>> obj.Y_std
        5
        
        >>> obj = MyClass(1, 2)
        >>> obj.source
        1
        >>> obj.scenario
        2
        >>> obj.Y_std
        0.1

        >>> obj = MyClass(2, 5)
        UserWarning: source undefined, setting it to default 0
        >>> obj.source
        0
        >>> obj.scenario
        5
        >>> obj.Y_std
        5

        >>> obj = MyClass(1, 5)
        UserWarning: scenario number undefined, setting it to default 1
        >>> obj.source
        1
        >>> obj.scenario
        1
        >>> obj.Y_std
        0.1
        """
        if source < 0 or source > 2:
            warnings.warn("source undefined, setting it to default 0")
            source = 0

        self.source = source

        if (source == 0 and scenario not in [1, 2, 3, 4, 5, 6, 7, 8]) or (source == 1 and scenario not in [1, 2]) or (source == 2 and scenario not in [1, 2]):
            warnings.warn("scenario number undefined, setting it to default 1")
            scenario = 1

        self.scenario = scenario

        if source == 0:
            self.Y_std = 5
        elif source == 1:
            self.Y_std = 0.1

            if not (interventional_mode == 0 or interventional_mode in ["HARD","SOFT"]):
                warnings.warn("interventional_mode undefined, setting it to no intervention: default 0")
                interventional_mode = 0
            self.interventional_mode = interventional_mode

            if self.interventional_mode != 0:
                self.interventional_delta = interventional_delta

        elif source == 2:
            self.Y_std = 2

    # Synthetic data generation

    def get_cardinal_function(self, feature_array: np.ndarray):
        """
        Compute the cardinal function based on the current source and scenario.

        Parameters
        ----------
        feature_array : np.ndarray
            A 2D numpy array where each row represents a different data point and 
            each column corresponds to a feature.

        Returns
        -------
        np.ndarray or float
            The computed cardinal function values. If `self.source` is 0, it returns 
            a 1D array with the computed values based on the scenario. If `self.source`
            is 1, it returns the feature array multiplied by 5.

        Raises
        ------
        IndexError
            If the feature_array does not have at least 6 columns when `self.source` is 0.
        """

        if self.source == 0:
            C1 = feature_array[:, 0]
            C2 = feature_array[:, 1]
            C3 = feature_array[:, 2]
            C4 = feature_array[:, 3]
            C5 = feature_array[:, 4]
            C6 = feature_array[:, 5]


            if self.scenario == 3:
                # return -0.8 + 0.1*C1 + 0.1*C2 - 0.1*C3 + 0.2*C4 + 0.1*C5 + 0.1*C6
                return -0.8 + 1*C1 + 0.1*C2 - 0.1*C3 + 0.2*C4 + 0.1*C5 + 0.1*C6 + 3/2 * C3**2
            else:
                return -0.8 + 1*C1 + 0.1*C2 - 0.1*C3 + 0.2*C4 + 0.1*C5 + 0.1*C6
                # return 3 + 1*C1 + 0.1*C2 - 0.1*C3 + 0.2*C4 + 0.1*C5 + 0.1*C6

        elif self.source == 1:
            if self.interventional_mode == 0:
                return 5*feature_array
            
            elif self.interventional_mode == "SOFT":
                return 5*feature_array + self.interventional_delta
            
            elif self.interventional_mode == "HARD":
                return feature_array * self.interventional_delta
            
        elif self.source == 2:
            C1 = feature_array[:, 0]
            C2 = feature_array[:, 1]
            C3 = feature_array[:, 2]

            return C2 + 0.1 * C1

    def calculate_y_mean(self, feature_array: np.ndarray, treatment_array: np.ndarray):
        """
        Calculate the mean response variable `Y` based on the given feature and treatment arrays.

        Parameters
        ----------
        feature_array : np.ndarray
            A 2D numpy array where each row represents a different data point and 
            each column corresponds to a feature.
        
        treatment_array : np.ndarray
            A 1D numpy array where each element corresponds to the treatment applied 
            to the respective data point in `feature_array`.

        Returns
        -------
        np.ndarray
            A 1D array with the computed mean response variable `Y` for each data point.

        Raises
        ------
        IndexError
            If the feature_array does not have at least 6 columns when `self.source` is 0.

        Notes
        -----
        - For `self.source` == 0:
        The function computes:
        `-1 - (2*C1 + 2*C2 + 3*C3**3 - 20*C4 - 2*C5 + 20*C6) - (treatment_array) * (0.1 - 0.1*C1 + 0.1*C4 + 0.1*C5 + 0.1 * C3**2) + 0.13**2 * (np.abs(treatment_array))**3 * np.sin(C4)`
        - For `self.source` == 1:
        The function computes:
        `np.sin(np.pi/2 * (0.1*treatment_array - 0.1*feature_array))`
        """
        # Y_normal_mean = -1 - (2*C1 + 2*C2 + 3*C3**3 -1*C4 - 20*C5 + 20*C6) - (W) * (0.1 - 0.1*C1 + 0.1*C4 + 0.1*C5 + 0.1 * C3**2) + 0.13**2 * (W)**3
        # Y_normal_mean = -1 - (2*C1 + 2*C2 + 3*C3**3 -20*C4 - 2*C5 + 20*C6) - (W) * (0.1 - 0.1*C1 + 0.1*C4 + 0.1*C5 + 0.1 * C3**2) + 0.13**2 * (np.abs(W))**3 * np.sin(C4)
        # Y_normal_mean = -1 - (2*C1 + 2*C2 + 3*C3**3 -20*C4 - 2*C5 + 20*C6) - 2*(W) * (C4**3) + (0.13**2) * (W)**(2+np.abs(C5))

        # return -1 - (2*C1 + 2*C2 + 3*C3**3 -20*C4 - 2*C5 + 20*C6) - (W) * (0.1 - 0.1*C1 + 0.1*C4 + 0.1*C5 + 0.1 * C3**2) + 0.13**2 * (np.abs(W))**3 * np.sin(C4)
        # return -1 - (2*C1 + 2*C2 + 3*C3**3 -20*C4 - 2*C5 + 20*C6) - (W) * (0.1 - 0.1*C1 + 0.1*C4 + 0.1*C5 + 0.1 * C3**2) + 0.13**2 * (np.abs(W))**3 * np.sin(C4)

        if self.source == 0:
            C1 = feature_array[:, 0]
            C2 = feature_array[:, 1]
            C3 = feature_array[:, 2]
            C4 = feature_array[:, 3]
            C5 = feature_array[:, 4]
            C6 = feature_array[:, 5]

            return -1 - (2*C1 + 2*C2 + 3*C3**3 - 20*C4 - 2*C5 + 20*C6) - (treatment_array) * (0.1 - 0.1*C1 + 0.1*C4 + 0.1*C5 + 0.1 * C3**2) + 0.13**2 * (np.abs(treatment_array))**3 * np.sin(C4)

        elif self.source == 1:
            return np.sin(np.pi/2 * (0.1*treatment_array - 0.1*feature_array))
        
        elif self.source == 2:
            C1 = feature_array[:, 0]
            C2 = feature_array[:, 1]
            C3 = feature_array[:, 2]

            if self.scenario == 1:
                return (np.sign(C3)) * ((2 * (treatment_array - C2)) ** 2) + 33 * treatment_array * np.sign(C1)
            elif self.scenario == 2:
                return (np.sign(C3)) * ((2 * (treatment_array - C2)) ** 2) + 33 * treatment_array * np.sign(C1) + ((np.sign(C3) + 1) / 2.0) * np.random.normal(0, 30 , len(treatment_array))

    def get_treatment_scenario(self, cardinal_function_array: np.ndarray, return_type_distribution = False, N : int = None):
        """
        Calculate the treatment scenario values based on the cardinal function array.

        Parameters
        ----------
        cardinal_function_array : np.ndarray
            A 1D numpy array with the computed values from the cardinal function.
        
        return_type_distribution : bool, optional
            If True, returns the distribution object. If False, returns sampled values from 
            the distribution. Default is False.
        
        N : int, optional
            The number of samples to return if `return_type_distribution` is False. This parameter 
            is required in that case.

        Returns
        -------
        np.ndarray or scipy.stats.rv_continuous
            Depending on the `return_type_distribution` flag:
            - If True, returns the distribution object.
            - If False, returns a numpy array with sampled values.

        Raises
        ------
        ValueError
            If `N` is not provided when `return_type_distribution` is False.

        UserWarning
            If the return distribution is not supported for the specified scenario and source combination.

        Notes
        -----
        - For `self.source` == 0:
        - Scenario 1: 
            `W_mean = 9 * cardinal_function_array + 17`
            - Normal distribution with mean `W_mean` and std dev 5.
        - Scenario 2:
            `W_mean = 15 * cardinal_function_array + 22`
            - t-distribution with 2 degrees of freedom, location `W_mean`.
        - Scenario 3:
            `W_mean = 9 * cardinal_function_array + 15`
            - Normal distribution with mean `W_mean` and std dev 5.
        - Scenario 4:
            `W_mean = 49 * np.exp(cardinal_function_array) / (1 + np.exp(cardinal_function_array)) - 6`
            - Normal distribution with mean `W_mean` and std dev 5.
        - Scenario 5:
            `W_mean = 42 * 1 / (1 + np.exp(cardinal_function_array)) - 18`
            - Normal distribution with mean `W_mean` and std dev 5.
        - Scenario 6:
            `W_mean = 7 * np.log(np.abs(cardinal_function_array) + 0.001) + 13`
            - Normal distribution with mean `W_mean` and std dev 4.
        - Scenario 7:
            `W_mean = 7 * cardinal_function_array + 16`
            - Normal distribution with mean `W_mean` and std dev 1.
        - Scenario 8:
            `W_mean = 7 * cardinal_function_array + 16`
            - Beta distribution with parameters 2 and 8, location `W_mean`, scale 20.

        - For `self.source` == 1:
        - Scenario 1:
            `W_mean = cardinal_function_array`
            - No distribution support warning.
            - Binomially determined uniform distribution between 0 and `W_mean` or `W_mean` and 40.
        - Scenario 2:
            `W_mean = cardinal_function_array`
            - Normal distribution with mean `W_mean` and std dev 10.
        """

        if self.source == 0:
            if self.scenario == 1:
                W_mean = 9 * cardinal_function_array + 17

                if return_type_distribution:
                    return NormalDist(W_mean, 5)
                else:
                    return W_mean + np.random.normal(0, 5, N)

            elif self.scenario == 2:
                W_mean = 15 * cardinal_function_array + 22

                if return_type_distribution:
                    return StudentTDist(df=2, loc=W_mean)
                else:
                    return W_mean + np.random.standard_t(2, N)

            elif self.scenario == 3:
                W_mean = 9 * cardinal_function_array + 15

                if return_type_distribution:
                    return NormalDist(W_mean, 5)
                else:
                    return W_mean + np.random.normal(0, 5, N)

            elif self.scenario == 4:
                W_mean = 49 * np.exp(cardinal_function_array) / \
                    (1 + np.exp(cardinal_function_array)) - 6

                if return_type_distribution:
                    return NormalDist(W_mean, 5)
                else:
                    return W_mean + np.random.normal(0, 5, N)

            elif self.scenario == 5:
                W_mean = 42 * 1 / (1 + np.exp(cardinal_function_array)) - 18

                if return_type_distribution:
                    return NormalDist(W_mean, 5)
                else:
                    return W_mean + np.random.normal(0, 5, N)

            elif self.scenario == 6:
                W_mean = 7 * np.log(np.abs(cardinal_function_array)+0.001) + 13

                if return_type_distribution:
                    return NormalDist(W_mean, 4)
                else:
                    return W_mean + np.random.normal(0, 4, N)

            elif self.scenario == 7:
                W_mean = 7 * cardinal_function_array + 16

                if return_type_distribution:
                    return NormalDist(W_mean, 1)
                else:
                    return W_mean + np.random.normal(0, 1, N)

            elif self.scenario == 8:
                W_mean = 7 * cardinal_function_array + 16

                if return_type_distribution:
                    return BetaDist(2, 8, loc=W_mean, scale=20)
                else:
                    return W_mean + np.random.beta(2, 8, N)*20

        elif self.source == 1:
            if self.scenario == 1:
                W_mean = cardinal_function_array
                if return_type_distribution:
                    # warnings.warn(
                    #     "The return distribution only supports the pdf and interval functions")
                    return CombinedUniform(0.3, W_mean, UniformDist(0, W_mean), UniformDist(W_mean, 40))
                else:
                    binomail_arr = np.random.binomial(1, 0.3, N)
                    return binomail_arr * np.random.uniform(0, W_mean, N) + (1-binomail_arr) * np.random.uniform(W_mean, 40, N)

            elif self.scenario == 2:
                W_mean = cardinal_function_array
                if return_type_distribution:
                    return NormalDist(W_mean, 10)
                else:
                    return W_mean + np.random.normal(0, 10, N)
                
        elif self.source == 2:
            if self.scenario == 1 or self.scenario == 2:
                W_mean = cardinal_function_array

                if return_type_distribution:
                    return NormalDist(W_mean, 4)
                else:
                    return W_mean + np.random.normal(0, 4, N)

    # Source: Matching on Generalized Propensity Scores with Continuous Exposures
    # C1-C4 is from Normal(0,I4)
    # C5 from V(-2,2) = discrete uniform
    # C6 from U(-3,3) = continuous uniform
    # Cardinal function =  gamma(C) = −0.8 + (0.1, 0.1, −0.1, 0.2, 0.1, 0.1)C
    # Y = cubic function in function of treatment W and covariates C ~ N(mu(W,C), 10^2), with
    # mu(W,C) = −10 − (2, 2, 3, −1, 2, 2)C − W(0.1 − 0.1 C1 + 0.1 C4 + 0.1 C5 +0.1 C3**2) + 0.13**2 W**3

    def generate_synthetic_DRF_data(self, N : int = 1000) -> pd.DataFrame:
        """
        Generate synthetic data for DRF (Dose Response Function) analysis.

        Parameters
        ----------
        N : int, optional
            Number of data points to generate. Default is 1000.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the synthetic data, including features, treatment, and response variables.

        Notes
        -----
        - For `self.source` == 0:
        - Features `C1, C2, C3, C4` are drawn from a multivariate normal distribution.
        - Feature `C5` is drawn from a discrete uniform distribution with values `-2, -1, 0, 1, 2`.
        - Feature `C6` is drawn from a uniform distribution in the range `[-3, 3]`.
        - For `self.source` == 1:
        - Feature `X` is drawn from a discrete uniform distribution with values `1, 2, 3, 4`.

        """
        if self.source == 0:
            C_1234 = np.random.multivariate_normal(
                [0, 0, 0, 0], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], N)
            C1 = C_1234[:, 0].reshape(-1,1)
            C2 = C_1234[:, 1].reshape(-1,1)
            C3 = C_1234[:, 2].reshape(-1,1)
            C4 = C_1234[:, 3].reshape(-1,1)
            C5 = np.random.choice([-2, -1, 0, 1, 2], N).reshape(-1,1)
            C6 = np.random.uniform(-3, 3, N).reshape(-1,1)

            feature_array = np.hstack((C1, C2, C3, C4, C5, C6))
            feature_columns = ["C1", "C2", "C3", "C4", "C5", "C6"]

        elif self.source == 1:
            X = np.random.choice([1, 2, 3, 4], N)

            feature_array = X
            feature_columns = ["X"]

        elif self.source == 2:
            C1 = np.random.normal(0,5,N).reshape(-1,1)
            C2 = np.random.normal(0,5,N).reshape(-1,1)
            C3 = np.random.normal(0,5,N).reshape(-1,1)

            feature_array = np.hstack((C1, C2, C3))
            feature_columns = ["C1", "C2", "C3"]

        cardinal_function_array = self.get_cardinal_function(feature_array)

        W = self.get_treatment_scenario(
            cardinal_function_array, return_type_distribution=False, N=N)
        
        Y_normal_mean = self.calculate_y_mean(feature_array, W)
        Y = Y_normal_mean + np.random.normal(0, self.Y_std, N)

        if self.source == 0 or self.source == 2:
            synthetic_DRF_df = pd.DataFrame(np.hstack(
                (feature_array, W.reshape(-1,1), Y.reshape(-1,1))), columns=feature_columns+["W", "Y"])

        elif self.source == 1:
            synthetic_DRF_df = pd.DataFrame(np.hstack(
                (feature_array.reshape(-1,1), W.reshape(-1,1), Y.reshape(-1,1))), columns=feature_columns+["W", "Y"])

        return synthetic_DRF_df

    def get_treatment_mean(self, feature_array : np.ndarray):
        """
        Calculate the mean treatment values based on the given feature array.

        Parameters
        ----------
        feature_array : np.ndarray
            A 2D numpy array where each row represents a different data point and each column corresponds to a feature.

        Returns
        -------
        scipy.stats.rv_continuous
            The distribution object representing the mean treatment values.

        """
        cardinal_function_array = self.get_cardinal_function(feature_array)

        return self.get_treatment_scenario(cardinal_function_array, return_type_distribution=True)

    def get_treatment_std(self, feature_array: np.ndarray) -> float:
        """
        Calculate the standard deviation of treatment values based on the given feature array.

        Parameters
        ----------
        feature_array : np.ndarray
            A 2D numpy array where each row represents a different data point and each column corresponds to a feature.

        Returns
        -------
        float
            The standard deviation of the treatment values.

        """
        return (self.get_treatment_mean(feature_array).interval(0.6827)[1] - self.get_treatment_mean(feature_array).interval(0.6827)[0])/2

    def extract_CDRF(self, feature_array: np.ndarray, treatment_array: np.ndarray) -> np.ndarray:
        """
        Extract the Conditional Data-Driven Response Function (CDRF) based on the given feature and treatment arrays.

        Parameters
        ----------
        feature_array : np.ndarray
            A 2D numpy array where each row represents a different data point and each column corresponds to a feature.
        
        treatment_array : np.ndarray
            A 1D numpy array where each element corresponds to the treatment applied to the respective data point in `feature_array`.

        Returns
        -------
        np.ndarray
            A 1D array with the computed mean response variable `Y` for each data point based on the features and treatments.
        """

        Y_normal_mean = self.calculate_y_mean(feature_array, treatment_array)

        return Y_normal_mean
