
from crepes_weighted import ConformalPredictiveSystem, WrapRegressor  # type: ignore
from crepes_weighted.extras import DifficultyEstimator, binning  # type: ignore

from scipy.stats import norm, t, rayleigh, beta
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from catboost import CatBoostRegressor

def gaussian_kernel_function(target_treatment, T, lengthscale_arr):
        return np.exp(- (((T - target_treatment)**2) / lengthscale_arr))



class OraclePropensityWrapper():
    def __init__(self, init_datagenerator):
        self.datagenerator = init_datagenerator

    def get_std(self, X):
        if np.shape(X)[1] == 1:
            return_arr = np.array([self.datagenerator.get_treatment_std(X[idx]) for idx in range(len(X))])
        else:
            return_arr = np.array([self.datagenerator.get_treatment_std(X[idx].reshape(1,-1)) for idx in range(len(X))])

        if len(np.shape(return_arr)) > 1:
            return return_arr[:,0]
        else:
            return return_arr
    
    def get_oracle_propensity(self, X, T):

        if len(np.shape(T))>0:
            if np.shape(X)[1] == 1:
                return_arr = np.array([self.datagenerator.get_treatment_mean(X[idx]).pdf(T[idx]) for idx in range(len(X))])
            else:
                return_arr = np.array([self.datagenerator.get_treatment_mean(X[idx].reshape(1,-1)).pdf(T[idx]) for idx in range(len(X))])
        else: 
            return_arr = np.array([self.datagenerator.get_treatment_mean(X[idx].reshape(1,-1)).pdf(T) for idx in range(len(X))])

        if len(np.shape(return_arr)) > 1:
            return return_arr[:,0]
        else:
            return return_arr
        # return np.array([self.datagenerator.get_treatment_mean(X[idx].reshape(1,-1)).pdf(T[idx]) for idx in range(len(X))])



class DRFWrapRegressor(WrapRegressor):
    """
    A regressor wrapper for Dose Response Function (DRF) calibration.

    This class extends the `WrapRegressor` from weighted crepes to include methods for fitting,
    calibrating, and evaluating machine learning models with Weighted Conformal Prediction.

    Parameters
    ----------
    propensityLearner : object
        The propensity learner model.
    learner : object
        The base learner model.

    Attributes
    ----------
    propensityLearner : WrapRegressor
        The wrapped propensity learner model.
    multicalibration : bool
        Flag indicating if multicalibration is enabled.
    DRF_calibrated : bool
        Flag indicating if the model is calibrated using DRF.
    effective_sample_rate : float
        The effective sample rate after calibration.
    local_conditional_mode : bool
        A model to calibrate locally on treatment
    density_bandwidth : float
        The bandwidth that is used for the Kernel Density
    local_kernel_function: function
        A variable to save the kernel function, must have the arguments: target_treatment, T, and lengthscale_arr

    Methods
    -------
    fit_propensity(X, T, **kwargs)
        Fits the propensity model.
    calibrate_propensity(X, T, **kwargs)
        Calibrates the propensity model.
    get_propensity_pdf(X, T, KernelDensity_kernel='gaussian')
        Returns the propensity probability density function.
    get_lengthscale(X, multiplier=0.2)
        Computes the lengthscale for calibration.
    prepare_calibration(X, T, multiplier=0.2)
        Prepares the DRF model for calibration.
    get_target_treatment_propensity_pdf(target_treatment, X, KernelDensity_kernel='gaussian')
        Returns the propensity PDF for a target treatment.
    calculate_calibration_likelihood(D_di_propensity_arr, target_treatment, T, lengthscale_arr)
        Calculates the calibration likelihood.
    calculate_evaluation_likelihood(evaluation_propensities)
        Calculates the evaluation likelihood.
    calibrate(X, y, T, target_treatment, multiplier=0.2, **kwargs)
        Calibrates the DRF model using Weighted Conformal Prediction.
    predict_int(X, **kwargs)
        Predicts intervals using the calibrated model.
    predict_cps(X, **kwargs)
        Predicts the uncertainty distributions using the calibrated model.
    search_lengthscale(X, T, N_treatments=20, arr_Mult=np.arange(0.05, 2, 0.05))
        Searches for the optimal lengthscale for calibration.
    """

    def __init__(self, propensityLearner, learner, difficultyEstimator = DifficultyEstimator(), adaptive = False, oracle_propensity = False):
        """
        Initializes the DRFWrapRegressor.

        Parameters
        ----------
        propensityLearner : object
            The propensity learner model.
        learner : object
            The base learner model.
        """
        super().__init__(learner)
        self.oracle_propensity = oracle_propensity
        if not self.oracle_propensity:
            self.propensityLearner = WrapRegressor(propensityLearner)
        else:
            self.propensityLearner = propensityLearner
        self.multicalibration = False
        self.local_conditional_mode = True
        self.density_bandwidth = 1.0
        self.DRF_calibrated = False
        self.adaptive_mode = adaptive
        self.effective_sample_rate = 0
        self.local_kernel_function = gaussian_kernel_function
        self.de = difficultyEstimator
        self.de_fitted_flag = False

    def set_local_kernel_function(self, function_name):
        self.local_kernel_function = function_name
    
    def fit_propensity(self, X, T, **kwargs):
        """
        Fits the propensity model.

        Parameters
        ----------
        X : array-like
            The input features.
        T : array-like
            The treatment assignments.
        **kwargs : dict
            Additional arguments for the fit method.
        """
        if not self.oracle_propensity:
            self.propensityLearner.fit(X, T, **kwargs)
        else:
            raise Exception("propensity fit called however, oracles are set to true, verify the method")



    def calibrate_propensity(self, X, T, **kwargs):
        """
        Calibrates the propensity model.

        Parameters
        ----------
        X : array-like
            The input features.
        T : array-like
            The treatment assignments.
        **kwargs : dict
            Additional arguments for the calibrate method.
        """
        if not self.oracle_propensity:
            self.propensityLearner.calibrate(X, T, **kwargs, cps=True)
        else:
            raise Exception("propensity calibrate called however, oracles are set to true, verify the method")

    def get_propensity_pdf(self, X, T, KernelDensity_kernel='gaussian'):
        """
        Returns the propensity probability density function.

        Parameters
        ----------
        X : array-like
            The input features.
        T : array-like
            The treatment assignments.
        KernelDensity_kernel : str, optional
            The kernel to use for Kernel Density Estimation (default is 'gaussian').

        Returns
        -------
        array-like
            The propensity probability density function for each input.
        """
        if not self.oracle_propensity:
            output_cps = self.propensityLearner.predict_cps(
                X, return_cpds=True)

            return np.array([np.exp(KernelDensity(kernel=KernelDensity_kernel, bandwidth=self.density_bandwidth).fit((output_cps[idx, :, 0]).reshape(-1, 1)).score_samples(np.array(T[idx]).reshape(-1, 1))[0]) for idx in range(len(X))])
        else:
            return self.propensityLearner.get_oracle_propensity(X, T)
         

    def get_lengthscale(self, X, multiplier = 0.2):
        """
        Computes the lengthscale for calibration.

        Parameters
        ----------
        X : array-like
            The input features.
        multiplier : float, optional
            The multiplier to apply to the lengthscale (default is 0.2).

        Returns
        -------
        array-like
            The lengthscale for each input.
        """
        if not self.oracle_propensity:
            std_bounds = self.propensityLearner.predict_int(
                X, confidence=0.6827)
            return 2.0 * ((multiplier * (std_bounds[:, 1] - std_bounds[:, 0])/2.0) ** 2.0)
        else:
            std_output = self.propensityLearner.get_std(X)
            return 2.0 * ((multiplier * (std_output)/2.0) ** 2.0)


    def prepare_calibration(self, X, T, multiplier = 0.2, density_bandwidth = 1.0, use_propensity = True):
        """
        Prepares the model for DRF calibration.

        Parameters
        ----------
        X : array-like
            The input features.
        T : array-like
            The treatment assignments.
        multiplier : float, optional
            The multiplier to apply to the lengthscale (default is 0.2).
        density_bandwidth: float, optional
            The bandwidth used for the KernelDensity estimator for the propensity model
        """
        self.multicalibration = True
        self.density_bandwidth = density_bandwidth
        if use_propensity:
            self.P_D_di_propensities = self.get_propensity_pdf(X, T)
        else:
            self.P_D_di_propensities = np.zeros(len(T))

        
        self.calibration_Lengthscale = self.get_lengthscale(X, multiplier)

    
    def fit_difficulty_estimator(self, X, T, Y):
        """
        Fits the difficuly estimator.

        Parameters
        ----------
        X : array-like
            The input features.
        T : array-like
            The treatment assignments.
        Y : array-like
            The outcomes.
        """
        self.de.fit(self.get_propensity_pdf(X, T).reshape(-1,1), Y)
        self.de_fitted_flag = True


    def get_target_treatment_propensity_pdf(self, X, target_treatment, KernelDensity_kernel='gaussian'):
        """
        Returns the propensity PDF for a target treatment.

        Parameters
        ----------
        target_treatment : float or array-like
            The target treatment assignment(s).
        X : array-like
            The input features.
        KernelDensity_kernel : str, optional
            The kernel to use for Kernel Density Estimation (default is 'gaussian').

        Returns
        -------
        array-like
            The propensity PDF for the target treatment.
        
        Raises
        ------
        Exception
            If the target_treatment is not a single float or an array of the same length as X.
        """

        if not self.oracle_propensity:
            output_cps = self.propensityLearner.predict_cps(
                X, return_cpds=True)

            if len(np.shape(target_treatment)) == 0:
                return np.array([np.exp(KernelDensity(kernel=KernelDensity_kernel, bandwidth=self.density_bandwidth).fit((output_cps[idx, :, 0]).reshape(-1, 1)).score_samples(np.array(target_treatment).reshape(-1, 1))[0]) for idx in range(len(X))])
            
            elif len(target_treatment) == len(X):
                return np.array([np.exp(KernelDensity(kernel=KernelDensity_kernel, bandwidth=self.density_bandwidth).fit((output_cps[idx, :, 0]).reshape(-1, 1)).score_samples(np.array(target_treatment[idx]).reshape(-1, 1))[0]) for idx in range(len(X))])
            
            else:
                raise Exception("T is in an incorrect format, requires either a single float, or a one dimensional array with the same length as X")
        else:
            return self.propensityLearner.get_oracle_propensity(X, target_treatment)



    def calculate_calibration_likelihood(self, D_di_propensity_arr, target_treatment: float = None, T = None, lengthscale_arr = None ):
        """
        Calculates the calibration likelihood.

        Parameters
        ----------
        D_di_propensity_arr : array-like
            The propensity PDF array.
        target_treatment : float
            The target treatment assignment.
        T : array-like
            The treatment assignments.
        lengthscale_arr : array-like, optional
            The lengthscale array.

        Returns
        -------
        array-like
            The calibration likelihood for each input.
        """
        if self.local_conditional_mode:
            if (lengthscale_arr is not None) and (target_treatment is not None) and (T is not None):
                if self.use_propensity:
                    return self.local_kernel_function(target_treatment, T, lengthscale_arr) / D_di_propensity_arr
                else:
                    return self.local_kernel_function(target_treatment, T, lengthscale_arr)   
            else:
                raise Exception("local conditional mode is set to true but any of the required arguments (target_treatment, T, or lengthscale_arr) is None. Set the mode to off.")
        else:
            if self.use_propensity:
                return 1.0 / D_di_propensity_arr
            else:
                return np.ones(len(D_di_propensity_arr))


    def calculate_evaluation_likelihood(self, evaluation_propensities):
        """
        Calculates the evaluation likelihood.

        Parameters
        ----------
        evaluation_propensities : array-like
            The evaluation propensities.

        Returns
        -------
        array-like
            The evaluation likelihood for each input.
        """
        if self.use_propensity:
            return 1.0 / evaluation_propensities
        else:
            return np.ones(len(evaluation_propensities))

    

    def calibrate(self, X, y, T, target_treatment = None, multiplier = 0.2, local_conditional_mode = True, use_propensity = True, **kwargs):
        """
        Calibrates the model using Weighted Conformal Prediction.

        Parameters
        ----------
        X : array-like
            The input features.
        y : array-like
            The target values.
        T : array-like
            The treatment assignments.
        target_treatment : float, optional
            The target treatment assignment.
        multiplier : float, optional
            The multiplier to apply to the lengthscale (default is 0.2).
        local_conditional_mode : bool
            A model to calibrate locally on treatment
        **kwargs : dict
            Additional arguments for the calibrate method.
        """

        self.local_conditional_mode = local_conditional_mode
        self.use_propensity = use_propensity

        if self.adaptive_mode and self.use_propensity:
            if self.de_fitted_flag:
                prop_sigmas = self.de.apply(self.P_D_di_propensities.reshape(-1,1))
            else: 
                raise Exception("The difficulty estimator was not fit yet, call fit_difficulty_estimator first")
        else:
            prop_sigmas = None
        
        if self.local_conditional_mode and target_treatment is None:
            raise Exception("You need to specify a target treatment when using local conditional mode")
        
        if self.multicalibration:
            if self.local_conditional_mode:
                DRF_likelihood_ratios = self.calculate_calibration_likelihood(
                    self.P_D_di_propensities,
                    target_treatment, 
                    T, 
                    self.calibration_Lengthscale
                )
            else:
                DRF_likelihood_ratios = self.calculate_calibration_likelihood(
                    self.P_D_di_propensities
                )
        else:
            if self.local_conditional_mode:
                DRF_likelihood_ratios = self.calculate_calibration_likelihood(
                    self.get_propensity_pdf(X[:, :-1], T),
                    target_treatment, 
                    T, 
                    self.get_lengthscale(X, multiplier)
                )
            else:
                DRF_likelihood_ratios = self.calculate_calibration_likelihood(
                    self.P_D_di_propensities
                )

        super().calibrate(X, y, **kwargs, likelihood_ratios = DRF_likelihood_ratios, sigmas = prop_sigmas)
        self.DRF_calibrated = True
        self.target_treatment = target_treatment
        self.effective_sample_rate = (np.sum(np.abs(DRF_likelihood_ratios))**2)/np.sum(np.abs(DRF_likelihood_ratios)**2) 


    def predict_int(self, X, **kwargs):
        """
        Predicts intervals using the calibrated model.

        Parameters
        ----------
        X : array-like
            The input features, also containing the treatment as the last column
        **kwargs : dict
            Additional arguments for the predict_int method.

        Returns
        -------
        array-like
            The predicted intervals.

        Raises
        ------
        Exception
            If the DRF model is not calibrated for a specific treatment.
        """
        if self.DRF_calibrated:
            if self.use_propensity:
                temporary_propensities = self.get_target_treatment_propensity_pdf(X[:, :-1], self.target_treatment)
                evaluation_likelihoods = self.calculate_evaluation_likelihood(temporary_propensities)
            else:
                evaluation_likelihoods = self.calculate_evaluation_likelihood(np.ones(len(X)))

            if self.adaptive_mode and self.use_propensity:
                prop_sigmas = self.de.apply(temporary_propensities.reshape(-1,1))
            else:
                prop_sigmas = None
    
            return super().predict_int(X, **kwargs, likelihood_ratios = evaluation_likelihoods, sigmas = prop_sigmas)
        else:
            raise Exception("The DRF model is not calibrated for a specific treatment.")
        
    def predict_multi_int(self, X, confidence_range, **kwargs):
        """
        Predicts intervals using the calibrated model.

        Parameters
        ----------
        X : array-like
            The input features, also containing the treatment as the last column
        **kwargs : dict
            Additional arguments for the predict_int method.

        Returns
        -------
        matrix-like
            The predicted intervals in format (len(cov_range), 2, len(X))

        Raises
        ------
        Exception
            If the DRF model is not calibrated for a specific treatment.
        """

        if self.DRF_calibrated:
            if self.use_propensity:
                temporary_propensities = self.get_target_treatment_propensity_pdf(X[:, :-1], self.target_treatment)
                evaluation_likelihoods = self.calculate_evaluation_likelihood(temporary_propensities)
            else:
                evaluation_likelihoods = self.calculate_evaluation_likelihood(np.ones(len(X)))

            if self.adaptive_mode and self.use_propensity:
                prop_sigmas = self.de.apply(temporary_propensities.reshape(-1,1))
            else:
                prop_sigmas = None

            return_bounds_matrix = np.array([])
            for cov in confidence_range:
                return_bounds = super().predict_int(X, **kwargs, confidence = cov, likelihood_ratios = evaluation_likelihoods, sigmas = prop_sigmas)
                if len(return_bounds_matrix) > 0:
                    return_bounds_matrix = np.vstack((return_bounds_matrix, [return_bounds]))
                else:
                    return_bounds_matrix = [return_bounds]
            
            return return_bounds_matrix
        else:
            raise Exception("The DRF model is not calibrated for a specific treatment.")
        
    def predict_cps(self, X, **kwargs):
        """
        Predicts distributions using the calibrated model using cps.

        Parameters
        ----------
        X : array-like
            The input features, also containing the treatment as the last column
        **kwargs : dict
            Additional arguments for the predict_cps method.

        Returns
        -------
        array-like
            The predicted counterfactual distributions.

        Raises
        ------
        Exception
            If the DRF model is not calibrated for a specific treatment.
        """

        

        if self.DRF_calibrated:
            if self.use_propensity:
                temporary_propensities = self.get_target_treatment_propensity_pdf(X[:, :-1], self.target_treatment)
                evaluation_likelihoods = self.calculate_evaluation_likelihood(temporary_propensities)
            else:
                evaluation_likelihoods = self.calculate_evaluation_likelihood(np.ones(len(X)))

            if self.adaptive_mode and self.use_propensity:
                prop_sigmas = self.de.apply(temporary_propensities.reshape(-1,1))
            else:
                prop_sigmas = None

            return super().predict_cps(X, **kwargs, likelihood_ratios = evaluation_likelihoods, sigmas = prop_sigmas)
        else:
            raise Exception("The DRF model is not calibrated for a specific treatment.")
        


    def search_lengthscale(self, X, T, N_treatments = 20, arr_Mult = np.arange(0.05,2,0.05)):
        """
        Searches for the optimal lengthscale for calibration.

        Parameters
        ----------
        X : array-like
            The input features.
        T : array-like
            The treatment assignments.
        N_treatments : int, optional
            The number of treatments to evaluate (default is 20).
        arr_Mult : array-like, optional
            Array of multipliers to search for the optimal lengthscale (default is np.arange(0.05, 2, 0.05)).

        Returns
        -------
        float
            The optimal lengthscale multiplier.
        """
        self.local_conditional_mode = True
        
        P_D_di_propensities = self.get_propensity_pdf(X, T)
        W_min = np.min(T)
        W_max = np.max(T)

        treatment_range =  np.arange(W_min,W_max,(W_max-W_min)/int(N_treatments))

        max_sample_rates_list = []
        for mult in arr_Mult:
            sample_rate_list = []

            lengthscale_arr = self.get_lengthscale(X, multiplier = mult)

            for d_t in treatment_range:

                calibration_likelihood = self.calculate_calibration_likelihood(
                        P_D_di_propensities,
                        d_t, 
                        T,
                        lengthscale_arr
                    )


                if np.sum(calibration_likelihood) < 0.0001:
                    sample_rate_list.append(0)
                else:
                    sample_rate_list.append((np.sum(np.abs(calibration_likelihood))**2)/np.sum(calibration_likelihood**2))

            max_sample_rates_list.append(np.max(np.array(sample_rate_list)))

        return np.arange(0.05,10,0.05)[np.argmax(np.array(max_sample_rates_list))]