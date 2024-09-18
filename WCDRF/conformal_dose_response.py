from crepes_weighted import ConformalPredictiveSystem, WrapRegressor  # type: ignore
from crepes_weighted.extras import DifficultyEstimator, binning  # type: ignore

from scipy.stats import norm, t, rayleigh, beta
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

## Synthetic data generation

def get_cardinal_function(C1, C2, C3, C4, C5, C6):
    # cardinal_function_array = np.sin(0.5*np.pi*(-0.8 + 0.1*C1 + 0.1*C2 - 0.1*C3 + 0.2*C4 + 0.1*C5 + 0.1*C6))
    # cardinal_function_array = -0.8 + 0.1*C1 + 0.1*C2 - 0.1*C3 + 0.2*C4 + 0.1*C5 + 0.1*C6

    # return -0.8 + 0.1*C1 + 0.1*C2 - 0.1*C3 + 0.2*C4 + 0.1*C5 + 0.1*C6
    # return -0.8 + 1*C1 + 0.1*C2 - 0.1*C3 + 0.2*C4 + 0.1*C5 + 0.1*C6
    
    return 5*C5

def calculate_y_mean(C1, C2, C3, C4, C5, C6, W):
    # Y_normal_mean = -1 - (2*C1 + 2*C2 + 3*C3**3 -1*C4 - 20*C5 + 20*C6) - (W) * (0.1 - 0.1*C1 + 0.1*C4 + 0.1*C5 + 0.1 * C3**2) + 0.13**2 * (W)**3
    # Y_normal_mean = -1 - (2*C1 + 2*C2 + 3*C3**3 -20*C4 - 2*C5 + 20*C6) - (W) * (0.1 - 0.1*C1 + 0.1*C4 + 0.1*C5 + 0.1 * C3**2) + 0.13**2 * (np.abs(W))**3 * np.sin(C4)
    # Y_normal_mean = -1 - (2*C1 + 2*C2 + 3*C3**3 -20*C4 - 2*C5 + 20*C6) - 2*(W) * (C4**3) + (0.13**2) * (W)**(2+np.abs(C5))

    # return -1 - (2*C1 + 2*C2 + 3*C3**3 -20*C4 - 2*C5 + 20*C6) - (W) * (0.1 - 0.1*C1 + 0.1*C4 + 0.1*C5 + 0.1 * C3**2) + 0.13**2 * (np.abs(W))**3 * np.sin(C4)
    # return -1 - (2*C1 + 2*C2 + 3*C3**3 -20*C4 - 2*C5 + 20*C6) - (W) * (0.1 - 0.1*C1 + 0.1*C4 + 0.1*C5 + 0.1 * C3**2) + 0.13**2 * (np.abs(W))**3 * np.sin(C4)
    
    return np.sin(np.pi/2 * (0.1*W - 0.1*C5))

def get_treatment_scenario(scenario, cardinal_function_array, return_type_distribution=False, N = None):

    if scenario < 1 or scenario > 10:
        print("scenario number undefined, setting it to default 1")
        scenario = 1

    if scenario == 1:
        W_mean = 9 * cardinal_function_array + 17 

        if return_type_distribution:
            return norm(W_mean,5)
        else:
            return W_mean + np.random.normal(0, 5, N)
    
    elif scenario == 2:
        W_mean = 15 * cardinal_function_array + 22 

        if return_type_distribution:
            return t(df=2,loc=W_mean)
        else:
            return W_mean + np.random.standard_t(2, N)
    
    elif scenario == 3:
        W_mean = 9 * cardinal_function_array + 3/2 * C3**2 + 15

        if return_type_distribution:
            return norm(W_mean,5)
        else:
            return W_mean + np.random.normal(0, 5, N)
    
    elif scenario == 4:
        W_mean = 49 * np.exp(cardinal_function_array) / (1 + np.exp(cardinal_function_array)) - 6 

        if return_type_distribution:
            return norm(W_mean,5)
        else:
            return W_mean + np.random.normal(0, 5, N)
    
    elif scenario == 5:
        W_mean = 42 * 1 / (1 + np.exp(cardinal_function_array)) - 18 

        if return_type_distribution:
            return norm(W_mean,5)
        else:
            return W_mean + np.random.normal(0, 5, N)
    
    elif scenario == 6:
        W_mean = 7 * np.log(np.abs(cardinal_function_array)+0.001) + 13

        if return_type_distribution:
            return norm(W_mean,4)
        else:
            return W_mean + np.random.normal(0, 4, N)
    
    elif scenario == 7:
        W_mean = 7 * cardinal_function_array + 16 

        if return_type_distribution:
            return norm(W_mean,1)
        else:
            return W_mean + np.random.normal(0, 1, N)

    elif scenario == 8:
        W_mean = 7 * cardinal_function_array + 16 

        if return_type_distribution:
            return beta(2, 8, loc=W_mean, scale=20)
        else:
            return W_mean + np.random.beta(2, 8, N)*20

    elif scenario == 9:
        W_mean = cardinal_function_array
        if return_type_distribution:
            return norm(W_mean,1)
        else:
            binomail_arr = np.random.binomial(1,0.3, N)
            return binomail_arr * np.random.uniform(0,W_mean, N) + (1-binomail_arr) * np.random.uniform(W_mean, 40, N)

    elif scenario == 10:
        W_mean = cardinal_function_array
        if return_type_distribution:
            return norm(W_mean,10)
        else:
            return W_mean + np.random.normal(0, 10, N)


# Source: Matching on Generalized Propensity Scores with Continuous Exposures
# C1-C4 is from Normal(0,I4)
# C5 from V(-2,2) = discrete uniform 
# C6 from U(-3,3) = continuous uniform
# Cardinal function =  gamma(C) = −0.8 + (0.1, 0.1, −0.1, 0.2, 0.1, 0.1)C
# Y = cubic function in function of treatment W and covariates C ~ N(mu(W,C), 10^2), with 
# mu(W,C) = −10 − (2, 2, 3, −1, 2, 2)C − W(0.1 − 0.1 C1 + 0.1 C4 + 0.1 C5 +0.1 C3**2) + 0.13**2 W**3
def generate_synthetic_DRF_data(scenario=1, N=1000):

    C_1234 = np.random.multivariate_normal([0,0,0,0],[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],N)
    C1 = C_1234[:,0]
    C2 = C_1234[:,1]
    C3 = C_1234[:,2]
    C4 = C_1234[:,3]
    # C5 = np.random.choice([-2,-1,0,1,2],N)
    C5 = np.random.choice([1,2,3,4],N)
    C6 = np.random.uniform(-3,3,N)
    
    cardinal_function_array = get_cardinal_function(C1, C2, C3, C4, C5, C6)

    W = get_treatment_scenario(scenario, cardinal_function_array, return_type_distribution=False, N = N)
    Y_normal_mean = calculate_y_mean(C1, C2, C3, C4, C5, C6, W)

    # Y = Y_normal_mean + np.random.normal(0, 5, N)
    Y = Y_normal_mean + np.random.normal(0, 0.1, N)

    synthetic_DRF_df = pd.DataFrame(np.vstack([C1,C2,C3,C4,C5,C6,W,Y]).transpose(),columns=["C1","C2","C3","C4","C5","C6","W","Y"])

    return synthetic_DRF_df

def get_treatment_mean(C_arr, scenario = 1):
    C1 = C_arr[0]
    C2 = C_arr[1]
    C3 = C_arr[2]
    C4 = C_arr[3]
    C5 = C_arr[4]
    C6 = C_arr[5]

    cardinal_function_array = get_cardinal_function(C1, C2, C3, C4, C5, C6)

    return get_treatment_scenario(scenario, cardinal_function_array, return_type_distribution=True)
        

def get_treatment_std(C_arr, scenario = 1):
    return (get_treatment_mean(C_arr, scenario).interval(0.6827)[1] - get_treatment_mean(C_arr, scenario).interval(0.6827)[0])/2


def extract_CDRF(C_arr, treatment_array):

    C1 = C_arr[0]
    C2 = C_arr[1]
    C3 = C_arr[2]
    C4 = C_arr[3]
    C5 = C_arr[4]
    C6 = C_arr[5]
    W = treatment_array

    Y_normal_mean = calculate_y_mean(C1, C2, C3, C4, C5, C6, W)

    return Y_normal_mean



def get_propensity_pdf(likelihood_prop_model, to_likelihood_dataset, propensity_features_columns, treatment_column, KernelDensity_kernel='gaussian'):
    output_cps = likelihood_prop_model.predict_cps(
        to_likelihood_dataset[propensity_features_columns], return_cpds=True)

    return np.array([np.exp(KernelDensity(kernel=KernelDensity_kernel, bandwidth=1.0).fit((output_cps[idx, :, 0]).reshape(-1, 1)).score_samples(np.array(row[treatment_column]).reshape(-1, 1))[0]) for idx, row in to_likelihood_dataset.reset_index(drop=True).iterrows()])


def get_target_treatment_propensity_pdf(likelihood_prop_model, target_treatment, to_likelihood_dataset, propensity_features_columns, KernelDensity_kernel='gaussian'):
    output_cps = likelihood_prop_model.predict_cps(
        to_likelihood_dataset[propensity_features_columns], return_cpds=True)

    if len(np.shape(target_treatment)) == 0:
        return np.array([np.exp(KernelDensity(kernel=KernelDensity_kernel, bandwidth=1.0).fit((output_cps[idx, :, 0]).reshape(-1, 1)).score_samples(np.array(target_treatment).reshape(-1, 1))[0]) for idx in range(len(to_likelihood_dataset))])
    elif len(target_treatment) == len(to_likelihood_dataset):
        return np.array([np.exp(KernelDensity(kernel=KernelDensity_kernel, bandwidth=1.0).fit((output_cps[idx, :, 0]).reshape(-1, 1)).score_samples(np.array(target_treatment[idx]).reshape(-1, 1))[0]) for idx in range(len(to_likelihood_dataset))])
    else:
        return ["error"]


def get_lengthscale(likelihood_prop_model, to_likelihood_dataset, propensity_features_columns, multiplier=0.2):
    std_bounds = likelihood_prop_model.predict_int(
        to_likelihood_dataset[propensity_features_columns], confidence=0.6827)
    return 2 * ((multiplier * (std_bounds[:, 1] - std_bounds[:, 0])/2) ** 2)


def calculate_calibration_likelihood(target_treatment, treatment_arr, lengthscale_arr, D_di_propensity_arr):
    return np.exp(- ((treatment_arr - target_treatment)**2) / lengthscale_arr) / D_di_propensity_arr


def calculate_evaluation_likelihood(evaluation_propensities):
    return 1 / evaluation_propensities