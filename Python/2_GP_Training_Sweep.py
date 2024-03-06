# -*- coding: utf-8 -*-
"""
Script to train several GPs on sigma profiles to predict a physicochemical
propery. Goal is to obtain the best GP configuration. Training and testing
datasets taken from 10.1039/D2CC01549H.

Sections:
    . Imports
    . Configuration
    . Auxiliary Functions
        . normalize()
        . buildGP()
        . gpPredict()
        . testGPConfig()
    . Main Script
    . Plots

Last edit: 2024-02-10
Author: Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import warnings

# Specific
import numpy
import pandas
from sklearn import metrics
from sklearn import preprocessing
import gpflow
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# Path to database folder
dbPath=r'/path/to/Databases'
# List of database codes
dbList=['MM','BP','VP','D_20','RI_20','S_25']
# List of kernels
kernelList=['RBF','RQ','Matern32','Matern52']
# Normalization
featureNormList=[None,'MinMax','Log+bStand']
labelNormList=[None,'Standardization','LogStand']

# =============================================================================
# Auxiliary Functions
# =============================================================================

def normalize(inputArray,skScaler=None,method='Standardization',reverse=False):
    """
    normalize() normalizes (or unnormalizes) inputArray using the method
    specified and the skScaler provided.

    Parameters
    ----------
    inputArray : numpy array
        Array to be normalized. If dim>1, array is normalized column-wise.
    skScaler : scikit-learn preprocessing object or None
        Scikit-learn preprocessing object previosly fitted to data. If None,
        the object is fitted to inputArray.
        Default: None
    method : string, optional
        Normalization method to be used.
        Methods available:
            . Standardization - classic standardization, (x-mean(x))/std(x)
            . MinMax - scale to range (0,1)
            . LogStand - standardization on the log of the variable,
                         (log(x)-mean(log(x)))/std(log(x))
            . Log+bStand - standardization on the log of variables that can be
                           zero; uses a small buffer,
                           (log(x+b)-mean(log(x+b)))/std(log(x+b))
        Defalt: 'Standardization'
    reverse : bool
        Whether  to normalize (False) or unnormalize (True) inputArray.
        Defalt: False

    Returns
    -------
    inputArray : numpy array
        Normalized (or unnormalized) version of inputArray.
    skScaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object fitted to inputArray. It is the same
        as the inputted skScaler, if it was provided.

    """
    # If inputArray is a labels vector of size (N,), reshape to (N,1)
    if inputArray.ndim==1:
        inputArray=inputArray.reshape((-1,1))
        warnings.warn('Input to normalize() was of shape (N,). It was assumed'\
                      +' to be a column array and converted to a (N,1) shape.')
    # If skScaler is None, train for the first time
    if skScaler is None:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand': aux=numpy.log(inputArray)
        elif method=='Log+bStand': aux=numpy.log(inputArray+10**-3)
        else: raise ValueError('Could not recognize method in normalize().')
        if method!='MinMax':
            skScaler=preprocessing.StandardScaler().fit(aux)
        else:
            skScaler=preprocessing.MinMaxScaler().fit(aux)
    # Do main operation (normalize or unnormalize)
    if reverse:
        # Rescale the data back to its original distribution
        inputArray=skScaler.inverse_transform(inputArray)
        # Check method
        if method=='LogStand': inputArray=numpy.exp(inputArray)
        elif method=='Log+bStand': inputArray=numpy.exp(inputArray)-10**-3
    elif not reverse:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand': aux=numpy.log(inputArray)
        elif method=='Log+bStand': aux=numpy.log(inputArray+10**-3)
        else: raise ValueError('Could not recognize method in normalize().')
        inputArray=skScaler.transform(aux)
    # Return
    return inputArray,skScaler

def buildGP(X_Train,Y_Train,gpConfig={}):
    """
    buildGP() builds and fits a GP model using the training data provided.

    Parameters
    ----------
    X_Train : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).
    Y_Train : numpy array (N,1)
        Training labels (e.g., property of a given molecule).
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP. If a key is not
        present in the dictionary, its default value is used.
        Keys:
            . kernel : string
                Kernel to be used. One of:
                    . 'RBF' - gpflow.kernels.RBF()
                    . 'RQ' - gpflow.kernels.RationalQuadratic()
                    . 'Matern32' - gpflow.kernels.Matern32()
                    . 'Matern52' - gpflow.kernels.Matern52()
                The default is 'RQ'.
            . useWhiteKernel : boolean
                Whether to use a White kernel (gpflow.kernels.White).
                The default is True.
            . trainLikelihood : boolean
                Whether to treat the variance of the likelihood of the modeal
                as a trainable (or fitting) parameter. If False, this value is
                fixed at 10^-5.
                The default is True.
        The default is {}.
    Raises
    ------
    UserWarning
        Warning raised if the optimization (fitting) fails to converge.

    Returns
    -------
    model : gpflow.models.gpr.GPR object
        GP model.

    """
    # Unpack gpConfig
    kernel=gpConfig.get('kernel','RQ')
    useWhiteKernel=gpConfig.get('useWhiteKernel','True')
    trainLikelihood=gpConfig.get('trainLikelihood','True')
    # Select and initialize kernel
    if kernel=='RBF':
        gpKernel=gpflow.kernels.SquaredExponential()
    if kernel=='RQ':
        gpKernel=gpflow.kernels.RationalQuadratic()
    if kernel=='Matern32':
        gpKernel=gpflow.kernels.Matern32()
    if kernel=='Matern52':
        gpKernel=gpflow.kernels.Matern52()
    # Add White kernel
    if useWhiteKernel: gpKernel=gpKernel+gpflow.kernels.White()
    # Build GP model    
    model=gpflow.models.GPR((X_Train,Y_Train),gpKernel,noise_variance=10**-5)
    # Select whether the likelihood variance is trained
    gpflow.utilities.set_trainable(model.likelihood.variance,trainLikelihood)
    # Build optimizer
    optimizer=gpflow.optimizers.Scipy()
    # Fit GP to training data
    aux=optimizer.minimize(model.training_loss,
                           model.trainable_variables,
                           method='L-BFGS-B')
    # Check convergence
    if aux.success==False:
        warnings.warn('GP optimizer failed to converge.')
    # Output
    return model

def gpPredict(model,X):
    """
    gpPredict() returns the prediction and standard deviation of the GP model
    on the X data provided.

    Parameters
    ----------
    model : gpflow.models.gpr.GPR object
        GP model.
    X : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).

    Returns
    -------
    Y : numpy array (N,1)
        GP predictions.
    STD : numpy array (N,1)
        GP standard deviations.

    """
    # Do GP prediction, obtaining mean and variance
    GP_Mean,GP_Var=model.predict_f(X)
    # Convert to numpy
    GP_Mean=GP_Mean.numpy()
    GP_Var=GP_Var.numpy()
    # Prepare outputs
    Y=GP_Mean
    STD=numpy.sqrt(GP_Var)
    # Output
    return Y,STD

def testGPConfig(dbPath,code,featureNorm,labelNorm,gpConfig):
    """
    testGPConfig() fits a GP with the requested configuration to a given
    dataset and evaluates its performance.

    Parameters
    ----------
    dbPath : string
        Path to the folder containing the property datasets.
    code : string
        Dataset code. One of 'MM','BP','VP','D_20','RI_20','S_25'.
    featureNorm : string or None
        Normalization procedure for the features. One of None, Standardization,
        MinMax, LogStand, or Log+bStand.
    labelNorm : string or None
        Normalization procedure for the labels. One of None, Standardization,
        MinMax, LogStand, or Log+bStand..
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP. If a key is not
        present in the dictionary, its default value is used.
        Keys:
            . kernel : string
                Kernel to be used. One of:
                    . 'RBF' - gpflow.kernels.RBF()
                    . 'RQ' - gpflow.kernels.RationalQuadratic()
                    . 'Matern32' - gpflow.kernels.Matern32()
                    . 'Matern52' - gpflow.kernels.Matern52()
                The default is 'RQ'.
            . useWhiteKernel : boolean
                Whether to use a White kernel (gpflow.kernels.White).
                The default is True.
            . trainLikelihood : boolean
                Whether to treat the variance of the likelihood of the modeal
                as a trainable (or fitting) parameter. If False, this value is
                fixed at 10^-5.
                The default is True.
        The default is {}.

    Returns
    -------
    R2_Train : float
        R2 for the training set.
    R2_Test : float
        R2 for the testing set.

    """
    # Path to CSV files
    trainDB_Path=os.path.join(dbPath,code+'_mlDatabase.csv')
    testDB_Path=os.path.join(dbPath,code+'_mlDatabase_TestSet.csv')
    # Load CSV file
    trainDB=pandas.read_csv(trainDB_Path)
    testDB=pandas.read_csv(testDB_Path)
    # Extract sigma profiles and target variable as numpy arrays
    X_Train=trainDB.iloc[:,4:-1].to_numpy()
    Y_Train=trainDB.iloc[:,3].to_numpy().reshape(-1,1)
    X_Test=testDB.iloc[:,4:-1].to_numpy()
    Y_Test=testDB.iloc[:,3].to_numpy().reshape(-1,1)
    # Normalize
    if featureNorm is not None:
        X_Train_N,skScaler_X_Train=normalize(X_Train,method=featureNorm)
        X_Test_N,__=normalize(X_Test,skScaler=skScaler_X_Train,
                              method=featureNorm)
    else:
        X_Train_N=X_Train
        X_Test_N=X_Test
    if labelNorm is not None:
        Y_Train_N,skScaler_Y_Train=normalize(Y_Train,method=labelNorm)
    else:
        Y_Train_N=Y_Train
    # Train GP
    model=buildGP(X_Train_N,Y_Train_N,gpConfig=gpConfig)
    # Get GP predictions
    Y_Train_Pred_N,STD_Train=gpPredict(model,X_Train_N)
    Y_Test_Pred_N,STD_Test=gpPredict(model,X_Test_N)
    # Unnormalize
    if labelNorm is not None:
        Y_Train_Pred,__=normalize(Y_Train_Pred_N,skScaler=skScaler_Y_Train,
                                  method=labelNorm,reverse=True)
        Y_Test_Pred,__=normalize(Y_Test_Pred_N,skScaler=skScaler_Y_Train,
                                 method=labelNorm,reverse=True)
    else:
        Y_Train_Pred=Y_Train_Pred_N
        Y_Test_Pred=Y_Test_Pred_N
    # Compute metrics
    if code=='VP' or code=='S_25':
        # Try/except to prevent cases where Y contains infinity
        try:
            R2_Train=metrics.r2_score(numpy.log(Y_Train),
                                      numpy.log(Y_Train_Pred))
        except:
            print('R2_Train calculation failed for',code,'|',featureNorm,'|',
                  labelNorm,'|',gpConfig.get('kernel'))
            R2_Train=-100
        try:
            R2_Test=metrics.r2_score(numpy.log(Y_Test),numpy.log(Y_Test_Pred))
        except:
            print('R2_Test calculation failed for',code,'|',featureNorm,'|',
                  labelNorm,'|',gpConfig.get('kernel'))
            R2_Test=-100
    else:
        try:
            R2_Train=metrics.r2_score(Y_Train,Y_Train_Pred)
        except:
            print('R2_Train calculation failed for',code,'|',featureNorm,'|',
                  labelNorm,'|',gpConfig.get('kernel'))
            R2_Train=-100
        try:
            R2_Test=metrics.r2_score(Y_Test,Y_Test_Pred)
        except:
            print('R2_Test calculation failed for',code,'|',featureNorm,'|',
                  labelNorm,'|',gpConfig.get('kernel'))
            R2_Test=-100
    return R2_Train,R2_Test

# =============================================================================
# Main Script
# =============================================================================

# Define metric containers
sweepResult_Train=numpy.zeros((4, # kernelList
                               3, # featureNorm
                               3, # labelNorm
                               6, # database code
                               ))
sweepResult_Test=numpy.zeros((4, # kernelList
                              3, # featureNorm
                              3, # labelNorm
                              6, # database code
                              ))
# Define main loops
for k,kernel in tqdm(enumerate(kernelList),'Kernel: '):
    for fN,featureNorm in enumerate(featureNormList):
        for fL,labelNorm in enumerate(labelNormList):
            for c,code in enumerate(dbList):
                # Define input variables
                gpConfig={'kernel':kernel,
                          'useWhiteKernel':True,
                          'trainLikelihood':True}
                # Test GP configuration
                R2_Train,R2_Test=testGPConfig(dbPath,code,featureNorm,
                                              labelNorm,gpConfig)
                # Update containers
                sweepResult_Train[k,fN,fL,c]=R2_Train
                sweepResult_Test[k,fN,fL,c]=R2_Test
# Print best combinations
featureNormList[0]='None' # Replace None with 'None'
labelNormList[0]='None' # Replace None with 'None'
for c,code in enumerate(dbList): # Iterate over datasets
    # Get index tuple of max
    index=numpy.unravel_index(sweepResult_Test[:,:,:,c].argmax(),
                              sweepResult_Test[:,:,:,c].shape)
    # Print
    print('\n\nCode: '+code)
    print('\n     Best kernel: '+kernelList[index[0]])
    print('\n     Best feature norm.: '+featureNormList[index[1]])
    print('\n     Best label norm.: '+labelNormList[index[2]])
    print('\n     Performance: '\
          +'{:.2f} '.format(sweepResult_Test[:,:,:,c].max()))
# Print results (remove R^2<0)
sweepResult_Test[sweepResult_Test<0]=numpy.nan
print(sweepResult_Test)

for n in sweepResult_Test[3,2,2,:]: print('{:.2f} '.format(n))