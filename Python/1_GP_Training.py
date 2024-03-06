# -*- coding: utf-8 -*-
"""
Script to train a GP on sigma profiles to predict a physicochemical propery.
Training and testing datasets taken from 10.1039/D2CC01549H.

Sections:
    . Imports
    . Configuration
    . Auxiliary Functions
        . normalize()
        . buildGP()
        . gpPredict()
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
import time

# Specific
import numpy
import pandas
from sklearn import metrics
from sklearn import preprocessing
import gpflow
from matplotlib import pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

# Path to database folder
dbPath=r'/path/to/Databases'
# Database code
code='BP' # 'MM','BP','VP','D_20','RI_20','S_25'
# Define normalization methods
featureNorm=None # None,Standardization,MinMax,LogStand,Log+bStand
labelNorm='Standardization' # None,Standardization,MinMax,LogStand,Log+bStand
# GP Configuration
gpConfig={'kernel':'RBF',
          'useWhiteKernel':True,
          'trainLikelihood':True}

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

# =============================================================================
# Main Script
# =============================================================================

# Iniate timer
ti=time.time()
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
# Get target variable denominator
varName=trainDB.columns[3]
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

# =============================================================================
# Plots
# =============================================================================

# Pyplot Configuration
plt.rcParams['figure.dpi']=600
plt.rcParams['savefig.dpi']=600
plt.rcParams['text.usetex']=False
plt.rcParams['font.family']='serif'
plt.rcParams['font.serif']='Times New Roman'
plt.rcParams['font.weight']='bold'
plt.rcParams['mathtext.rm']='serif'
plt.rcParams['mathtext.it']='serif:italic'
plt.rcParams['mathtext.bf']='serif:bold'
plt.rcParams['mathtext.fontset']='custom'
plt.rcParams['axes.titlesize']=8
plt.rcParams['axes.labelsize']=8
plt.rcParams['xtick.labelsize']=7
plt.rcParams['ytick.labelsize']=7
plt.rcParams['font.size']=8
plt.rcParams["savefig.pad_inches"]=0.02

# Histogram of the target data
print('Training set size: '+str(Y_Train.shape[0]))
print('Training set min: ' +str(Y_Train.min()))
print('Training set max: ' +str(Y_Train.max()))
print('Testing set size: '+str(Y_Test.shape[0]))
print('Testing set min: ' +str(Y_Test.min()))
print('Testing set max: ' +str(Y_Test.max()))
plt.figure(figsize=(2.3,2))
if code=='VP' or code=='S_25':
    plt.hist(Y_Train,
             bins=numpy.exp(numpy.histogram_bin_edges(numpy.log(Y_Train))),
             color='white',edgecolor='red',hatch='.')
    plt.hist(Y_Test,
             bins=numpy.exp(numpy.histogram_bin_edges(numpy.log(Y_Test))),
             color='white',edgecolor='blue',hatch='x')
    plt.xscale('log')
else:
    plt.hist(Y_Train,color='white',edgecolor='red',hatch='.')
    plt.hist(Y_Test,color='white',edgecolor='blue',hatch='x')
plt.yscale('log')
plt.xlabel(varName,weight='bold')
plt.ylabel('Count',weight='bold')
plt.show()

# Predictions Scatter Plot
if code=='VP' or code=='S_25':
    # Compute metrics
    R2_Train=metrics.r2_score(numpy.log(Y_Train),numpy.log(Y_Train_Pred))
    R2_Test=metrics.r2_score(numpy.log(Y_Test),numpy.log(Y_Test_Pred))
    MAE_Train=metrics.mean_absolute_error(Y_Train,Y_Train_Pred)
    MAE_Test=metrics.mean_absolute_error(Y_Test,Y_Test_Pred)
    # Plot
    plt.figure(figsize=(2.3,2))
    plt.loglog(Y_Train,Y_Train_Pred,'ow',markersize=3,mec='red')
    plt.loglog(Y_Test,Y_Test_Pred,'^b',markersize=2)
else:
    # Compute metrics
    R2_Train=metrics.r2_score(Y_Train,Y_Train_Pred)
    R2_Test=metrics.r2_score(Y_Test,Y_Test_Pred)
    MAE_Train=metrics.mean_absolute_error(Y_Train,Y_Train_Pred)
    MAE_Test=metrics.mean_absolute_error(Y_Test,Y_Test_Pred)
    # Plot
    plt.figure(figsize=(2.3,2))
    plt.plot(Y_Train,Y_Train_Pred,'ow',markersize=3,mec='red')
    plt.plot(Y_Test,Y_Test_Pred,'^b',markersize=2)
lims=[numpy.min([plt.gca().get_xlim(),plt.gca().get_ylim()]),
      numpy.max([plt.gca().get_xlim(),plt.gca().get_ylim()])]
plt.axline((lims[0],lims[0]),(lims[1],lims[1]),color='k',
           linestyle='--',linewidth=1)
plt.xlabel('Exp. '+varName,weight='bold')
plt.ylabel('Pred. '+varName,weight='bold')
# Prevent wrong units from showing for RI
if code=='RI_20': varName=varName+' , '
plt.text(0.03,0.93,
         'MAE = '+'{:.2f} '.format(MAE_Train)+varName.split()[-1][1:],
         horizontalalignment='left',
         transform=plt.gca().transAxes,c='r')
plt.text(0.03,0.85,
         'MAE = '+'{:.2f} '.format(MAE_Test)+varName.split()[-1][1:],
         horizontalalignment='left',
         transform=plt.gca().transAxes,c='b')
plt.text(0.97,0.11,'$R^2$ = '+'{:.2f}'.format(R2_Train),
         horizontalalignment='right',
         transform=plt.gca().transAxes,c='r')
plt.text(0.97,0.03,'$R^2$ = '+'{:.2f}'.format(R2_Test),
         horizontalalignment='right',
         transform=plt.gca().transAxes,c='b')
plt.show()

# Print elapsed time
tf=time.time()
print('Time elapsed: '+'{:.2f}'.format(tf-ti)+' s')