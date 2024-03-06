# -*- coding: utf-8 -*-
"""
Script to perform gradient search on the sigma profile space.

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

# Specific
import numpy
import pandas
from sklearn import preprocessing
from sklearn import decomposition
import gpflow
import tensorflow
from tqdm import tqdm
from matplotlib import pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

# Path to database folder
dbPath=r'/path/to/Databases'
# Database code
code='BP' # 'MM','BP','VP','D_20','RI_20','S_25'
# GP Configuration
gpConfig={'kernel':'RBF',
          'useWhiteKernel':True,
          'trainLikelihood':True}
# Gradient Descent Configuration
VT2005_Index=387
direction='Max' # 'Max' or 'Min'
patience=10
maxIter=100
polarityOpt=None # None,'+','-','0'

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
# Main Script (Part 1)
# =============================================================================

# Path to CSV files
fullDB_Path=os.path.join(dbPath,'MM_mlDatabase.csv')
DB_Path=os.path.join(dbPath,code+'_mlDatabase_Original.csv')
trainDB_Path=os.path.join(dbPath,code+'_mlDatabase.csv')
testDB_Path=os.path.join(dbPath,code+'_mlDatabase_TestSet.csv')
# Load CSV file
fullDB=pandas.read_csv(fullDB_Path)
DB=pandas.read_csv(DB_Path)
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
X_Train_N=X_Train
X_Test_N=X_Test
if code=='VP' or code=='S_25': labelNorm='LogStand'
else: labelNorm='Standardization'
Y_Train_N,skScaler_Y_Train=normalize(Y_Train,method=labelNorm)
# Train GP
model=buildGP(X_Train_N,Y_Train_N,gpConfig=gpConfig)

# =============================================================================
# Main Script (Part 2)
# =============================================================================

# Define available sigma profiles
X_Avail=DB.iloc[:,4:-1].to_numpy()
# Define fragment sigma profile
fragment=numpy.zeros((1,51))
if polarityOpt=='+': fragment[0,:17]=1
elif polarityOpt=='-': fragment[0,17+17:]=1
elif polarityOpt=='0': fragment[0,17:17+17]=1
else: fragment=numpy.ones((1,51))
# Get optimization direction
if direction=='Max': mult=1
elif direction=='Min': mult=-1
# Get initial molecular structure
if not DB[DB['Index']==VT2005_Index].empty:
    initial_SP=DB[DB['Index']==VT2005_Index].iloc[0:1,4:-1].to_numpy('double')
else:
    raise ValueError('Could not find requested initial molecule in dataset.')
# Convert to tensorflow tensor
current_SP=tensorflow.convert_to_tensor(initial_SP)
# Initialize containers
gradY_norm_hist=[]
Y_hist=[]
SP_hist=numpy.empty((0,51))
improvement=0
itImp=0
# Optimization loop
for n in tqdm(range(maxIter),'Gradient Descent: '):
    # Compute improvement from previous iteration
    if n>1:
        if mult==1: improvement=Y_hist[n-1]-max(Y_hist[:n-1])
        elif mult==-1: improvement=Y_hist[n-1]-min(Y_hist[:n-1])
        # Check itImp
        if mult*improvement<=0: itImp+=1
        else: itImp=0
    # Compute gradients for current_SP
    with tensorflow.GradientTape() as tape:
        # Set current_SP as domain variable
        tape.watch(current_SP)
        # Perform GP prediction
        GP_Mean,GP_Var=model.predict_f(current_SP)
    # Retrieve gradients
    gradY=tape.gradient(GP_Mean,current_SP)
    # Define propagator
    propagator=mult*gradY*fragment
    # Get propagator norm
    gradY_norm=tensorflow.norm(propagator).numpy()
    # Save history
    gradY_norm_hist.append(gradY_norm)
    Y_hist.append(GP_Mean.numpy()[0,0])
    SP_hist=numpy.append(SP_hist,current_SP.numpy().copy(),axis=0)
    # Propagate
    centerPoint=X_Avail-current_SP
    dot=numpy.sum(propagator.numpy()*centerPoint.numpy(),axis=1)
    den=numpy.linalg.norm(propagator.numpy(),axis=1)\
        *numpy.linalg.norm(centerPoint.numpy(),axis=1)
    similarity=dot/den
    index=numpy.nanargmax(similarity)
    current_SP=tensorflow.convert_to_tensor(
        X_Avail[index,:].copy().reshape(1,-1))
    # Check convergence
    if itImp>=patience: break    
# Get best hit
if mult==1:
    X_Best=SP_hist[Y_hist.index(max(Y_hist)),:].reshape(1,-1)
    Y_Best=max(Y_hist)
elif mult==-1:
    X_Best=SP_hist[Y_hist.index(min(Y_hist)),:].reshape(1,-1)
    Y_Best=min(Y_hist)
index=numpy.linalg.norm(X_Avail-X_Best,axis=1).argmin()
# Print
print('\nInitial molecule name: '+DB[DB['Index']==VT2005_Index].iloc[0,1])
print('Initial molecule VT-2005 index: '+str(
    DB[DB['Index']==VT2005_Index].iloc[0,0]))
print('Initial molecule property: '+\
      str(DB[DB['Index']==VT2005_Index].iloc[0,3])+' '+varName.split()[-1])
print('Optimized molecule name: '+DB.iloc[index,1])
print('Optimized molecule VT-2005 index: '+str(DB.iloc[index,0]))
print('Optimized molecule property: '+str(DB.iloc[index,3])\
      +' '+varName.split()[-1])

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
plt.rcParams['axes.titlesize']=9
plt.rcParams['axes.labelsize']=9
plt.rcParams['xtick.labelsize']=8
plt.rcParams['ytick.labelsize']=8
plt.rcParams['font.size']=9
plt.rcParams["savefig.pad_inches"]=0.02

# Convergence
Y_hist,__=normalize(numpy.array(Y_hist).reshape(-1,1),
                              skScaler=skScaler_Y_Train,
                              method=labelNorm,
                              reverse=True)
fig,ax1=plt.subplots(figsize=(2.3,2))
ax1.semilogy(gradY_norm_hist,'--r')
ax2=ax1.twinx()
ax2.semilogy(Y_hist,'--b')
ax1.set_xlabel('Iteration',weight='bold')
ax1.set_ylabel('Gradient Norm',weight='bold')
ax2.set_ylabel('Pred. '+varName,weight='bold')
plt.show()

# Sigma Profiles
plt.figure(figsize=(3.2,2.5))
sigma=numpy.linspace(-0.025,0.025,51)
plt.plot(sigma,SP_hist[0,:],'--om',label='Initial',markersize=3,linewidth=1)
plt.plot(sigma,X_Best[0,:],'--^g',label='Final',markersize=3,linewidth=1)
plt.xlabel(r'$\rm\sigma$ $\rm/e\cdotÅ^{2}$')
plt.ylabel(r'$\rm P(\sigma) \cdot A$ $\rm/Å^{2}$')
plt.legend()
plt.show()

# PCI space
# Convert fullDB to SP matrix
SP_Matrix=fullDB.iloc[:,4:-1].to_numpy('double')
# Perform PCA
PCA=decomposition.PCA(n_components=2,svd_solver='full').fit(SP_Matrix)
SP_Matrix_PCA=PCA.transform(SP_Matrix)
SP_hist_PCA=PCA.transform(SP_hist)
SP_start_PCA=PCA.transform(SP_hist[0,:].reshape(1,-1))
SP_end_PCA=PCA.transform(X_Best[0,:].reshape(1,-1))
# # Make plot
plt.figure(figsize=(3.2,2.5))
plt.plot(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],'.k',markersize=3)
# Highlight optimization path
plt.plot(SP_hist_PCA[:,0],SP_hist_PCA[:,1],'x-r',label='Opt. Path',
         linewidth=1,markersize=3)
plt.plot(SP_start_PCA[:,0],SP_start_PCA[:,1],'sm',label='Start',markersize=4)
plt.plot(SP_end_PCA[:,0],SP_end_PCA[:,1],'sg',label='End',markersize=4)
plt.xlabel('PCA Var. #1',weight='bold')
plt.ylabel('PCA Var. #2',weight='bold')
plt.show()