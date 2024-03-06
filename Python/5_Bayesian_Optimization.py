# -*- coding: utf-8 -*-
"""
Script to perform Bayesian Optimization on the sigma profile space towards the
maximization/minimization of a physicochemical property.

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
import scipy
from sklearn import decomposition
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
# GP Config
gpConfig={'kernel':'RBF',
          'useWhiteKernel':True,
          'trainLikelihood':True}
# BO Config
VT2005_Index=387
AF_max_stop=10**-3
maxIter=100
# GIF Config
doGIF=False # True or False (Requires RDKit)
pathGIF=r'/path/to/Movie.gif'
tempPath=r'/path/to/_temp'

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
        gpKernel=gpflow.kernels.SquaredExponential(variance=7.62,
                                                   lengthscales=37.58)
    if kernel=='RQ':
        gpKernel=gpflow.kernels.RationalQuadratic()
    if kernel=='Matern32':
        gpKernel=gpflow.kernels.Matern32()
    if kernel=='Matern52':
        gpKernel=gpflow.kernels.Matern52()
    # Add White kernel
    if useWhiteKernel: gpKernel=gpKernel+gpflow.kernels.White(variance=0.036)
    # Build GP model    
    model=gpflow.models.GPR((X_Train,Y_Train),gpKernel,
                            noise_variance=1.01*10**-5)
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
    GP_Mean,GP_Var=model.predict_y(X)
    # Convert to numpy
    GP_Mean=GP_Mean.numpy()
    GP_Var=GP_Var.numpy()
    # Prepare outputs
    Y=GP_Mean
    STD=numpy.sqrt(GP_Var)
    # Output
    return Y,STD

def plotIteration(fullDB,X_BO,savePath):
    
    # Load SMILES string info
    smilesDB=pandas.read_csv(os.path.join(dbPath,'spDatabase.csv'))
    # Get SMILES of current iteration
    smilesIndex=numpy.abs(smilesDB.iloc[:,2:].to_numpy()\
                          -X_BO[-1,:].reshape(1,-1)).sum(axis=1).argmin()
    SMILES=smilesDB.iloc[smilesIndex,1]
    # Generate RDKit mol object
    molecule=Chem.MolFromSmiles(SMILES)
    molecule=AllChem.AddHs(molecule)    
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
    # Convert fullDB to SP matrix
    SP_Matrix=fullDB.iloc[:,4:-1].to_numpy('double')
    # Perform PCA
    PCA=decomposition.PCA(n_components=2,svd_solver='full').fit(SP_Matrix)
    SP_Matrix_PCA=PCA.transform(SP_Matrix)
    SP_hist_PCA=PCA.transform(X_BO)
    SP_start_PCA=PCA.transform(X_BO[0,:].reshape(1,-1))
    SP_end_PCA=PCA.transform(X_BO[-1,:].reshape(1,-1))
    # Plots
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.4,2.5))
    ax1.plot(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],'.k',markersize=3)
    # Highlight optimization path
    ax1.plot(SP_hist_PCA[:,0],SP_hist_PCA[:,1],'x-r',label='Opt. Path',
             linewidth=1,markersize=3)
    ax1.plot(SP_start_PCA[:,0],SP_start_PCA[:,1],'sm',label='Start',
             markersize=4)
    ax1.plot(SP_end_PCA[:,0],SP_end_PCA[:,1],'sg',label='End',markersize=4)
    ax1.set_xlabel('PCA Var. #1',weight='bold')
    ax1.set_ylabel('PCA Var. #2',weight='bold')
    ax2.set_axis_off()
    ax2.set_position([0.2, 0, 1, 1], which='both')
    ax2.imshow(Draw.MolToImage(molecule))
    fig.savefig(savePath,bbox_inches='tight')
    # Output
    return None

def getGIF(gifPath,frameList,duration=0.1):
    """
    getGIF() generates a GIF from the image paths provided in frameList.

    Parameters
    ----------
    gifPath : string
        Path where the GIF should be saved.
    frameList : list of strings
        List containing the path of each frame (properly ordered).
    duration : int, optional
        Duration of each frame (seconds) in the final GIF.
        The default is 1.

    Returns
    -------
    None.

    """
    # Initialize list of frames
    frames=[] 
    # Iterate over individual snapshots
    for file in tqdm(frameList,'Generating GIF'):
        # Read image and append to frames
        frames.append(imageio.v2.imread(file,format='png'))
    # Save list of frames as GIF
    imageio.mimsave(gifPath,frames,duration=duration)
    # Output
    return None

# =============================================================================
# Main Script
# =============================================================================

# Check doGIF
if doGIF:
    # Additional imports
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
    import imageio
    import secrets
    from tqdm import tqdm
    # Define temp name
    tempName=secrets.token_hex(5)
    # Initialize pathList
    pathList=[]
# Check label normalization
if code=='VP' or code=='S_25': labelNorm='LogStand'
else: labelNorm='Standardization'
# Path to CSV files
fullDB_Path=os.path.join(dbPath,'MM_mlDatabase.csv')
DB_Path=os.path.join(dbPath,code+'_mlDatabase_Original.csv')
# Load CSV files
fullDB=pandas.read_csv(fullDB_Path)
DB=pandas.read_csv(DB_Path)
# Extract sigma profiles and target variable as numpy arrays available for BO
X_Avail=DB.iloc[:,4:-1].to_numpy()
Y_Avail=DB.iloc[:,3].to_numpy().reshape(-1,1)
# Get target variable denominator
varName=DB.columns[3]
# Define intial BO data point
if not DB[DB['Index']==VT2005_Index].empty:
    index=DB[DB['Index']==VT2005_Index].index
else:
    raise ValueError('Could not find requested initial molecule in dataset.')
# Define intial BO data point
X_BO=X_Avail[index,:].copy().reshape(1,-1)
Y_BO=Y_Avail[index,:].copy().reshape(1,-1)
# Delete initial data point from training set
X_Avail=numpy.delete(X_Avail,index,axis=0)
Y_Avail=numpy.delete(Y_Avail,index,axis=0)
# Define standard distribution
standard=scipy.stats.norm(loc=0,scale=1)
# Define history containers
AF_max_hist=[]
Target_hist=[]
# Initiate main BO loop
for n in range(maxIter):
    # Check doGIF
    if doGIF:
        savePath=os.path.join(tempPath,tempName+'_'+str(n)+'.png')
        pathList.append(savePath)
        plotIteration(fullDB,X_BO,savePath)
    # Normalize Y_BO
    Y_BO_N,skScaler_Y=normalize(Y_BO.copy(),method=labelNorm)
    # Train GP
    model=buildGP(X_BO,Y_BO_N,gpConfig=gpConfig)
    # Get GP predictions for X_Avail
    Y_Avail_Pred_N,STD=gpPredict(model,X_Avail)
    # Get best so far
    best=Y_BO_N.max()
    # Compute delta
    delta=Y_Avail_Pred_N-best
    # Compute AF
    AF=delta*standard.cdf(delta/STD)+STD*standard.pdf(delta/STD)
    # Compute AF mean and max
    AF_max=AF.max()
    index=AF.argmax()
    # Unnormalize
    best_UN,__=normalize(best.reshape(1,1),skScaler=skScaler_Y,
                         method=labelNorm,reverse=True)
    # Print iteration details
    print('\n\nIteration #'+str(n))
    print('\n   - Current AF max: '+'{:.2f}'.format(AF_max))
    print('\n   - Current Best: '+'{:.2f}'.format(best_UN[0,0]))
    # Update history
    AF_max_hist.append(AF_max)
    Target_hist.append(best_UN[0,0])
    # Check stopping criterion
    if (AF_max<AF_max_stop and n>0) or n==maxIter-1: break
    # Add new training point to X_BO and Y_BO
    X_BO=numpy.append(X_BO,X_Avail[index,:].reshape(1,-1).copy(),axis=0)
    Y_BO=numpy.append(Y_BO,Y_Avail[index,:].reshape(1,-1).copy(),axis=0)
    # Delete new training point from X_Train and Y_Train
    X_Avail=numpy.delete(X_Avail,index,axis=0)
    Y_Avail=numpy.delete(Y_Avail,index,axis=0)
# Check doGIF
if doGIF:
    getGIF(pathGIF,pathList,duration=1)
    for path in pathList: os.remove(path)
# Get best hit
X_Best=X_BO[Y_BO.argmax(),:].reshape(1,-1)
Y_Best=Y_BO.max().reshape(1,1)
index=numpy.linalg.norm(DB.iloc[:,4:-1].to_numpy('double')\
                        -X_Best,axis=1).argmin()
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
plt.rcParams['axes.titlesize']=8
plt.rcParams['axes.labelsize']=8
plt.rcParams['xtick.labelsize']=7
plt.rcParams['ytick.labelsize']=7
plt.rcParams['font.size']=8
plt.rcParams["savefig.pad_inches"]=0.02

# Convergence
fig,ax1=plt.subplots(figsize=(2.3,2))
ax1.semilogy(list(range(1,len(AF_max_hist))),AF_max_hist[1:],
                        '--r',label='Max AF',linewidth=1)
ax2=ax1.twinx()
ax2.semilogy(list(range(1,len(Target_hist))),Target_hist[1:],
         '--b',label='Test R2',linewidth=1)
ax1.set_xlabel('BO Iteration',weight='bold')
ax1.set_ylabel('Max. Expected Improvement',weight='bold')
ax2.set_ylabel('Pred. '+varName,weight='bold')
plt.show()

# PCI space
# Generate sigma axis
sigma=numpy.linspace(-0.025,0.025,51)
# Convert fullDB to SP matrix
SP_Matrix=fullDB.iloc[:,4:-1].to_numpy('double')
# Perform PCA
PCA=decomposition.PCA(n_components=2,svd_solver='full').fit(SP_Matrix)
SP_Matrix_PCA=PCA.transform(SP_Matrix)
SP_hist_PCA=PCA.transform(X_BO)
SP_start_PCA=PCA.transform(X_BO[0,:].reshape(1,-1))
SP_end_PCA=PCA.transform(X_Best)
# Make plot
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
