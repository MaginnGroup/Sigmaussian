# -*- coding: utf-8 -*-
"""
Script to characterize and visualize the sigma profile space.

Sections:
    . Imports
    . Configuration
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

# Specific
import numpy
from numpy import matlib
import pandas
from sklearn import decomposition
from matplotlib import pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

# Path to database folder
dbPath=r'/path/to/Databases'


# =============================================================================
# Main Script
# =============================================================================

# Path to CSV file
DB_Path=os.path.join(dbPath,'MM_mlDatabase_Original.csv')
# Load CSV file
DB=pandas.read_csv(DB_Path)
# Generate sigma axis
sigma=numpy.linspace(-0.025,0.025,51)
# Get sigma profiles of water, methane, and propane
SP_water=DB[DB['Index']==1076].iloc[0,4:-1].to_numpy('double')
SP_methane=DB[DB['Index']==1].iloc[0,4:-1].to_numpy('double')
SP_propane=DB[DB['Index']==3].iloc[0,4:-1].to_numpy('double')
# Convert DB to SP matrix
SP_Matrix=DB.iloc[:,4:-1].to_numpy('double')
# Perform PCA
PCA_Matrix=decomposition.PCA(n_components=2,
                             svd_solver='full').fit_transform(SP_Matrix)

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
plt.rcParams['font.size']=6
plt.rcParams["savefig.pad_inches"]=0.02

# Plot water and methane
plt.figure(figsize=(4.5,2.7))
plt.plot(sigma,SP_water,'--ok',label='Water',markersize=3,linewidth=1)
plt.plot(sigma,SP_methane,'--^r',label='Methane',markersize=3,linewidth=1)
plt.plot(sigma,SP_propane,'--*b',label='Propane',markersize=3,linewidth=1)
plt.xlabel(r'$\rm\sigma$ $\rm/e\cdotÅ^{2}$')
plt.ylabel(r'$\rm P(\sigma) \cdot A$ $\rm/Å^{2}$')
plt.legend()
plt.show()

# Plot Sigma Profile space
plt.figure(figsize=(2,2.5))
ssigma=matlib.repmat(sigma.reshape(1,-1),SP_Matrix.shape[0],1)
plt.plot(ssigma.T,SP_Matrix.T,'--.',markersize=3)
plt.xlabel(r'$\rm\sigma$ $\rm/e\cdotÅ^{2}$')
plt.ylabel(r'$\rm P(\sigma) \cdot A$ $\rm/Å^{2}$')
plt.show()

# Plot Sigma Profile space
plt.figure(figsize=(3.2,2.5))
ssigma=matlib.repmat(sigma.reshape(1,-1),SP_Matrix.shape[0],1)
plt.semilogy(ssigma,SP_Matrix,'.k',markersize=3)
# Highlight Polyols
DB_indexes=[547,550,552,553,554,555,556,557,558,559,560,561,562,563,564,565,
            566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,
            1172]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.semilogy(ssigma[index,:].reshape(len(index),51),
             SP_Matrix[index,:].reshape(len(index),51),'sy',markersize=1,
             label='Polyols')
# Highlight CHF compounds
DB_indexes=[844,845,846,847,848,849,850,851,852,854,857,879,1022,1026,1264,
            1265,1266,1267,1268,1269,1275,1281,1283,1284,1285,1300,1303]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.semilogy(ssigma[index,:].reshape(len(index),51),
             SP_Matrix[index,:].reshape(len(index),51),'sr',markersize=1,
             label='CHFs')
# Highlight n-Alkanes
DB_indexes=[1,2,3,5,6,9,14,23,41,51,54,55,56,57,58,59,60,61,62,64,65,66,67,68,
            69,70,71]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.semilogy(ssigma[index,:].reshape(len(index),51),
             SP_Matrix[index,:].reshape(len(index),51),'sb',markersize=1,
             label='n-Alkanes')
# Highlight n-Alcohols
DB_indexes=[477,478,479,481,485,490,500,504,506,508,509,511,512,513,514,515,
            516,517,519,520]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.semilogy(ssigma[index,:].reshape(len(index),51),
             SP_Matrix[index,:].reshape(len(index),51),'sg',markersize=1,
             label='n-Alcohols')
# Highlight Terpenes
DB_indexes=[195,196,214,215,369,370]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.semilogy(ssigma[index,:].reshape(len(index),51),
             SP_Matrix[index,:].reshape(len(index),51),'sm',markersize=1,
             label='Terpenes')
# Highlight Alkylcyclohexanes
DB_indexes=[100,101,102,103,104,105,106,107,108,109,110,111,112,113,117,118,
            1726,1727,1728]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.semilogy(ssigma[index,:].reshape(len(index),51),
             SP_Matrix[index,:].reshape(len(index),51),'sc',markersize=1,
             label='ACH')
# Highlight water
index=DB[DB['Index']==1076].index
plt.semilogy(ssigma[index,:].reshape(len(index),51),
             SP_Matrix[index,:].reshape(len(index),51),'^w',mec='blue',
             label='Water',markersize=3)
# Highlight squalane
index=DB[DB['Index']==50].index
plt.semilogy(ssigma[index,:].reshape(len(index),51),
             SP_Matrix[index,:].reshape(len(index),51),'^w',mec='red',
             label='Squalane',markersize=3)
# Highlight TETRAETHYLENE-GLYCOL-DIMETHYL-ET
index=DB[DB['Index']==752].index
plt.semilogy(ssigma[index,:].reshape(len(index),51),
             SP_Matrix[index,:].reshape(len(index),51),'^w',mec='c',
             label='TGDE',markersize=3)
plt.xlabel(r'$\rm\sigma$ $\rm/e\cdotÅ^{2}$')
plt.ylabel(r'$\rm P(\sigma) \cdot A$ $\rm/Å^{2}$')
plt.show()

# Plot PCA analysis
plt.figure(figsize=(3.2,2.5))
plt.plot(PCA_Matrix[:,0],PCA_Matrix[:,1],'.k',markersize=3)
# Highlight n-Alkanes
DB_indexes=[1,2,3,5,6,9,14,23,41,51,54,55,56,57,58,59,60,61,62,64,65,66,67,68,
            69,70,71]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.plot(PCA_Matrix[index,0],PCA_Matrix[index,1],'sb',markersize=2,
         label='n-Alkanes')
# Highlight n-Alcohols
DB_indexes=[477,478,479,481,485,490,500,504,506,508,509,511,512,513,514,515,
            516,517,519,520]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.plot(PCA_Matrix[index,0],PCA_Matrix[index,1],'sg',markersize=2,
         label='n-Alcohols')
# Highlight Terpenes
DB_indexes=[195,196,214,215,369,370]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.plot(PCA_Matrix[index,0],PCA_Matrix[index,1],'sm',markersize=2,
         label='Terpenes')
# Highlight Polyols
DB_indexes=[547,550,552,553,554,555,556,557,558,559,560,561,562,563,564,565,
            566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,
            1172]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.plot(PCA_Matrix[index,0],PCA_Matrix[index,1],'sy',markersize=2,
         label='Polyols')
# Highlight CHF compounds
DB_indexes=[844,845,846,847,848,849,850,851,852,854,857,879,1022,1026,1264,
            1265,1266,1267,1268,1269,1275,1281,1283,1284,1285,1300,1303]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.plot(PCA_Matrix[index,0],PCA_Matrix[index,1],'sr',markersize=2,
         label='CHFs')
# Highlight Alkylcyclohexanes
DB_indexes=[100,101,102,103,104,105,106,107,108,109,110,111,112,113,117,118,
            1726,1727,1728]
index=[]
for DB_index in DB_indexes: index.append(DB[DB['Index']==DB_index].index)
plt.plot(PCA_Matrix[index,0],PCA_Matrix[index,1],'sc',markersize=2,
         label='ACHs')
# Highlight water
index=DB[DB['Index']==1076].index
plt.plot(PCA_Matrix[index,0],PCA_Matrix[index,1],'^w',mec='blue',
         label='Water',markersize=3)
# Highlight squalane
index=DB[DB['Index']==50].index
plt.plot(PCA_Matrix[index,0],PCA_Matrix[index,1],'^w',mec='red',
         label='Squalane',markersize=3)
# Highlight TETRAETHYLENE-GLYCOL-DIMETHYL-ET
index=DB[DB['Index']==752].index
plt.plot(PCA_Matrix[index,0],PCA_Matrix[index,1],'^w',mec='c',
         label='TGDE',markersize=3)
plt.xlabel('PCA Var. #1',weight='bold')
plt.ylabel('PCA Var. #2',weight='bold')
plt.legend()
plt.show()
