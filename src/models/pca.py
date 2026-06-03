# src/models/pca.py
"""
Functions to perform PCA — temporal and spatial utilities.
"""

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import nibabel as nib

# PCA 1: each 3D image as a sample (how voxels co-vary over time, temporal dynamics) -> 30 lines, >100 000 columns (dimensions).
def pca1_transpose(data_array, print_infos=True):
    """
    From 4D numpy array to a 2D aarray (30, >100000)
    """
    data_transposed = np.transpose(data_array, (3, 0, 1, 2))
    X = data_transposed.reshape(data_array.shape[3], -1)
    if print_infos:
        print("Shape of X:", X.shape)  # Should be (30, >100000)
    return X

def pca1_reformat(X_reconstructed, data_array, nii_obj, patient_name_1, n_pc_toreconstruct, save=True):
    """
    """
    X_reconstructed_3d = X_reconstructed.reshape(data_array.shape[-1], *data_array.shape[:-1]) # Shape of the initial 4D data, but with epochs as first dimension
    X_reconstructed_4d = np.transpose(X_reconstructed_3d, (1, 2, 3, 0)) # put epochs as last dimension
    nii_reconstructed = nib.Nifti1Image(X_reconstructed_4d, nii_obj.affine, nii_obj.header)
    if save:
        nib.save(nii_reconstructed, RESULTS_FOLDER / f"{patient_name_1}_projected_{n_pc_toreconstruct}_4d.nii.gz")
    return nii_reconstructed

def pca_clean(X):
    """
    Use StandardScaler not to introduce NaNs with divisions with variances = 0
    """
    # Clean data to remove constant features
    selector = VarianceThreshold() 
    X_filtered = selector.fit_transform(X)
     # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    return X_scaled

def eigenvector_to_nii(vec, shape_3d, nii_ref):
    """Reshape a 1D eigenvector into a 3D NIfTI using the reference affine/header."""
    return nib.Nifti1Image(vec.reshape(shape_3d), nii_ref.affine, nii_ref.header)


def pca1_reconstruct(X_reduced, pca, n_pc, data_array, nii_obj):
    """
    Reconstruct a 4D NIfTI from the n first temporal principal components.

    Parameters
    ----------
    X_reduced  : np.ndarray, shape (n_epochs, n_components)
    pca        : fitted sklearn PCA
    n_pc       : int — number of components to use for reconstruction
    data_array : np.ndarray, shape (x, y, z, t) — original 4D array (for shape)
    nii_obj    : nibabel NIfTI — for affine/header

    Returns
    -------
    Nifti1Image (4D)
    """
    X_rec = X_reduced[:, :n_pc] @ pca.components_[:n_pc, :] + pca.mean_
    n_epochs = data_array.shape[3]
    X_rec_3d = X_rec.reshape(n_epochs, *data_array.shape[:3])
    X_rec_4d = np.transpose(X_rec_3d, (1, 2, 3, 0))
    return nib.Nifti1Image(X_rec_4d, nii_obj.affine, nii_obj.header)


def pca_spatial_reconstruct(X_pca_row, pca, n_pc):
    """
    Reconstruct one flat 3D image from its spatial PCA coordinates.

    Parameters
    ----------
    X_pca_row : np.ndarray, shape (n_components,)
        PCA coordinates for one patient (one row of X_pca).
    pca : fitted sklearn PCA
    n_pc : int
        Number of components to use for reconstruction.

    Returns
    -------
    np.ndarray, shape (n_voxels,)
        Reconstructed flat image. Reshape to 3D and pass to eigenvector_to_nii
        to get a NIfTI.
    """
    return X_pca_row[:n_pc] @ pca.components_[:n_pc, :] + pca.mean_


# PCA2 : spatial

def pca2_reformat(X_reconstructed, data_array, nii_obj_template, patient_index):
    """
    """
    # Reconstruct 4D (all patients)
    n_patients = data_array.shape[0]
    spatial_shape = data_array.shape[1:]  # (256,256,10)
    img4d = X_reconstructed.reshape(n_patients, *spatial_shape)  # (n,256,256,10)
    # Get image 3D of the patient chosen
    img3d = img4d[patient_index]  # (256,256,10)
    nii = nib.Nifti1Image(img3d, nii_obj_template.affine, nii_obj_template.header)
    return nii
