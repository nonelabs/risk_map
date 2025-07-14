#!/usr/bin/env python3
import sys
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from nibabel.processing import resample_to_output, resample_from_to

t1_file_pn        = "Patients/Patient_PN_T1.nii.gz" 
cst_file_pn       = "Patients/Patient_PN_CST.nii.gz" 
tumor_file_pn     = "Patients/Patient_PN_Tumor.nii.gz"
risk_file_pn      = "Patients/Patient_PN_Risk.nii.gz" 

t1_1x1_file_pn    = "Patients/Patient_PN_T1_1x1.nii.gz" 
cst_1x1_file_pn   = "Patients/Patient_PN_CST_1x1.nii.gz" 
tumor_1x1_file_pn = "Patients/Patient_PN_Tumor_1x1.nii.gz" 
risk_1x1_file_pn  = "Patients/Patient_PN_Risk_1x1.nii.gz" 

patients = [ 1 ]

for p in patients:

    tumor_file = tumor_file_pn.replace("PN",str(p)) 
    cst_file = cst_file_pn.replace("PN",str(p)) 
    t1_file = t1_file_pn.replace("PN",str(p)) 
    risk_file = risk_file_pn.replace("PN",str(p)) 

    tumor_1x1_file = tumor_1x1_file_pn.replace("PN",str(p)) 
    cst_1x1_file = cst_1x1_file_pn.replace("PN",str(p)) 
    t1_1x1_file = t1_1x1_file_pn.replace("PN",str(p)) 
    risk_1x1_file = risk_1x1_file_pn.replace("PN",str(p)) 

    t1 = nib.as_closest_canonical(nib.load(t1_file))
    nib.save(t1,t1_file)
    t1_1x1 = resample_to_output(t1,
            voxel_sizes=(1.0, 1.0, 1.0),     
            order=0,                         
            mode="nearest")
    nib.save(t1_1x1,t1_1x1_file)

    cst = nib.as_closest_canonical(nib.load(cst_file))
    nib.save(cst,cst_file)
    tumor = nib.as_closest_canonical(nib.load(tumor_file))
    nib.save(tumor,tumor_file)
    t1_1x1 = sitk.ReadImage(t1_1x1_file)
    tumor = sitk.ReadImage(tumor_file)
    cst = sitk.ReadImage(cst_file)
    
    cst_1x1 = sitk.Resample(
        cst,
        t1_1x1,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
        )

    tumor_1x1 = sitk.Resample(
        tumor,
        t1_1x1,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
        )

    writer = sitk.ImageFileWriter()
    writer.SetFileName(cst_1x1_file)
    writer.Execute(cst_1x1)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(tumor_1x1_file)
    writer.Execute(tumor_1x1)

    tumor = nib.load(tumor_1x1_file)
    cst = nib.load(cst_1x1_file)

    affine     = tumor.affine
    header = tumor.header
    cst_data = cst.get_fdata().astype(np.uint8)
    cst_data[ cst_data != 0 ] = 1
    tumor_data = tumor.get_fdata().astype(np.uint8)
    tumor_data[ tumor_data != 0 ] = 1
    print(cst_data.shape)
    print(tumor_data.shape)


    dist_ds_mm = ndimage.distance_transform_edt(
        ~cst_data.astype(bool),          
        sampling=(1.0, 1.0, 1.0))      
    dist_ds_mm[~tumor_data.astype(bool)] = 0
    nib.save(nib.Nifti1Image(dist_ds_mm,affine,header),"dist.nii.gz")

    risk_data = np.zeros(dist_ds_mm.shape)
    risk_data[dist_ds_mm < 10] = 3
    risk_data[dist_ds_mm < 6] = 2
    risk_data[dist_ds_mm < 3] = 1
    risk_data[dist_ds_mm ==0] = 0
    risk_data[(cst_data != 0) & (risk_data==0)] = 4
    risk_data[(tumor_data != 0) & (risk_data==0)] = 5

    risk = nib.Nifti1Image(risk_data, affine, header)
    nib.save(risk,risk_file)
    risk = sitk.ReadImage(risk_file)  
    ref= sitk.ReadImage(t1_file)  

    risk = sitk.Resample(
        risk,
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
        )

    writer = sitk.ImageFileWriter()
    writer.SetFileName(risk_file)
    writer.Execute(risk)
    nib.save(nib.as_closest_canonical(nib.load(risk_file)),risk_file)
