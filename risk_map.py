#!/usr/bin/env python3
import sys
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from nibabel.processing import resample_to_output, resample_from_to

if len(sys.argv) == 4:
    segfile  = sys.argv[1]
    t1file = sys.argv[2] 
    outfile = sys.argv[3] 
else:
    segfile  = "test_case/objects.nii.gz"
    t1file = "test_case/t1.nii.gz" 
    outfile = "test_case/risk_map.nii.gz" 

seg   = resample_to_output(nib.load(segfile),
        voxel_sizes=(1.0, 1.0, 1.0),     
        order=0,                         
        mode="nearest")
seg_data   = seg.get_fdata().astype(np.float32)
affine     = seg.affine
header     = seg.header
m, s = ndimage.label(seg_data)

seg_data = np.zeros(seg_data.shape)
tumour = (seg_data == 2).astype(np.uint8)          
cst    = (seg_data == 1).astype(np.uint8)          

dist_ds_mm = ndimage.distance_transform_edt(
    ~cst.astype(bool),          
    sampling=(1.0, 1.0, 1.0))      

dist_ds_mm[~tumour.astype(bool)] = 0
risk_zones = np.zeros(dist_ds_mm.shape)
risk_zones[dist_ds_mm < 10] = 3
risk_zones[dist_ds_mm < 6] = 2
risk_zones[dist_ds_mm < 3] = 1
risk_zones[dist_ds_mm ==0] = 0

risk_img = nib.Nifti1Image(risk_zones, affine, header)
nib.save(risk_img,outfile)

risk_img_sitk = sitk.ReadImage(outfile) 
ref_img_sitk = sitk.ReadImage(t1file)  

resampled_sitk = sitk.Resample(
    risk_img_sitk,
    ref_img_sitk,
    sitk.Transform(),
    sitk.sitkNearestNeighbor,
    0,
    sitk.sitkUInt8
    )

writer = sitk.ImageFileWriter()
writer.SetFileName(outfile)
writer.Execute(resampled_sitk)
