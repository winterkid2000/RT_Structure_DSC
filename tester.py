import numpy as np 
from rt_utils import RTStructBuilder
import os
from tqdm import tqdm
import pandas as pd 
import SimpleITK as sitk

def load_dicom_image(dicom_path):

    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_files)
    return reader.Execute()

def get_mask_and_ref(dicom_path, total_rt_path):
    ref_image = load_dicom_image(dicom_path)
    rt_total = RTStructBuilder.create_from(dicom_dir, total_rt_path)
    
    mask_total = rt_total.get_mask_by_name("Pancreas")
    
    return mask_total, ref_image
  
def tester(mask_total, ref_image):
    mask_coord = mask_total.shape
    ref_image_coord = ref_image.GetSize()
    return mask_coord, ref_image_coord
def tester2(mask_total, ref_image): ##roi mask가 yxz인지 xyz인지 확인, numpy가 zyx, ref_image의 좌표가 xyz라는 가정
    sitk1 = sitk.GetImageFromArray(np.transpose(mask_total, (2, 0, 1)).astype(np.uint8))
    sitk1.CopyInformation(ref_image) ##yxz
    yxz = sitk1.GetSize()

    sitk2 = sitk.GetImageFromArray(np.transpose(mask_total, (2, 1, 0)).astype(np.uint8))
    sitk2.CopyInformation(ref_image) ##xyz
    xyz = sitk2.GetSize()
    return yxz, xyz

def main():
    print('테스트')
    dicom_path = r''
    total_rt_path = r''
    mask_total, ref_image = get_mask_and_ref(dicom_dir, total_rt_path)
    mask_coord, ref_image_coord = tester(mask_total, ref_image)
    print(f'마스크: {mask_coord}, CT: {ref_image_coord}')
    yxz, xyz = tester2(mask_total, ref_image)
    print(f'yxz: {yxz}, xyz: {xyz}') 

if __name__ == "__main__":
  main()
