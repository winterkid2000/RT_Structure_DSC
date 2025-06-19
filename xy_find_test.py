import numpy as np 
from rt_utils import RTStructBuilder
import os
from tqdm import tqdm
import pandas as pd 
import SimpleITK as sitk
  
def tester_xy(dicom_path, total_rt_path, output):
    rtstruct = RTStructBuilder.create_from(dicom_path, total_rt_path)
    mask_total = rtstruct.get_roi_mask_by_name("Pancreas")
    rtstruct.add_roi(mask_total, name="xy_suppose") 
    rtxy_path = os.path.join(output, "rt_xy_suppose.dcm")
    rtstruct.save(rtxy_path)
    return rtxy_path
def main():
    print('테스트')
    dicom_path = r''
    total_rt_path = r''
    output = r''
    rtxy = tester_xy(dicom_path, total_rt_path, output)
    print(f'tester xy의 축: {mask_total.shape}')
    

if __name__ == "__main__":
  main()
