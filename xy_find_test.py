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
    return rtxy_path, mask_total

def tester_yx(dicom_path, total_rt_path, output):
    rtstruct = RTStructBuilder.create_from(dicom_path, total_rt_path)
    mask_total_yx = rtstruct.get_roi_mask_by_name("Pancreas")
    mask_total_yx = np.transpose(mask_total_yx, (1, 0, 2))  # yx → xy
    mask_total_yx = mask_total_yx > 0
    rtstruct.add_roi(mask_total_yx, name="yx_suppose") 
    rtyx_path = os.path.join(output, "rt_yx_suppose.dcm")
    rtstruct.save(rtyx_path)
    return rtyx_path, mask_total_yx

def main():
    print('테스트 시작')
    dicom_path = r''         # DICOM 경로 입력
    total_rt_path = r''      # 기존 RTStruct 경로 입력
    output = r''             # 출력 디렉터리 입력
    
    rtxy, mask_total = tester_xy(dicom_path, total_rt_path, output)
    rtyx, mask_total_yx = tester_yx(dicom_path, total_rt_path, output)
    
    print(f'tester_xy의 축: {mask_total.shape}, tester_yx의 축: {mask_total_yx.shape}')

if __name__ == "__main__":
    main()
