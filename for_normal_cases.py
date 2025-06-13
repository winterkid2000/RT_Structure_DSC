import numpy as np 
from rt_utils import RTStructureBuilder
import os
import 
from tqdm import tqdm

def dice_similarity(total_root, utilised_root):
    
    total_root

    ##여기서부턴 먼저 total_root랑 utilised_root 경로부터 정의해줘야 함 
  ##아니면 그냥 main에서 경로 다 지정하니까 roi_mask 빼는 것만? 

  rtstruct = RTStructBuilder.create_from(dicom_series_path, rtstruct_path)

roi_name1 = "ROI1"
roi_name2 = "ROI2"

roi1_mask = rtstruct.get_roi_mask_by_name(roi_name1)
roi2_mask = rtstruct.get_roi_mask_by_name(roi_name2)

    intersection = np.logical_and(mask_total, mask_utils).sum()
    union = mask_total.sum()+mask_utils.sum()

    if union == 0:
        return 0
    return (2*intersection)/union

def main():
    print('Dice Similarity Coefficient 계산')
    rt_total_base = input('TotalSegmentator의 경로를 입력하시오: ')
    rt_utils_base = input('MIM/Onco의 경로를 입력하시오: ')
    dicom__base = input('DICOM의 경로를 입력하시오: ')

    for i in range
    case_id = {i}
    

    for i in tqdm(number, desc = 'Dice Similarity Coefficient 계산 중!'):



if __name__ == "main":
    main()

  
