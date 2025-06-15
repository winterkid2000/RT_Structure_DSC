import numpy as np 
from rt_utils import RTStructBuilder
import os
from tqdm import tqdm
import pandas as pd 
import SimpleITK as sitk

def get_mask(dicom_dir, total_dir, utils_dir):
    rtstruct1 = RTStructBuilder.create_from(dicom_dir, total_dir)
    rtstruct2 = RTStructBuilder.create_from(dicom_dir, utils_dir)

    mask_total = rtstruct1.get_mask_by_name("Pancreas")
    mask_utils = rtstruct2.get_mask_by_name("Pancreas_onco")
    return mask_total, mask_utils

def dice_similarity(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum()
    if union == 0:
        return 0
    return 2 * intersection / union

def jaccard_similarity(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def precision(mask1, mask2):

    tp = np.logical_and(mask1, mask2).sum()
    fp = np.logical_and(np.logical_not(mask1), mask2).sum()

    return tp/(tp+fp) if (tp+fp)>0 else 0

def recall(mask1, mask2):

    tp = np.logical_and(mask1, mask2).sum()
    fn = np.logical_and(mask1, np.logical_not(mask2)).sum()

    return tp/(tp+fn) if (tp+fn) > 0 else 0

def hausdorff(mask1, mask2):

    mask1_sitk = sitk.GetImageFromArray(mask1.astype(np.uint8))
    mask2_sitk = sitk.GetImageFromArray(mask2.astype(np.uint8))

    hd = sitk.HausdorffDistanceImageFilter()
    hd.Execute(mask1_sitk, mask2_sitk)

    return hd.GetHausdorffDistance()

def ASD(mask1, mask2):
    mask1_sitk = sitk.GetImageFromArray(mask1.astype(np.uint8))
    mask2_sitk = sitk.GetImageFromArray(mask2.astype(np.uint8))

    sd = sitk.SurfaceDistanceImageFilter()
    sd.Execute(mask1_sitk, mask2_sitk)

    return sd.GetAverageSymmetricSurfaceDistance()

def hausdorff95(mask1, mask2):

    mask1_sitk = sitk.GetImageFromArray(mask1.astype(np.uint8))
    mask2_sitk = sitk.GetImageFromArray(mask2.astype(np.uint8))

    hd = sitk.HausdorffDistanceImageFilter()
    hd.SetPercentile(95)
    hd.Execute(mask1_sitk, mask2_sitk)

    return hd.GetHausdorffDistance()

def volume_dif(mask1, mask2):

    volume1 = mask1.sum()
    volume2 = mask2.sum()

    return 1-abs(volume1-volume2)/(volume1+volume2) if (volume1+volume2) > 0 else 0

def main():
    print('유사도 계산 시작')
    rt_total_base = input('TotalSegmentator의 경로를 입력하시오: ').strip('"')
    rt_utils_base = input('MIM/Onco의 경로를 입력하시오: ').strip('"')
    dicom_base = input('DICOM의 경로를 입력하시오: ').strip('"')
    output = input('출력 경로를 입력하시오: ').strip('"')

    results = []
    failed_cases = []

    for i in tqdm(range(1, 105), desc = "통계는 너무 힘들어"):
        case_id = str(i)
        dicom_folder = os.path.join(dicom_base, case_id, "PRE")
        rt_total_folder = os.path.join(rt_total_base, case_id)
        rt_utils_folder = os.path.join(rt_utils_base, case_id)
        rt_total_file = [t for t in os.listdir(rt_total_folder) if t.endswith('.dcm')][0]
        rt_utils_file = [u for u in os.listdir(rt_utils_folder) if u.endswith('.dcm')][0]

        rt_total = os.path.join(rt_total_folder, rt_total_file)
        rt_utils = os.path.join(rt_utils_folder, rt_utils_file)

        try:
            mask_total, mask_utils = get_mask(dicom_folder, rt_total, rt_utils)
            dsc = dice_similarity(mask_total, mask_utils)
            jsc = jaccard_similarity(mask_total, mask_utils)
            pre = precision(mask_total, mask_utils)
            rec = recall(mask_total, mask_utils)
            hd = hausdorff(mask_total, mask_utils)
            hd95 = hausdorff95(mask_total, mask_utils)
            asd =ASD(mask_total, mask_utils)
            vol_dif = volume_dif(mask_total, mask_utils)
        except Exception as e:
            failed_cases.append(case_id)
            print(f"{case_id} 실패: {e}")
            continue

        results.append({
            "Number": case_id,
            "Dice_Similarity": dsc,
            "Jaccard_Similarity": jsc,
            "Precision": pre,
            "Recall": rec,
            "Hausdorff_Distance": hd,
            "Hausdorff_Distance_95": hd95,
            "Average_Surface_Distance": asd,
            "Volume_Similarity": vol_dif
        })

    df = pd.DataFrame(results)
    output_path = os.path.join(output, "Total-Onco_statistics.xlsx")
    df.to_excel(output_path, index=False)

    print(f"\n모든 유사도 계산 완료, 실패한 케이스: {failed_cases}")

if __name__ == "__main__":
    main()
