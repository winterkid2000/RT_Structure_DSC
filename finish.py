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

def mm_unit(dicom_path):

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image.GetSpacing()

def mask_stuffs(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union_dice = mask1.sum() + mask2.sum()
    union_jaccard = np.logical_or(mask1, mask2).sum()

    tp = intersection
    fn = np.logical_and(np.logical_not(mask1), mask2).sum()
    fp = np.logical_and(mask1, np.logical_not(mask2)).sum()

    if union_dice == 0 or union_jaccard == 0 or (tp + fp) == 0 or (tp + fn) == 0:
        return 0, 0, 0, 0, 0

    dice = 2 * intersection / union_dice
    jaccard = intersection / union_jaccard
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    volume1 = mask1.sum()
    volume2 = mask2.sum()
    volume_similarity = 1 - abs(volume1 - volume2) / (volume1 + volume2)

    return dice, jaccard, precision, recall, volume_similarity

def simple_stuffs(mask1, mask2, spacing):

    mask1_sitk = sitk.GetImageFromArray(mask1.astype(np.uint8))
    mask2_sitk = sitk.GetImageFromArray(mask2.astype(np.uint8))
    mask1_sitk.SetSpacing(spacing)
    mask2_sitk.SetSpacing(spacing)

    hd = sitk.HausdorffDistanceImageFilter()
    sd = sitk.SurfaceDistanceImageFilter()
    hd95 = sitk.HausdorffDistanceImageFilter()
    hd95.SetPercentile(95)
    hd.Execute(mask1_sitk, mask2_sitk)
    sd.Execute(mask1_sitk, mask2_sitk)
    hd95.Execute(mask1_sitk, mask2_sitk)

    return hd.GetHausdorffDistance(), sd.GetAverageSymmetricSurfaceDistance(), hd95.GetHausdorffDistance()


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
            spacing = mm_unit(dicom_folder)
            dsc, jsc, pre, rec, vol_dif = mask_stuffs(mask_total, mask_utils)
            hd, asd, hd95 = simple_stuffs(mask_total, mask_utils, spacing)
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
    output_path = os.path.join(output, "Norm", "PRE_Total-Onco_statistics.xlsx")
    df.to_excel(output_path, index=False)

    print(f"\n모든 유사도 계산 완료, 실패한 케이스: {failed_cases}")

if __name__ == "__main__":
    main()
