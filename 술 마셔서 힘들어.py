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

def get_mask_and_ref(dicom_dir, total_rt_path, utils_rt_path):
    ref_image = load_dicom_image(dicom_dir)
    rt_total = RTStructBuilder.create_from(dicom_dir, total_rt_path)
    rt_utils = RTStructBuilder.create_from(dicom_dir, utils_rt_path)
    
    mask_total = rt_total.get_mask_by_name("Pancreas")
    mask_utils = rt_utils.get_mask_by_name("Pancreas_onco")
    
    return mask_total, mask_utils, ref_image

def mask_metrics(gt_mask, pred_mask):

    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union_dice = gt_mask.sum() + pred_mask.sum()
    union_jaccard = np.logical_or(gt_mask, pred_mask).sum()

    dice = 2 * intersection / union_dice if union_dice > 0 else 0
    jaccard = intersection / union_jaccard if union_jaccard > 0 else 0

    tp = intersection
    fp = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    volume_dif = (pred_mask.sum()-gt_mask.sum())/gt_mask.sum() 
    volume_dif = volume_dif if (gt_mask.sum()) > 0 else 0

    return dice, jaccard, precision, recall, volume_dif

def surface_metrics(gt_mask, pred_mask, ref_image):

    gt_sitk = sitk.GetImageFromArray(np.transpose(gt_mask, (2, 1, 0)).astype(np.uint8))
    gt_sitk.CopyInformation(ref_image)  
    
    pred_sitk = sitk.GetImageFromArray(np.transpose(pred_mask, (2, 1, 0)).astype(np.uint8))
    pred_sitk.CopyInformation(ref_image)
    
    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(gt_sitk, pred_sitk)
    hd = hd_filter.GetHausdorffDistance()
    
    gt_surface = sitk.LabelContour(gt_sitk)
    pred_surface = sitk.LabelContour(pred_sitk)

    gt_surface_points = sitk.GetArrayFromImage(gt_surface)
    pred_surface_points = sitk.GetArrayFromImage(pred_surface)

    gt_distmap= sitk.Abs(sitk.SignedMaurerDistanceMap(gt_surface, squared=False, useImageSpacing=True))
    pred_distmap = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_surface, squared=False, useImageSpacing=True))

    gt_to_pred = sitk.GetArrayFromImage(pred_distmap)[sitk.GetArrayFromImage(gt_surface) > 0]
    pred_to_gt = sitk.GetArrayFromImage(gt_distmap)[sitk.GetArrayFromImage(pred_surface) > 0]

    points_gt_2_pred_tol = np.sum(gt_to_pred <= 1.0)
    points_pred_2_gt_tol = np.sum(pred_to_gt <= 1.0)

    total_points = gt_surface_points.sum() + pred_surface_points.sum()

    if len(gt_to_pred) > 0 and len(pred_to_gt) > 0 and total_points > 0:
        asd = (np.mean(np.abs(gt_to_pred)) + np.mean(np.abs(pred_to_gt))) / 2
        hd95 = max(np.percentile(gt_to_pred, 95), np.percentile(pred_to_gt, 95))
        nsd = (points_gt_2_pred_tol + points_pred_2_gt_tol)/(total_points)
    else:
        asd = np.nan
        hd95 = np.nan
        nsd = np.nan

    return hd, asd, hd95, nsd

def main():
    print('유사도 계산 시작')
    rt_total_base = input('TotalSegmentator 경로: ').strip('"')
    rt_utils_base = input('MIM/Onco 경로: ').strip('"')
    dicom_base = input('DICOM 경로: ').strip('"')
    output = input('출력 경로: ').strip('"')

    results = []
    failed_cases = []
    
    output_norm_dir = os.path.join(output, "Norm")
    os.makedirs(output_norm_dir, exist_ok=True)

    for i in tqdm(range(1, 105), desc="진행 중"):
        case_id = str(i)
        dicom_folder = os.path.join(dicom_base, case_id, "PRE")
        rt_total_folder = os.path.join(rt_total_base, case_id)
        rt_utils_folder = os.path.join(rt_utils_base, case_id)
        
        try:
            rt_total_files = [f for f in os.listdir(rt_total_folder) if f.endswith('.dcm')]
            rt_utils_files = [f for f in os.listdir(rt_utils_folder) if f.endswith('.dcm')]
            
            if not rt_total_files or not rt_utils_files:
                raise FileNotFoundError("RTStruct 파일 없음")
                
            rt_total_path = os.path.join(rt_total_folder, rt_total_files[0])
            rt_utils_path = os.path.join(rt_utils_folder, rt_utils_files[0])

            mask_total, mask_utils, ref_image = get_mask_and_ref(
                dicom_folder, rt_total_path, rt_utils_path
            )
                      
            dsc, jsc, pre, rec, vol_dif = mask_metrics(mask_utils, mask_total)
            hd, asd, hd95, nsd = surface_metrics(mask_utils, mask_total, ref_image)
            
            results.append({
                "Number": case_id,
                "Dice_Sørensen_Coefficient(DSC)": dsc,
                "Jaccard_Similarity": jsc,
                "Precision": pre,
                "Recall": rec,
                "Hausdorff_Distance": hd,
                "Hausdorff_Distance_95": hd95,
                "Average_Surface_Distance": asd,
                "Normalised_Surface_Distance": nsd,
                "Volume_Difference": vol_dif
            })
            
        except Exception as e:
            failed_cases.append(case_id)
            print(f"{case_id} 실패: {str(e)}")
            continue

    df = pd.DataFrame(results)
    output_path = os.path.join(output_norm_dir, "PRE_Total-Onco_statistics.xlsx")
    df.to_excel(output_path, index=False)
    print(f"\n계산 완료! 실패한 케이스: {failed_cases}")

if __name__ == "__main__":
    main()
