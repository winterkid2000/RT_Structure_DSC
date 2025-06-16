import numpy as np 
from rt_utils import RTStructBuilder
import os
from tqdm import tqdm
import pandas as pd 
import SimpleITK as sitk

def get_mask_and_ref(dicom_dir, total_rt_path, utils_rt_path):
    # 공통 참조 DICOM 시리즈 로드
    rt_ref = RTStructBuilder.create_from(dicom_dir)
    
    # 지오메트리 정보 추출
    ref_image = rt_ref.get_reference_ct()
    
    # 각 RTStruct 로드
    rt_total = RTStructBuilder.create_from(dicom_dir, total_rt_path)
    rt_utils = RTStructBuilder.create_from(dicom_dir, utils_rt_path)
    
    mask_total = rt_total.get_mask_by_name("Pancreas")
    mask_utils = rt_utils.get_mask_by_name("Pancreas_onco")
    
    return mask_total, mask_utils, ref_image

def mask_metrics(gt_mask, pred_mask):
    """Ground Truth(gt)와 Prediction(pred) 기반 메트릭 계산"""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union_dice = gt_mask.sum() + pred_mask.sum()
    union_jaccard = np.logical_or(gt_mask, pred_mask).sum()

    # Dice와 Jaccard는 원래 정의대로 계산
    dice = 2 * intersection / union_dice if union_dice > 0 else 0
    jaccard = intersection / union_jaccard if union_jaccard > 0 else 0

    # Precision, Recall 계산
    tp = intersection
    fp = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Volume Similarity 계산
    volume_sim = 1 - abs(gt_mask.sum() - pred_mask.sum()) / (gt_mask.sum() + pred_mask.sum()) 
    volume_sim = volume_sim if (gt_mask.sum() + pred_mask.sum()) > 0 else 0

    return dice, jaccard, precision, recall, volume_sim

def surface_metrics(gt_mask, pred_mask, ref_image):
    """참조 영상의 지오메트리 정보를 이용한 표면 거리 계산"""
    # Ground Truth 마스크 생성
    gt_sitk = sitk.GetImageFromArray(gt_mask.astype(np.uint8))
    gt_sitk.CopyInformation(ref_image)  # 모든 지오메트리 정보 복제
    
    # Prediction 마스크 생성
    pred_sitk = sitk.GetImageFromArray(pred_mask.astype(np.uint8))
    pred_sitk.CopyInformation(ref_image)
    
    # Hausdorff Distance
    hd_filter = sitk.HausdorffDistanceImageFilter()
    hd_filter.Execute(gt_sitk, pred_sitk)
    hd = hd_filter.GetHausdorffDistance()
    
    # 95% Hausdorff Distance
    hd95_filter = sitk.HausdorffDistanceImageFilter()
    hd95_filter.SetPercentile(95)
    hd95_filter.Execute(gt_sitk, pred_sitk)
    hd95 = hd95_filter.GetHausdorffDistance()
    
    # Average Symmetric Surface Distance
    sd_filter = sitk.SurfaceDistanceImageFilter()
    sd_filter.Execute(gt_sitk, pred_sitk)
    asd = sd_filter.GetAverageSymmetricSurfaceDistance()
    
    return hd, asd, hd95

def main():
    print('유사도 계산 시작')
    rt_total_base = input('TotalSegmentator 경로: ').strip('"')
    rt_utils_base = input('MIM/Onco 경로: ').strip('"')
    dicom_base = input('DICOM 경로: ').strip('"')
    output = input('출력 경로: ').strip('"')

    results = []
    failed_cases = []
    
    # 출력 디렉토리 생성
    output_norm_dir = os.path.join(output, "Norm")
    os.makedirs(output_norm_dir, exist_ok=True)

    for i in tqdm(range(1, 105), desc="진행 중"):
        case_id = str(i)
        dicom_folder = os.path.join(dicom_base, case_id, "PRE")
        rt_total_folder = os.path.join(rt_total_base, case_id)
        rt_utils_folder = os.path.join(rt_utils_base, case_id)
        
        try:
            # RTStruct 파일 찾기
            rt_total_files = [f for f in os.listdir(rt_total_folder) if f.endswith('.dcm')]
            rt_utils_files = [f for f in os.listdir(rt_utils_folder) if f.endswith('.dcm')]
            
            if not rt_total_files or not rt_utils_files:
                raise FileNotFoundError("RTStruct 파일 없음")
                
            rt_total_path = os.path.join(rt_total_folder, rt_total_files[0])
            rt_utils_path = os.path.join(rt_utils_folder, rt_utils_files[0])

            # 마스크 및 참조 영상 가져오기
            mask_total, mask_utils, ref_image = get_mask_and_ref(
                dicom_folder, rt_total_path, rt_utils_path
            )
            
            # 메트릭 계산: mask_utils(Onco)를 Ground Truth, mask_total(TotalSegmentator)를 Prediction으로 간주
            dsc, jsc, pre, rec, vol_sim = mask_metrics(mask_utils, mask_total)
            hd, asd, hd95 = surface_metrics(mask_utils, mask_total, ref_image)
            
            results.append({
                "Number": case_id,
                "Dice_Similarity": dsc,
                "Jaccard_Similarity": jsc,
                "Precision": pre,
                "Recall": rec,
                "Hausdorff_Distance": hd,
                "Hausdorff_Distance_95": hd95,
                "Average_Surface_Distance": asd,
                "Volume_Similarity": vol_sim
            })
            
        except Exception as e:
            failed_cases.append(case_id)
            print(f"{case_id} 실패: {str(e)}")
            continue

    # 결과 저장
    df = pd.DataFrame(results)
    output_path = os.path.join(output_norm_dir, "PRE_Total-Onco_statistics.xlsx")
    df.to_excel(output_path, index=False)
    print(f"\n계산 완료! 실패한 케이스: {failed_cases}")

if __name__ == "__main__":
    main()
