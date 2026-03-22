import os
import re
import math
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from glob import glob
from collections import Counter, defaultdict

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_dir_inference', default='path/to/saved_inference_result', type=str)
    parser.add_argument('--saved_dir_scoring', default='path/to/saved_scoring_result', type=str)
    args = parser.parse_args()
    return args

df_measure_type = {'perception': ['inclusion', 'inspiration', 'trachea_deviation', 'ascending_aorta_enlargement'],
                   'measurement': ['projection', 'rotation', 'aortic_knob_enlargement',
                                  'carina_angle', 'cardiomegaly', 'mediastinal_widening',
                                  'descending_aorta_tortuous', 'descending_aorta_enlargement'],
                   'total': ['rotation', 'projection', 'cardiomegaly',
                             'mediastinal_widening', 'carina_angle', 'aortic_knob_enlargement',
                             'descending_aorta_enlargement', 'descending_aorta_tortuous',
                             'trachea_deviation', 'inclusion', 'inspiration', 'ascending_aorta_enlargement']}

def wilson_score_n_naive_score(num_correct, num_trial, z=1.96):
    if num_trial:
        wilson_score = (num_correct/num_trial + (z**2)/(2*num_trial)) / (1 + (z**2)/num_trial)

        lower = (num_correct/num_trial + (z**2)/(2*num_trial) -
                 z * math.sqrt((num_correct/num_trial)*(1-num_correct/num_trial)/num_trial
                                  + (z**2)/(4*num_trial**2))) / (1 + (z**2)/num_trial)

        upper = (num_correct / num_trial + (z ** 2) / (2 * num_trial) +
                 z * math.sqrt((num_correct / num_trial) * (1 - num_correct / num_trial) / num_trial
                                  + (z ** 2) / (4 * num_trial ** 2))) / (1 + (z ** 2) / num_trial)

        adjusted_wilson = wilson_score - z * (upper - lower) / 2
        naive_score = num_correct / num_trial

        bayesian = (num_correct + 1) / (num_trial + 2)
    else:
        num_correct = 'N/A'
        wilson_score = 'N/A'
        adjusted_wilson = 'N/A'
        lower = 'N/A'
        upper = 'N/A'
        naive_score = 'N/A'
        bayesian = 'N/A'

    return {'num_correct': num_correct,
            'wilson_score': wilson_score,
            'adjusted_wilson': adjusted_wilson,
            'lower_bound': lower, 'upper_bound': upper,
            'bayesian': bayesian,
            'naive_score': naive_score}

def save_dict2df(result: dict, fname: str):
    df_result = pd.DataFrame(result)
    df_result.to_csv(fname, index=False)
    print('Saved to:', fname)

def calculate_metrics(args, inference_path):
    if inference_path in ['reasoning']:
        scoring_result_dx_lst = sorted(glob(f"{args.saved_dir_scoring}/*"))
    elif inference_path in ['guidance']:
        merged_path_scoring = f'{args.saved_dir_scoring}/idk_merged'
        merged_path_inference = f'{args.saved_dir_inference}/idk_merged'
        if os.path.exists(merged_path_scoring):
            shutil.rmtree(merged_path_scoring)
        if os.path.exists(merged_path_inference):
            shutil.rmtree(merged_path_inference)
        os.makedirs(merged_path_scoring, exist_ok=True)
        os.makedirs(merged_path_inference, exist_ok=True)
        idk_path_lst_score = [p for p in glob(f'{args.saved_dir_scoring}/*') if os.path.isdir(p)]
        idk_path_lst_inference = [p for p in glob(f'{args.saved_dir_inference}/*') if os.path.isdir(p)]
        for idk_path_lst, merged_path in zip([idk_path_lst_score, idk_path_lst_inference], [merged_path_scoring, merged_path_inference]):
            for idk_path in idk_path_lst:
                for item in os.listdir(idk_path):
                    src_folder = os.path.join(idk_path, item)
                    dst_folder = os.path.join(merged_path, item)

                    if not os.path.isdir(src_folder):
                        continue

                    if os.path.exists(dst_folder):
                        src_files = set(os.listdir(src_folder))
                        dst_files = set(os.listdir(dst_folder))

                        files_to_copy = src_files - dst_files

                        for file_name in files_to_copy:
                            src_file = os.path.join(src_folder, file_name)
                            dst_file = os.path.join(dst_folder, file_name)

                            if os.path.isfile(src_file):
                                shutil.copy2(src_file, dst_file)
                    else:
                        shutil.copytree(src_folder, dst_folder)
        scoring_result_dx_lst = sorted(glob(f"{args.saved_dir_scoring}/idk_merged/*"))

    df_results_total = defaultdict()
    if len(scoring_result_dx_lst):
        for scoring_path_dx in scoring_result_dx_lst:
            if os.path.isdir(scoring_path_dx):
                dx = scoring_path_dx.split('/')[-1]

                if inference_path in ['reasoning']:
                    possible_stage_lst = glob(f'{args.qa_base_dir}/{dx}/path1/*')
                    header1 = ['init', 'criteria', 'bodypart', 'measurement', 'final']
                    header2 = ['init', 'criteria', 'custom_criteria', 'bodypart', 'measurement', 'final']
                    if len(header1) == len(possible_stage_lst):
                        header_stage = header1
                    elif len(header2) == len(possible_stage_lst):
                        header_stage = header2

                elif inference_path in ['guidance']:
                    possible_stage_lst = glob(f'{args.qa_base_dir}/{dx}/path2/*') + glob(f'{args.qa_base_dir}/{dx}/re-path1/*')
                    header_stage = ['guidance-bodypart', 'guidance-measurement', 'guidance-final',
                                    'review-init', 'review-criteria', 'review-bodypart', 'review-measurement', 'review-final']
                assert len(possible_stage_lst) == len(header_stage)

                df_scores = defaultdict(list)
                dicom_lst_measurement_matching_per_stage = defaultdict(list)

                scoring_file_lst = glob(f'{scoring_path_dx}/*.json')  # dicom_lst
                for scoring_file in scoring_file_lst:
                    dicom = scoring_file.split('/')[-1].split('.')[0]
                    with open(scoring_file, "r") as file:
                        scoring_result = json.load(file)

                    if dx in df_measure_type['measurement']:
                        if inference_path in ['reasoning']:
                            inference_file = os.path.join(args.saved_dir_inference, dx, f"{dicom}.json")
                        else:
                            inference_file = os.path.join(args.saved_dir_inference, 'idk_merged', dx, f"{dicom}.json")

                        with open(inference_file, 'r') as file:
                            inference_result = json.load(file)

                        for stage, val in scoring_result.items():
                            if stage.startswith('stage-') and isinstance(val, str):
                                if stage == 'stage-measured_value':
                                    answer_measured_value = inference_result['stage-measurement']['answer']
                                    model_measured_value = str(inference_result['stage-final']['response'])
                                elif stage == 'stage-measured_value_guidance':
                                    answer_measured_value = inference_result['stage-guidance-measurement']['answer']
                                    model_measured_value = str(inference_result['stage-guidance-final']['response'])
                                elif stage == 'stage-measured_value_review':
                                    answer_measured_value = inference_result['stage-review-measurement']['answer']
                                    model_measured_value = str(inference_result['stage-review-final']['response'])
                                answer_measured_value = re.findall(r"[-+]?\d*\.\d+|\d+", answer_measured_value)

                                model_measured_value_no_range = re.sub(r'\d+\.?\d*\s*-\s*\d+\.?\d*', '', model_measured_value)
                                model_measured_value = re.findall(r"[+]?\d*\.\d+|\d+", model_measured_value_no_range.replace(' ', ''))

                                if dx in ['projection']:
                                    if len(model_measured_value) == 2:
                                        model_measured_value_right = float(model_measured_value[0])
                                        model_measured_value_left = float(model_measured_value[-1])
                                        answer_measured_value_right = answer_measured_value[:2]
                                        answer_measured_value_left = answer_measured_value[2:]

                                        if float(answer_measured_value_right[0]) <= model_measured_value_right <= float(
                                                answer_measured_value_right[-1]):
                                            if float(answer_measured_value_left[0]) <= model_measured_value_left <= float(
                                                    answer_measured_value_left[-1]):
                                                score_measure_matching = 1
                                                dicom_lst_measurement_matching_per_stage[stage].append(dicom)
                                        else:
                                            score_measure_matching = 0
                                            dicom_lst_measurement_matching_per_stage[f'incorrect_{stage}'].append(dicom)
                                    else:
                                        score_measure_matching = 0
                                        dicom_lst_measurement_matching_per_stage[f'incorrect_{stage}'].append(dicom)
                                else:
                                    if len(set(model_measured_value)) == 1:
                                        if float(answer_measured_value[0]) <= float(model_measured_value[0]) <= float(answer_measured_value[-1]):
                                            score_measure_matching = 1
                                            dicom_lst_measurement_matching_per_stage[stage].append(dicom)
                                        else:
                                            score_measure_matching = 0
                                            dicom_lst_measurement_matching_per_stage[f'incorrect_{stage}'].append(dicom)
                                    else:
                                        score_measure_matching = 0
                                        dicom_lst_measurement_matching_per_stage[f'incorrect_{stage}'].append(dicom)

                    df_scores_dicom = dict()
                    stage_lst = scoring_result.keys()
                    naive_stage_lst = [re.sub(r'\d+', '', s) for s in stage_lst if s.startswith('stage-') and not isinstance(scoring_result[s], str)]
                    stage_by_cnt = Counter(naive_stage_lst)
                    for naive_stage, cnt in stage_by_cnt.items():
                        results = [result for stage, result in scoring_result.items() if stage.startswith(naive_stage)]
                        name_stage = naive_stage.replace('stage-', '').rstrip('_')
                        if naive_stage in scoring_result:
                            df_scores_dicom[name_stage] = scoring_result[naive_stage]
                        else:
                            if len(results) == sum(results):
                                df_scores_dicom[name_stage] = 1
                            else:
                                df_scores_dicom[name_stage] = 0

                    for remained_stage in header_stage:
                        if remained_stage not in df_scores_dicom:
                            df_scores_dicom[remained_stage] = 0

                    df_scores_dicom['dicom'] = dicom
                    for key in df_scores_dicom:
                        df_scores[key].append(df_scores_dicom[key])

                assert len(df_scores['dicom']) == len(scoring_file_lst)

                if inference_path in ['reasoning']:
                    idk_init = Counter(df_scores['init'])[-1]
                    pass_init = len(df_scores['init']) - idk_init
                    score_info_pass_init = wilson_score_n_naive_score(num_correct=pass_init,
                                                                      num_trial=len(df_scores['init']))

                    if pass_init:
                        score_info_idk_init = wilson_score_n_naive_score(num_correct=idk_init, num_trial=len(df_scores['init']))

                        num_correct_criteria = (np.array(df_scores['criteria']) == 1).sum()
                        num_correct_bodypart = (np.array(df_scores['bodypart']) == 1).sum()
                        num_correct_measurement = (np.array(df_scores['measurement']) == 1).sum()
                        num_correct_final = (np.array(df_scores['final']) == 1).sum()

                        mask_correct_final = (np.array(df_scores['final']) == 1)
                        dicom_correct_final = np.array(df_scores['dicom'])[mask_correct_final].tolist()

                        consistency_init = np.array(df_scores['init'])[mask_correct_final].sum()
                        assert consistency_init <= num_correct_final
                        assert consistency_init <= num_correct_measurement
                        score_info_consistency = wilson_score_n_naive_score(num_correct=consistency_init, num_trial=num_correct_measurement)

                        if dx in df_measure_type['measurement']:
                            num_correct_measurement_matching = len(dicom_lst_measurement_matching_per_stage['stage-measured_value'])
                            score_info_measure_matching = wilson_score_n_naive_score(num_correct=num_correct_measurement_matching, num_trial=num_correct_measurement)
                            num_correct_final_refined = len(set(dicom_correct_final).intersection(dicom_lst_measurement_matching_per_stage['stage-measured_value']))

                            mask_correct_final_refined = []
                            for dicom, correct in zip(df_scores['dicom'], df_scores['final']):
                                if correct == 1 and dicom in dicom_lst_measurement_matching_per_stage['stage-measured_value']:
                                    mask_correct_final_refined.append(1)
                                else:
                                    mask_correct_final_refined.append(0)
                            mask_correct_final_refined = (np.array(mask_correct_final_refined) == 1)
                            consistency_init_refined = np.array(df_scores['init'])[mask_correct_final_refined].sum()
                            assert consistency_init_refined <= num_correct_final_refined
                            assert consistency_init_refined <= num_correct_measurement_matching
                            score_info_consistency_refined = wilson_score_n_naive_score(num_correct=consistency_init_refined, num_trial=num_correct_measurement_matching)

                        if 'custom_criteria' in df_scores:
                            idk_custom = Counter(df_scores['custom_criteria'])[-1]
                            pass_custom = num_correct_criteria - idk_custom

                            score_info_idk_custom = wilson_score_n_naive_score(num_correct=idk_custom, num_trial=num_correct_criteria)
                            score_info_pass_custom = wilson_score_n_naive_score(num_correct=pass_custom, num_trial=num_correct_criteria)

                            num_incorrect_criteria = (pass_init - num_correct_criteria)

                            score_info_criteria = wilson_score_n_naive_score(num_correct=num_correct_criteria, num_trial=pass_init)
                            score_info_bodypart = wilson_score_n_naive_score(num_correct=num_correct_bodypart, num_trial=pass_custom)

                            score_info_measure = wilson_score_n_naive_score(num_correct=num_correct_measurement, num_trial=num_correct_bodypart)
                            score_info_final = wilson_score_n_naive_score(num_correct=num_correct_final, num_trial=num_correct_measurement)
                            score_info_binary = wilson_score_n_naive_score(num_correct=num_correct_final, num_trial=(pass_custom + num_incorrect_criteria))

                            conquer_stage_lst = ([header_stage.index('final') - 1] * num_correct_final) \
                                                + ([header_stage.index('measurement') - 1] * (num_correct_measurement - num_correct_final)) \
                                                + ([header_stage.index('bodypart') - 1] * (num_correct_bodypart - num_correct_measurement)) \
                                                + ([header_stage.index('criteria')] * (pass_custom - num_correct_bodypart)) \
                                                + ([header_stage.index('init')] * (pass_init - num_correct_criteria))
                            assert len(conquer_stage_lst) == (num_incorrect_criteria + pass_custom)

                            if dx in df_measure_type['measurement']:
                                score_info_measure_refined = wilson_score_n_naive_score(num_correct=num_correct_measurement_matching, num_trial=num_correct_bodypart)
                                score_info_final_refined = wilson_score_n_naive_score(num_correct=num_correct_final_refined, num_trial=num_correct_measurement_matching)
                                score_info_binary_refined = wilson_score_n_naive_score(num_correct=num_correct_final_refined, num_trial=(pass_custom + num_incorrect_criteria))

                                conquer_stage_lst_refined = ([header_stage.index('final') - 1] * num_correct_final_refined) \
                                                            + ([header_stage.index('measurement') - 1] * (num_correct_measurement_matching - num_correct_final_refined)) \
                                                            + ([header_stage.index('bodypart') - 1] * (num_correct_bodypart - num_correct_measurement_matching)) \
                                                            + ([header_stage.index('criteria')] * (pass_custom - num_correct_bodypart)) \
                                                            + ([header_stage.index('init')] * (pass_init - num_correct_criteria))
                                stage_score_refined = np.array(conquer_stage_lst_refined).mean()
                                stage_std_refined = np.array(conquer_stage_lst_refined).std()
                                assert len(conquer_stage_lst_refined) == (num_incorrect_criteria + pass_custom)
                        else:
                            score_info_idk_custom = wilson_score_n_naive_score(num_correct='N/A', num_trial=0)
                            score_info_pass_custom = wilson_score_n_naive_score(num_correct='N/A', num_trial=0)

                            score_info_criteria = wilson_score_n_naive_score(num_correct=num_correct_criteria, num_trial=pass_init)
                            score_info_bodypart = wilson_score_n_naive_score(num_correct=num_correct_bodypart, num_trial=num_correct_criteria)

                            score_info_measure = wilson_score_n_naive_score(num_correct=num_correct_measurement, num_trial=num_correct_bodypart)
                            score_info_final = wilson_score_n_naive_score(num_correct=num_correct_final, num_trial=num_correct_measurement)
                            score_info_binary = wilson_score_n_naive_score(num_correct=num_correct_final, num_trial=pass_init)

                            conquer_stage_lst = ([header_stage.index('final')] * num_correct_final) \
                                                + ([header_stage.index('measurement')] * (num_correct_measurement - num_correct_final)) \
                                                + ([header_stage.index('bodypart')] * (num_correct_bodypart - num_correct_measurement)) \
                                                + ([header_stage.index('criteria')] * (num_correct_criteria - num_correct_bodypart)) \
                                                + ([header_stage.index('init')] * (pass_init - num_correct_criteria))
                            assert len(conquer_stage_lst) == pass_init

                            if dx in df_measure_type['measurement']:
                                score_info_measure_refined = wilson_score_n_naive_score(num_correct=num_correct_measurement_matching, num_trial=num_correct_bodypart)
                                score_info_final_refined = wilson_score_n_naive_score(num_correct=num_correct_final_refined, num_trial=num_correct_measurement_matching)
                                score_info_binary_refined = wilson_score_n_naive_score(num_correct=num_correct_final_refined, num_trial=pass_init)

                                conquer_stage_lst_refined = ([header_stage.index('final')] * num_correct_final_refined) \
                                                            + ([header_stage.index('measurement')] * (num_correct_measurement_matching - num_correct_final_refined)) \
                                                            + ([header_stage.index('bodypart')] * (num_correct_bodypart - num_correct_measurement_matching)) \
                                                            + ([header_stage.index('criteria')] * (num_correct_criteria - num_correct_bodypart)) \
                                                            + ([header_stage.index('init')] * (pass_init - num_correct_criteria))

                                stage_score_refined = np.array(conquer_stage_lst_refined).mean()
                                stage_std_refined = np.array(conquer_stage_lst_refined).std()
                                assert len(conquer_stage_lst_refined) == pass_init

                        stage_score = np.array(conquer_stage_lst).mean()
                        stage_std = np.array(conquer_stage_lst).std()

                        df_result_per_dx = {'inference_type': inference_path, 'dx': dx, 'model': args.model_id}
                        score_info_per_stage_flag = {
                            'init_pass': score_info_pass_init,
                            'init_idk': score_info_idk_init,
                            'custom_pass': score_info_pass_custom,
                            'custom_idk': score_info_idk_custom
                        }
                        for stage, score_info in score_info_per_stage_flag.items():
                            for score_name, value in score_info.items():
                                df_result_per_dx[f'flag_{score_name}_{stage}'] = value

                        score_info_per_stage = {'criteria': score_info_criteria, 'bodypart': score_info_bodypart,
                                                'measurement': score_info_measure, 'final': score_info_final,
                                                'binary': score_info_binary, 'consistency': score_info_consistency}
                        for stage, score_info in score_info_per_stage.items():
                            if isinstance(score_info, dict):
                                for score_name, value in score_info.items():
                                    df_result_per_dx[f'{score_name}_{stage}'] = value

                        df_result_per_dx['stage_score'] = stage_score
                        df_result_per_dx['stage_score_std'] = stage_std

                        if dx in df_measure_type['measurement']:
                            score_info_per_stage_refined = {
                                'measurement_matching': score_info_measure_matching,
                                'measurement_refined': score_info_measure_refined,
                                'final_refined': score_info_final_refined,
                                'binary_refined': score_info_binary_refined,
                                'stage_score_refined': stage_score_refined,
                                'stage_score_std_refined': stage_std_refined,
                                'consistency_refined': score_info_consistency_refined,
                            }
                        else:
                            score_info_per_stage_refined = {
                                'measurement_matching': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                                'measurement_refined': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                                'final_refined': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                                'binary_refined': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                                'stage_score_refined': 'N/A',
                                'stage_score_std_refined': 'N/A',
                                'consistency_refined': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                            }

                        for stage, score_info in score_info_per_stage_refined.items():
                            if isinstance(score_info, dict):
                                for score_name, value in score_info.items():
                                    df_result_per_dx[f'{score_name}_{stage}'] = value
                            else:
                                df_result_per_dx[stage] = score_info

                        df_result_per_dx.update({'num_conquer_stage': len(conquer_stage_lst), 'conquer_stage_lst': conquer_stage_lst})

                        if dx in df_measure_type['measurement']:
                            df_result_per_dx.update({'num_conquer_stage_refined': len(conquer_stage_lst), 'conquer_stage_lst_refined': conquer_stage_lst})
                        else:
                            df_result_per_dx.update({'num_conquer_stage_refined': 'N/A', 'conquer_stage_lst_refined': 'N/A'})

                    else:
                        df_result_per_dx = {}

                elif inference_path in ['guidance']:
                    # ============# ============# ============# ============
                    #                   Guidance
                    # ============# ============# ============# ============
                    num_correct_g_body = (np.array(df_scores['guidance-bodypart']) == 1).sum()
                    num_correct_g_measure = (np.array(df_scores['guidance-measurement']) == 1).sum()
                    num_correct_g_final = (np.array(df_scores['guidance-final']) == 1).sum()

                    mask_correct_g_final = (np.array(df_scores['guidance-final']) == 1)
                    dicom_correct_g_final = np.array(df_scores['dicom'])[mask_correct_g_final].tolist()

                    score_info_g_bodypart = wilson_score_n_naive_score(num_correct=num_correct_g_body, num_trial=len(df_scores['dicom']))
                    score_info_g_measure = wilson_score_n_naive_score(num_correct=num_correct_g_measure, num_trial=num_correct_g_body)
                    score_info_g_final = wilson_score_n_naive_score(num_correct=num_correct_g_final, num_trial=num_correct_g_measure)
                    score_info_g_binary = wilson_score_n_naive_score(num_correct=num_correct_g_final, num_trial=len(df_scores['dicom']))

                    conquer_stage_lst_guidance = ([3] * num_correct_g_final) \
                                                 + ([2] * (num_correct_g_measure - num_correct_g_final)) \
                                                 + ([1] * (num_correct_g_body - num_correct_g_measure)) \
                                                 + ([0] * (len(df_scores['dicom']) - num_correct_g_body))

                    assert len(conquer_stage_lst_guidance) == len(df_scores['dicom'])

                    stage_score_guidance = np.array(conquer_stage_lst_guidance).mean()
                    stage_std_guidance = np.array(conquer_stage_lst_guidance).std()

                    if dx in df_measure_type['measurement']:
                        num_correct_g_measure_matching = len(dicom_lst_measurement_matching_per_stage['stage-measured_value_guidance'])
                        score_info_g_measure_matching = wilson_score_n_naive_score(num_correct=num_correct_g_measure_matching, num_trial=num_correct_g_measure)
                        num_correct_g_final_refined = len(set(dicom_correct_g_final).intersection(dicom_lst_measurement_matching_per_stage['stage-measured_value_guidance']))

                        score_info_g_measure_refined = wilson_score_n_naive_score(num_correct=num_correct_g_measure_matching, num_trial=num_correct_g_body)
                        score_info_g_final_refined = wilson_score_n_naive_score(num_correct=num_correct_g_final_refined, num_trial=num_correct_g_measure_matching)
                        score_info_g_binary_refined = wilson_score_n_naive_score(num_correct=num_correct_g_final_refined, num_trial=len(df_scores['dicom']))

                        conquer_stage_lst_guidance_refined = ([3] * num_correct_g_final_refined) \
                                                             + ([2] * (num_correct_g_measure_matching - num_correct_g_final_refined)) \
                                                             + ([1] * (num_correct_g_body - num_correct_g_measure_matching)) \
                                                             + ([0] * (len(df_scores['dicom']) - num_correct_g_body))

                        stage_score_guidance_refined = np.array(conquer_stage_lst_guidance_refined).mean()
                        stage_std_guidance_refined = np.array(conquer_stage_lst_guidance_refined).std()
                        assert len(conquer_stage_lst_guidance_refined) == len(df_scores['dicom'])

                    # ============# ============# ============# ============
                    #                         Review
                    # ============# ============# ============# ============
                    idk_init_review = (np.array(df_scores['review-init']) == -1).sum()
                    idk_init_review_ratio = (np.array(df_scores['review-init']) == -1).sum() / num_correct_g_final

                    num_correct_r_init = (np.array(df_scores['review-init']) == 1).sum()
                    num_correct_r_criteria = (np.array(df_scores['review-criteria']) == 1).sum()
                    num_correct_r_bodypart = (np.array(df_scores['review-bodypart']) == 1).sum()
                    num_correct_r_measure = (np.array(df_scores['review-measurement']) == 1).sum()
                    num_correct_r_final = (np.array(df_scores['review-final']) == 1).sum()

                    mask_correct_r_final = (np.array(df_scores['review-final']) == 1)
                    dicom_correct_r_final = np.array(df_scores['dicom'])[mask_correct_r_final].tolist()

                    score_info_r_init = wilson_score_n_naive_score(num_correct=num_correct_r_init,num_trial=num_correct_g_final)
                    score_info_r_criteria = wilson_score_n_naive_score(num_correct=num_correct_r_criteria, num_trial=num_correct_r_init)
                    score_info_r_body = wilson_score_n_naive_score(num_correct=num_correct_r_bodypart, num_trial=num_correct_r_criteria)
                    score_info_r_measure = wilson_score_n_naive_score(num_correct=num_correct_r_measure, num_trial=num_correct_r_bodypart)
                    score_info_r_final = wilson_score_n_naive_score(num_correct=num_correct_r_final, num_trial=num_correct_r_measure)

                    consistency_r_init = np.array(df_scores['review-init'])[mask_correct_r_final].sum()
                    assert consistency_r_init <= num_correct_r_final
                    assert consistency_r_init <= num_correct_r_measure
                    score_info_r_consistency = wilson_score_n_naive_score(num_correct=consistency_r_init, num_trial=num_correct_r_measure)

                    score_info_r_binary = wilson_score_n_naive_score(num_correct=num_correct_r_final, num_trial=num_correct_r_init)

                    conquer_stage_lst_review = ([4] * num_correct_r_final) \
                                               + ([3] * (num_correct_r_measure - num_correct_r_final)) \
                                               + ([2] * (num_correct_r_bodypart - num_correct_r_measure)) \
                                               + ([1] * (num_correct_r_criteria - num_correct_r_bodypart)) \
                                               + ([0] * (num_correct_r_init - num_correct_r_criteria))
                    stage_score_review = np.array(conquer_stage_lst_review).mean()
                    stage_std_review = np.array(conquer_stage_lst_review).std()

                    if dx in df_measure_type['measurement']:
                        num_correct_r_measure_matching = len(dicom_lst_measurement_matching_per_stage['stage-measured_value_review'])
                        score_info_r_measure_matching = wilson_score_n_naive_score(num_correct=num_correct_r_measure_matching, num_trial=num_correct_r_measure)
                        num_correct_r_final_refined = len(set(dicom_correct_r_final).intersection(dicom_lst_measurement_matching_per_stage['stage-measured_value_review']))

                        score_info_r_measure_refined = wilson_score_n_naive_score(num_correct=num_correct_r_measure_matching, num_trial=num_correct_r_bodypart)
                        score_info_r_final_refined = wilson_score_n_naive_score(num_correct=num_correct_r_final_refined, num_trial=num_correct_r_measure_matching)

                        score_info_r_binary_refined = wilson_score_n_naive_score(num_correct=num_correct_r_final_refined, num_trial=num_correct_r_init)

                        mask_correct_final_refined = []
                        for dicom, correct in zip(df_scores['dicom'], df_scores['review-final']):
                            if correct == 1 and dicom in dicom_lst_measurement_matching_per_stage['stage-measured_value_review']:
                                mask_correct_final_refined.append(1)
                            else:
                                mask_correct_final_refined.append(0)
                        mask_correct_final_refined = (np.array(mask_correct_final_refined) == 1)
                        consistency_r_init_refined = np.array(df_scores['review-init'])[mask_correct_final_refined].sum()
                        assert consistency_r_init_refined <= num_correct_r_measure_matching
                        score_info_r_consistency_refined = wilson_score_n_naive_score(num_correct=consistency_r_init_refined, num_trial=num_correct_r_measure_matching)

                        conquer_stage_lst_review_refined = ([4] * num_correct_r_final_refined) \
                                                           + ([3] * (num_correct_r_measure_matching - num_correct_r_final_refined)) \
                                                           + ([2] * (num_correct_r_bodypart - num_correct_r_measure_matching)) \
                                                           + ([1] * (num_correct_r_criteria - num_correct_r_bodypart)) \
                                                           + ([0] * (num_correct_r_init - num_correct_r_criteria))
                        stage_score_review_refined = np.array(conquer_stage_lst_review_refined).mean()
                        stage_std_review_refined = np.array(conquer_stage_lst_review_refined).std()

                    df_result_per_dx = {
                        'inference_type': inference_path, 'dx': dx, 'model': args.model_id,
                        'num_idk_for_guidance': len(df_scores['dicom']),
                        'num_idk_init_review': idk_init_review, 'ratio_idk_init_review': idk_init_review_ratio,
                    }

                    score_info_g_per_stage = {'bodypart': score_info_g_bodypart, 'measurement': score_info_g_measure,
                                              'final': score_info_g_final, 'binary': score_info_g_binary}
                    for stage, score_info in score_info_g_per_stage.items():
                        if isinstance(score_info, dict):
                            for score_name, value in score_info.items():
                                df_result_per_dx[f'{score_name}_g_{stage}'] = value

                    df_result_per_dx['g_stage_score'] = stage_score_guidance
                    df_result_per_dx['g_stage_score_std'] = stage_std_guidance

                    if dx in df_measure_type['measurement']:
                        score_info_g_per_stage = {
                            'measurement_matching': score_info_g_measure_matching,
                            'measurement_refined': score_info_g_measure_refined,
                            'final_refined': score_info_g_final_refined,
                            'binary_refined': score_info_g_binary_refined,
                            'g_stage_score_refined': stage_score_guidance_refined,
                            'g_stage_score_std_refined': stage_std_guidance_refined}
                    else:
                        score_info_g_per_stage = {
                            'measurement_matching': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                            'measurement_refined': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                            'final_refined': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                            'binary_refined': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                            'g_stage_score_refined': 'N/A',
                            'g_stage_score_std_refined': 'N/A'}
                    for stage, score_info in score_info_g_per_stage.items():
                        if isinstance(score_info, dict):
                            for score_name, value in score_info.items():
                                df_result_per_dx[f'{score_name}_g_{stage}'] = value
                        else:
                            df_result_per_dx[stage] = score_info

                    score_info_r_per_stage = {'init': score_info_r_init, 'criteria': score_info_r_criteria,
                                              'bodypart': score_info_r_body, 'measurement': score_info_r_measure,
                                              'final': score_info_r_final, 'binary': score_info_r_binary, 'consistency': score_info_r_consistency}

                    for stage, score_info in score_info_r_per_stage.items():
                        if isinstance(score_info, dict):
                            for score_name, value in score_info.items():
                                df_result_per_dx[f'{score_name}_r_{stage}'] = value

                    df_result_per_dx['r_stage_score'] = stage_score_review
                    df_result_per_dx['r_stage_score_std'] = stage_std_review

                    if dx in df_measure_type['measurement']:
                        score_info_r_per_stage = {
                            'measurement_matching': score_info_r_measure_matching,
                            'measurement_refined': score_info_r_measure_refined,
                            'final_refined': score_info_r_final_refined,
                            'binary_refined': score_info_r_binary_refined,
                            'r_stage_score_refined': stage_score_review_refined,
                            'r_stage_score_std_refined': stage_std_review_refined,
                            'consistency_refined': score_info_r_consistency_refined
                        }
                    else:
                        score_info_r_per_stage = {
                            'measurement_matching': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                            'measurement_refined': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                            'final_refined': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                            'binary_refined': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                            'consistency_refined': wilson_score_n_naive_score(num_correct='N/A', num_trial=0),
                            'r_stage_score_refined': 'N/A',
                            'r_stage_score_std_refined': 'N/A'}

                    for stage, score_info in score_info_r_per_stage.items():
                        if isinstance(score_info, dict):
                            for score_name, value in score_info.items():
                                df_result_per_dx[f'{score_name}_r_{stage}'] = value
                        else:
                            df_result_per_dx[stage] = score_info

                    df_result_per_dx.update(
                        {'num_conquer_stage_g': len(conquer_stage_lst_guidance),
                         'conquer_stage_lst_g': conquer_stage_lst_guidance,
                         'num_conquer_stage_r': len(conquer_stage_lst_review),
                         'conquer_stage_lst_r': conquer_stage_lst_review})

                    if dx in df_measure_type['measurement']:
                        df_result_per_dx.update(
                            {'num_conquer_stage_g_refined': len(conquer_stage_lst_guidance_refined),
                             'conquer_stage_lst_g_refined': conquer_stage_lst_guidance_refined,
                             'num_conquer_stage_r_refined': len(conquer_stage_lst_review_refined),
                             'conquer_stage_lst_r_refined': conquer_stage_lst_review_refined})
                    else:
                        df_result_per_dx.update(
                            {'num_conquer_stage_g_refined': 'N/A', 'conquer_stage_lst_g_refined': 'N/A',
                             'num_conquer_stage_r_refined': 'N/A', 'conquer_stage_lst_r_refined': 'N/A'})

                if df_result_per_dx:
                    df_results_total[dx] = df_result_per_dx
    return df_results_total

def gather_scores(inference_path, df_results_total):
    df_scores_total = defaultdict(list)
    gathered_total_dx_score = defaultdict(list)
    gathered_total_perception_score = defaultdict(list)
    gathered_total_arithmatic_score = defaultdict(list)
    if len(df_results_total):
        for dx, df_result_per_dx in df_results_total.items():
            num_conquer_stage = df_result_per_dx['num_conquer_stage'] if inference_path in ['reasoning'] else 999
            if num_conquer_stage != 0:
                for key, val in df_result_per_dx.items():
                    if not isinstance(val, (str, list)):
                        flag_key = 'flag_num_correct_custom_pass'
                        if flag_key in df_result_per_dx:
                            gathered_total_dx_score[key].append(val)
                            if dx in df_measure_type['perception']:
                                gathered_total_perception_score[key].append(val)
                            else:
                                gathered_total_arithmatic_score[key].append(val)
                        else:  # guidance, review
                            gathered_total_dx_score[key].append(val)
                            if dx in df_measure_type['perception']:
                                gathered_total_perception_score[key].append(val)
                            else:
                                gathered_total_arithmatic_score[key].append(val)
                    elif key in ['dx']:
                        gathered_total_dx_score[key] = 'total'
                        gathered_total_perception_score[key] = 'perception'
                        gathered_total_arithmatic_score[key] = 'measurement'
                    elif key in ['inference_type', 'model']:
                        gathered_total_dx_score[key] = val
                        gathered_total_perception_score[key] = val
                        gathered_total_arithmatic_score[key] = val
                    elif isinstance(val, list):
                        gathered_total_dx_score[key] = '/'
                        gathered_total_perception_score[key] = '/'
                        gathered_total_arithmatic_score[key] = '/'
                    else:
                        if key not in gathered_total_dx_score:gathered_total_dx_score[key] = []
                        if key not in gathered_total_perception_score:gathered_total_perception_score[key] = []
                        if key not in gathered_total_arithmatic_score:gathered_total_arithmatic_score[key] = []

        exist_dx_perception = set(df_measure_type['perception']).intersection(df_results_total.keys())
        exist_dx_measurement = set(df_measure_type['measurement']).intersection(df_results_total.keys())

        gathered_results_lst = [gathered_total_dx_score]
        if exist_dx_perception:
            gathered_results_lst.append(gathered_total_perception_score)
        if exist_dx_measurement:
            gathered_results_lst.append(gathered_total_arithmatic_score)

        key_order = df_results_total[list(df_results_total.keys())[0]].keys()
        for idx, gathered_results in enumerate(gathered_results_lst):
            for key in key_order:
                assert key in gathered_results
                value = gathered_results[key]
                if not isinstance(value, str):
                    gathered_results[key] = np.array(value).mean()

        df_results_total['total'] = gathered_total_dx_score
        if exist_dx_perception:
            df_results_total['perception'] = gathered_total_perception_score
        if exist_dx_measurement:
            df_results_total['measurement'] = gathered_total_arithmatic_score

        for dx_type, df_results in df_results_total.items():
            for key in df_results:
                df_scores_total[key].append(df_results[key])
    df_scores_total = pd.DataFrame(df_scores_total)
    return df_scores_total

def print_scores(df_scores, target_score_lst, step):
    metric_names = ['Completion', 'Depth', 'Consistency', 'Alignment']
    metric_by_score = defaultdict(list)
    for dx in df_measure_type['total']:
        for idx, target_score in enumerate(target_score_lst):
            target_df = df_scores[df_scores['dx'] == dx]
            if len(target_df):
                if 'wilson' in target_score:
                    num_correct = target_score.replace('wilson_score', 'num_correct')
                    num_correct_value = target_df[num_correct].values[0]
                    if not isinstance(num_correct_value, str):
                        if num_correct_value == 0:
                            score = 0.0
                            refined_score = 0.0
                        else:
                            score = target_df[target_score]
                            lower = target_df[target_score.replace('wilson_score', 'lower_bound')]
                            upper = target_df[target_score.replace('wilson_score', 'upper_bound')]
                            adjusted = (score * (1 - ((upper - lower) / 2))).values[0]
                            score = round(float(adjusted) * 100, 2)
                            if dx in df_measure_type['measurement'] and f'{target_score}_refined' in df_scores.columns:
                                if step:
                                    refined_score = round(adjusted * target_df[f'naive_score_{step}_measurement_matching'].values[0] * 100, 2)
                                else:
                                    refined_score = round(adjusted * target_df[f'naive_score_measurement_matching'].values[0] * 100, 2)
                            else:
                                refined_score = score
                    else:
                        score, refined_score = 'N/A', 'N/A'
                else:
                    score = float(target_df[target_score].values[0])
                    if dx in df_measure_type['measurement'] and f'{target_score}_refined' in df_scores.columns:
                        refined_score = float(round(target_df[f'{target_score}_refined'].values[0], 2))
                    else:
                        refined_score = score
            else:
                score, refined_score = 'N/A', 'N/A'
            score = str(score) if str(score) != 'nan' else 'N/A'
            refined_score = str(refined_score) if str(refined_score) != 'nan' else 'N/A'
            metric_by_score[f'{metric_names[idx]}_score'].append(score)
            if metric_names[idx] in ['Completion', 'Depth', 'Alignment']:
                metric_by_score[f'{metric_names[idx]}_refined'].append(refined_score)

    for metric, scores in metric_by_score.items():
        scores = [eval(score) for score in scores if score != 'N/A']
        mean = np.mean(scores).round(2) if scores else 'N/A'
        print(f'{metric}: {mean}')

if __name__ == '__main__':
    args = config()

    config_file = os.path.join(args.saved_dir_inference, 'config.json')
    with open(config_file, "r") as f:
        loaded_dict = json.load(f)
    vars(args).update(loaded_dict)

    inference_path = args.evaluation_path

    df_results = calculate_metrics(args, inference_path)

    df_scores = gather_scores(inference_path, df_results)

    if inference_path in ['reasoning']:
        target_score_lst = ['wilson_score_binary', 'stage_score', 'wilson_score_measurement_matching', 'wilson_score_consistency']
        print('Path1')
        print_scores(df_scores, target_score_lst, None)
    else:
        for step, path in zip(['g', 'r'], ['Path2', 'Re-evaluted Path1']):
            target_score_lst = [f'wilson_score_{step}_binary', f'{step}_stage_score', f'wilson_score_{step}_measurement_matching']
            if step in ['r']:
                target_score_lst += [f'wilson_score_{step}_consistency']
            print(path)
            print_scores(df_scores, target_score_lst, step)

