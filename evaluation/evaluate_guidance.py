import os
import json
import time
import random
import argparse
from glob import glob
from tqdm import tqdm

from utils import set_seed, set_gcp_env, dx_task_measurement, dx_task_multi_bodyparts
from prompt import stage_by_sysmsg
from scoring import return_scoring_result, return_measured_value_result

from model_cards import load_model_n_prosessor, inference_vllms


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', default='path/to/saved/config.json', type=str)


    args = parser.parse_args()
    return args

def load_jsonl_if_exists(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    return []

if __name__ == '__main__':
    args = config()

    with open(args.config_path, "r") as f:
        loaded_dict = json.load(f)

    vars(args).update(loaded_dict)
    args.evaluation_path = 'guidance'

    args_dict = vars(args)

    save_dir_guidance = os.path.join(args.save_base_dir, 'inference', 'guidance', args.model_id)
    os.makedirs(save_dir_guidance, exist_ok=True)
    with open(f"{save_dir_guidance}/config.json", "w") as file:
        json.dump(args_dict, file, indent=4)

    set_gcp_env(args)
    set_seed(args.seed)

    saved_dir_reasoning = args.save_dir_reasoning
    saved_dir_scoring = f"{args.save_base_dir}/scoring/reasoning/{args.model_id4scoring}/{args.model_id}"

    model, processor = load_model_n_prosessor(args, args.model_id, args.model_path)

    with open(args.dx_by_dicoms_file, "r") as file:
        dx_by_dicoms = json.load(file)

    for run_guidance_type in ['idk_init', 'idk_custom']:
        saved_dir_scoring_reasoning_list = glob(os.path.join(args.save_base_dir, 'scoring', 'reasoning',
                                                             args.model_id4scoring, args.model_id, '*'))
        for saved_dir_scoring_reasoning in saved_dir_scoring_reasoning_list:
            target_dx = saved_dir_scoring_reasoning.split('/')[-1]

            correct_dicom_final = load_jsonl_if_exists(os.path.join(saved_dir_scoring_reasoning, "dicom_lst_correct_final.jsonl"))
            dicom_lst_idk_init = load_jsonl_if_exists(os.path.join(saved_dir_scoring_reasoning, "dicom_lst_idk_init.jsonl"))
            dicom_lst_idk_custom = load_jsonl_if_exists(os.path.join(saved_dir_scoring_reasoning, "dicom_lst_idk_custom.jsonl"))

            target_dicom_lst = dicom_lst_idk_init if 'idk_init' in run_guidance_type else dicom_lst_idk_custom
            if target_dicom_lst:
                save_dir_per_dx = os.path.join(args.save_base_dir, 'inference', 'guidance', args.model_id, run_guidance_type, target_dx)
                save_dir_per_dx_scoring = os.path.join(args.save_base_dir, 'scoring', 'guidance', args.model_id4scoring, args.model_id, run_guidance_type, target_dx)

                os.makedirs(save_dir_per_dx, exist_ok=True)
                os.makedirs(save_dir_per_dx_scoring, exist_ok=True)

                saved_dicom_lst = [path.split('/')[-1].split('.')[0] for path in glob(f"{save_dir_per_dx}/*.json")]
                remained_dicom_lst = list(set(target_dicom_lst).difference(saved_dicom_lst))
                for idxx, dicom in tqdm(enumerate(remained_dicom_lst), total=len(remained_dicom_lst), desc=f'{run_guidance_type}_{target_dx}_{args.model_id}'):
                    review_candidates = list(set(dx_by_dicoms[target_dx]).difference(correct_dicom_final))
                    review_candidates.remove(dicom)
                    if not review_candidates:
                        review_candidates = correct_dicom_final
                    # ====================================================================
                    #                       Prepare Chat History
                    # ====================================================================
                    responses_reasoning_fname = f"{saved_dir_reasoning}/{target_dx}/{dicom}.json"
                    with open(responses_reasoning_fname, "r") as file:
                        responses_reasoning = json.load(file)

                    system_message = responses_reasoning['system_message']

                    if 'healthgpt' in args.model_id.lower():
                        from llava import conversation as conversation_lib
                        conv = conversation_lib.conv_templates["phi4_instruct"].copy()

                    if run_guidance_type in ['idk_init']:
                        init_query = responses_reasoning['stage-init']['query']
                        init_response = responses_reasoning['stage-init']['response']
                        init_answer = responses_reasoning['stage-init']['answer']

                        qa_init_file = f'{args.qa_base_dir}/{target_dx}/path1/init/basic/{dicom}.json'
                        with open(qa_init_file, "r") as file:
                            qa_init = json.load(file)
                        init_img_path = [f'{args.mimic_cxr_base}/{path}' for path in qa_init['img_path']]
                        chat_history = [[init_query, init_img_path, init_response]]

                        if 'healthgpt' in args.model_id.lower():
                            conv.append_message(conv.roles[0], init_query)
                            conv.append_message(conv.roles[1], init_response)

                        answer_per_stage = {'reasoning-init': init_answer}

                    elif run_guidance_type in ['idk_custom']:
                        chat_history = []
                        answer_per_stage = {}
                        qa_init_file = f'{args.qa_base_dir}/{target_dx}/path1/init/basic/{dicom}.json'
                        with open(qa_init_file, "r") as file:
                            qa_init = json.load(file)
                        init_img_path = [f'{args.mimic_cxr_base}/{path}' for path in qa_init['img_path']]
                        for stage in responses_reasoning.keys():
                            if stage.startswith(('stage-init', 'stage-criteria', 'stage-custom_criteria')):
                                if stage.startswith('stage-init'):
                                    chat_history.append([responses_reasoning[stage]['query'],
                                                         init_img_path,
                                                         responses_reasoning[stage]['response']])
                                else:
                                    chat_history.append([responses_reasoning[stage]['query'],
                                                         responses_reasoning[stage]['img_path'],
                                                         responses_reasoning[stage]['response']])

                                if 'healthgpt' in args.model_id.lower():
                                    conv.append_message(conv.roles[0], responses_reasoning[stage]['query'])
                                    conv.append_message(conv.roles[1], responses_reasoning[stage]['response'])

                                answer_per_stage[f"reasoning-{stage.split('-')[-1]}"] = responses_reasoning[stage]['answer']

                    scoring_per_stage = {}
                    sysmsg4scoring_per_stage = {}
                    # ====================================================================
                    #                       Question - Guidance - Body Part
                    # ====================================================================
                    qa_bodypart_file = f'{args.qa_base_dir}/{target_dx}/path2/stage1/basic/{dicom}.json'
                    with open(qa_bodypart_file, "r") as file:
                        qa_bodypart = json.load(file)

                    score_bodypart_lst = []

                    for idx, q_bodypart in enumerate(qa_bodypart['question']):
                        img_path_lst_bodypart = [f'{args.segmask_base_dir}{path}' for path in qa_bodypart['img_path']] if idx == 0 else []
                        response_bodypart, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                         query=q_bodypart,
                                                                                         img_path_lst=img_path_lst_bodypart,
                                                                                         system_message=system_message,
                                                                                         chat_history=chat_history)
                        answer_per_stage[f'guidance-bodypart_{idx}'] = qa_bodypart['answer'][idx]

                        score_bodypart_sub, res_idx_body_sub = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['bodypart_one'],
                                                                                     q_bodypart,
                                                                                     qa_bodypart['answer'][idx],
                                                                                     response_bodypart)
                        score_bodypart_lst.append(score_bodypart_sub)
                        scoring_per_stage[f'guidance-bodypart_{idx}'] = score_bodypart_sub
                        sysmsg4scoring_per_stage[f'guidance-bodypart_{idx}'] = stage_by_sysmsg['bodypart_one']

                        if res_idx_body_sub is not None:
                            chat_history[-1][-1] = chat_history[-1][-1][res_idx_body_sub]

                    if len(score_bodypart_lst) == sum(score_bodypart_lst):
                        # ====================================================================
                        #                       Question - Guidance - Measurement
                        # ====================================================================
                        qa_measurement_file = f'{args.qa_base_dir}/{target_dx}/path2/stage2/basic/{dicom}.json'
                        with open(qa_measurement_file, "r") as file:
                            qa_measurement = json.load(file)

                        q_measurement = qa_measurement['question']
                        img_path_lst_measurement = [f'{args.pnt_base_dir}{path}' for path in qa_measurement['img_path']]
                        response_measurement, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                            query=q_measurement,
                                                                                            img_path_lst=img_path_lst_measurement,
                                                                                            system_message=system_message,
                                                                                            chat_history=chat_history)
                        answer_per_stage[f'guidance-measurement'] = qa_measurement['answer']

                        score_measurement, res_idx_measure = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['measurement'],
                                                                                   qa_measurement['question'],
                                                                                   qa_measurement['answer'],
                                                                                   response_measurement)

                        if res_idx_measure is not None:
                            chat_history[-1][-1] = chat_history[-1][-1][res_idx_measure]

                        scoring_per_stage[f'guidance-measurement'] = score_measurement
                        sysmsg4scoring_per_stage[f'guidance-measurement'] = stage_by_sysmsg['measurement']

                        if score_measurement == 1:
                            # ====================================================================
                            #                       Question - Guidance - Final Diagnosis
                            # ====================================================================
                            qa_final_file = f'{args.qa_base_dir}/{target_dx}/path2/stage3/basic/{dicom}.json'

                            with open(qa_final_file, "r") as file:
                                qa_final = json.load(file)
                            measured_value = '' # qa_final['measured_value']
                            response_final, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                          query=qa_final['question'],
                                                                                          img_path_lst=[],
                                                                                          system_message=system_message,
                                                                                          chat_history=chat_history)
                            answer_per_stage['guidance-final'] = qa_final['answer']

                            score_final, res_idx_final = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['final'],
                                                                               [qa_final['question']],
                                                                               qa_final['answer'], response_final)

                            if res_idx_final is not None:
                                chat_history[-1][-1] = chat_history[-1][-1][res_idx_final]
                                response_final = response_final[res_idx_final]

                            scoring_per_stage[f'guidance-final'] = score_final
                            sysmsg4scoring_per_stage[f'guidance-final'] = stage_by_sysmsg['final']


                            sysmsg_value_extraction = stage_by_sysmsg['extract_value_projection'] if target_dx in ['projection'] else stage_by_sysmsg['extract_value']
                            model_measured_value = return_measured_value_result(args.model_id4scoring, sysmsg_value_extraction, response_final) if target_dx in dx_task_measurement else ''

                            scoring_per_stage[f'measured_value_guidance'] = model_measured_value
                            sysmsg4scoring_per_stage[f'measured_value_guidance'] = sysmsg_value_extraction

                            if score_final == 1:
                                # ====================================================================
                                #                       Question - Review After Guidance - Init Diagnosis
                                # ====================================================================
                                review_dicom = random.choice(review_candidates)
                                qa_init_file = f'{args.qa_base_dir}/{target_dx}/re-path1/init/basic/{review_dicom}.json'
                                with open(qa_init_file, "r") as file:
                                    qa_init = json.load(file)
                                qa_init_img_path = [f'{args.mimic_cxr_base}/{path}' for path in qa_init['img_path']]
                                response_init, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                             query=qa_init['question'],
                                                                                             img_path_lst=qa_init_img_path,
                                                                                             system_message=system_message,
                                                                                             chat_history=chat_history)

                                answer_per_stage['review-init'] = qa_init['answer']

                                score_review_init, res_idx_r_init = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['final'],
                                                                                          [qa_init['question']],
                                                                                          qa_init['answer'], response_init)

                                if res_idx_r_init is not None:
                                    chat_history[-1][-1] = chat_history[-1][-1][res_idx_r_init]

                                scoring_per_stage['review-init'] = score_review_init
                                sysmsg4scoring_per_stage['review-init'] = stage_by_sysmsg['final']

                                if score_review_init == 1:
                                    # ====================================================================
                                    #                       Question - Review - Criteria
                                    # ====================================================================
                                    qa_criteria_file = \
                                    glob(f'{args.qa_base_dir}/{target_dx}/re-path1/stage1/*/{review_dicom}.json')[0]
                                    with open(qa_criteria_file, "r") as file:
                                        qa_criteria = json.load(file)

                                    score_review_criteria_lst = []
                                    for idx, (q_criteria, a_criteria) in enumerate(zip(qa_criteria['question'], qa_criteria['answer'])):
                                        response_review_criteria_sub, chat_history = inference_vllms(args.model_id)(args, model,
                                                                                                                    processor,
                                                                                                                    query=q_criteria,
                                                                                                                    img_path_lst=[],
                                                                                                                    system_message=system_message,
                                                                                                                    chat_history=chat_history)
                                        answer_per_stage[f'review-criteria_{idx}'] = a_criteria

                                        score_review_criteria_sub, res_idx_r_c = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['criteria'],
                                                                                                       q_criteria, a_criteria, response_review_criteria_sub)
                                        score_review_criteria_lst.append(score_review_criteria_sub)
                                        scoring_per_stage[f'review-criteria_{idx}'] = score_review_criteria_sub
                                        sysmsg4scoring_per_stage[f'review-criteria_{idx}'] = stage_by_sysmsg['criteria']

                                        if res_idx_r_c is not None:
                                            chat_history[-1][-1] = chat_history[-1][-1][res_idx_r_c]

                                    if len(score_review_criteria_lst) == sum(score_review_criteria_lst):
                                        # ====================================================================
                                        #                       Question - Review - Body Part
                                        # ====================================================================
                                        qa_bodypart_file = glob(f'{args.qa_base_dir}/{target_dx}/re-path1/stage2/*/{review_dicom}.json')[0]
                                        with open(qa_bodypart_file, "r") as file:
                                            qa_bodypart = json.load(file)

                                        score_review_body_lst = []
                                        for idx, q_bodypart in enumerate(qa_bodypart['question']):
                                            img_path_lst_bodypart = [f'{args.segmask_base_dir}{path}' for path in qa_bodypart['img_path'][idx]]

                                            response_review_body, chat_history = inference_vllms(args.model_id)(args, model,
                                                                                                                processor,
                                                                                                                query=q_bodypart,
                                                                                                                img_path_lst=img_path_lst_bodypart,
                                                                                                                system_message=system_message,
                                                                                                                chat_history=chat_history)
                                            answer_per_stage[f'review-bodypart_{idx}'] = qa_bodypart['answer'][idx]

                                            sys_msg_bodypart = stage_by_sysmsg['bodypart_all'] if target_dx in dx_task_multi_bodyparts else stage_by_sysmsg['bodypart_one']

                                            score_review_body_sub, res_idx_r_b = return_scoring_result(args.model_id4scoring, sys_msg_bodypart,
                                                                                                       q_bodypart, qa_bodypart['answer'][idx],
                                                                                                       response_review_body)
                                            score_review_body_lst.append(score_review_body_sub)
                                            scoring_per_stage[f'review-bodypart_{idx}'] = score_review_body_sub
                                            sysmsg4scoring_per_stage[f'review-bodypart_{idx}'] = sys_msg_bodypart

                                            if res_idx_r_b is not None:
                                                chat_history[-1][-1] = chat_history[-1][-1][res_idx_r_b]

                                        if len(score_review_body_lst) == sum(score_review_body_lst):

                                            # ====================================================================
                                            #                       Question - Review - Measurement
                                            # ====================================================================
                                            qa_measurement_file = f'{args.qa_base_dir}/{target_dx}/re-path1/stage3/basic/{review_dicom}.json'
                                            with open(qa_measurement_file, "r") as file:
                                                qa_measurement = json.load(file)

                                            response_review_measure, chat_history = inference_vllms(args.model_id)(args, model,
                                                                                                                   processor,
                                                                                                                   query=qa_measurement['question'],
                                                                                                                   img_path_lst=[],
                                                                                                                   system_message=system_message,
                                                                                                                   chat_history=chat_history)
                                            answer_per_stage['review-measurement'] = qa_measurement['answer']

                                            sys_msg_measurment = stage_by_sysmsg['measurement_projection'] if (target_dx in 'projection') else stage_by_sysmsg['measurement']

                                            score_review_measure, res_idx_r_measure = return_scoring_result(args.model_id4scoring, sys_msg_bodypart,
                                                                                                            qa_measurement['question'],
                                                                                                            qa_measurement['answer'],
                                                                                                            response_review_measure)

                                            if res_idx_r_measure is not None:
                                                chat_history[-1][-1] = chat_history[-1][-1][res_idx_r_measure]

                                            scoring_per_stage[f'review-measurement'] = score_review_measure
                                            sysmsg4scoring_per_stage[f'review-measurement'] = sys_msg_bodypart

                                            if score_review_measure == 1:
                                                # ====================================================================
                                                #                       Question - Review - Final
                                                # ====================================================================
                                                qa_final_file = f'{args.qa_base_dir}/{target_dx}/re-path1/stage4/basic/{review_dicom}.json'
                                                with open(qa_final_file, "r") as file:
                                                    qa_final = json.load(file)
                                                response_review_final, chat_history = inference_vllms(args.model_id)(args,
                                                                                                                     model,
                                                                                                                     processor,
                                                                                                                     query=qa_final['question'],
                                                                                                                     img_path_lst=[],
                                                                                                                     system_message=system_message,
                                                                                                                     chat_history=chat_history)
                                                answer_per_stage['review-final'] = qa_final['answer']

                                                score_review_final, res_idx_r_final = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['final'],
                                                                                                            [qa_final['question']], qa_final['answer'], response_review_measure)
                                                if res_idx_r_final is not None:
                                                    chat_history[-1][-1] = chat_history[-1][-1][res_idx_r_final]
                                                    response_review_final = response_review_final[res_idx_r_final]

                                                scoring_per_stage[f'review-final'] = score_review_final
                                                sysmsg4scoring_per_stage[f'review-final'] = stage_by_sysmsg['final']

                                                sysmsg_value_extraction = stage_by_sysmsg['extract_value_projection'] if target_dx in ['projection'] else stage_by_sysmsg['extract_value']

                                                model_measured_value = return_measured_value_result(args.model_id4scoring, sysmsg_value_extraction, response_review_final) if target_dx in dx_task_measurement else ''

                                                scoring_per_stage[f'measured_value_review'] = model_measured_value
                                                sysmsg4scoring_per_stage[f'measured_value_review'] = sysmsg_value_extraction

                    # ====================================================================
                    #                       SAVE Result
                    # ====================================================================
                    result = {'dicom': dicom,
                              'system_message': system_message,
                              'cxr_path': init_img_path[0]}

                    if 'guidance-final' in scoring_per_stage:
                        result['measured_value'] = measured_value
                    if 'review-init' in scoring_per_stage:
                        result['dicom_review'] = review_dicom
                        result['cxr_path_review'] = qa_init_img_path[0]
                    if 'review-final' in scoring_per_stage:
                        result['measured_value_review'] = ''# qa_init['measured_value']

                    for stage, history in zip(answer_per_stage, chat_history):
                        hist = {f'stage-{stage}': {'query': history[0], 'img_path': history[1],
                                                   'response': history[-1], 'answer': answer_per_stage[stage]}}

                        result.update(hist)

                    with open(f"{save_dir_per_dx}/{dicom}.json", "w") as file:
                        json.dump(result, file, indent=4)

                    result_scoring = {}
                    for stage, score in scoring_per_stage.items():
                        result_scoring[f'stage-{stage}'] = score

                    for stage, sysmsg in sysmsg4scoring_per_stage.items():
                        result_scoring[f'sysmsg-{stage}'] = sysmsg

                    with open(f"{save_dir_per_dx_scoring}/{dicom}.json", "w") as file:
                        json.dump(result_scoring, file, indent=4)
