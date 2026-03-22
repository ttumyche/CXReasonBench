import os
import json
import argparse
from tqdm import tqdm
from glob import glob
from datetime import datetime

from utils import set_seed, set_gcp_env, dx_task_measurement, dx_task_multi_bodyparts
from prompt import system_message, stage_by_sysmsg
from scoring import return_scoring_result, return_measured_value_result

from model_cards import load_model_n_prosessor, inference_vllms


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--shot', default=None, type=int)
    parser.add_argument('--img_size', default=1024, type=int)
    parser.add_argument('--tensor_parallel_size', default=2, type=int)
    parser.add_argument('--evaluation_path', default='reasoning', type=str)
    parser.add_argument('--model_id4scoring', default="gemini-2.0-flash", type=str)


    parser.add_argument('--model_id', default="Qwen/Qwen2.5-VL-7B-Instruct", type=str,
                        choices=[
                            'gemini-2.5-flash-preview-04-17', 'gemini-2.5-flash', 'gemini-2.5-pro-preview-03-25',
                            'gemini-2.5-pro', 'gpt-4.1', "mistralai/Pixtral-Large-Instruct-2411",
                            "meta-llama/Llama-3.2-90B-Vision-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct",
                            "mistral-community/pixtral-12b", "Qwen/Qwen2.5-VL-7B-Instruct",
                            "HealthGPT-L14", "KrauthammerLab/RadVLM", 'google/medgemma-4b-it', 'google/medgemma-27b-it'
                        ])
    parser.add_argument('--model_path', default='path/to/saved_model_path', type=str)

    parser.add_argument('--cxreasonbench_base_dir', default="", type=str)
    parser.add_argument('--mimic_cxr_base', default="", type=str)

    parser.add_argument('--save_base_dir', default='result', type=str)

    parser.add_argument('--GOOGLE_CLOUD_LOCATION', default=None, type=str)
    parser.add_argument('--GOOGLE_CLOUD_PROJECT', default=None, type=str)
    parser.add_argument('--GOOGLE_GENAI_USE_VERTEXAI', default="True", type=str)
    parser.add_argument('--TOKENIZERS_PARALLELISM', default='false', type=str)

    parser.add_argument('--gpt_endpoint', default=None, type=str)
    parser.add_argument('--gpt_api_key', default=None, type=str)
    parser.add_argument('--gpt_api_version', default="2025-01-01-preview", type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = config()
    args_dict = vars(args)

    args.segmask_base_dir = os.path.join(args.cxreasonbench_base_dir, 'segmask_bodypart')
    args.pnt_base_dir = os.path.join(args.cxreasonbench_base_dir, 'pnt_on_cxr')
    args.qa_base_dir = os.path.join(args.cxreasonbench_base_dir, 'qa')
    args.dx_by_dicoms_file = os.path.join(args.cxreasonbench_base_dir, 'dx_by_dicoms.json')

    set_gcp_env(args)
    set_seed(args.seed)

    save_dir_reasoning = os.path.join(args.save_base_dir, 'inference', args.evaluation_path, args.model_id)
    save_dir_reasoning_scoring = os.path.join(args.save_base_dir, 'scoring', args.evaluation_path, args.model_id4scoring, args.model_id)

    args.save_dir_reasoning = save_dir_reasoning
    args.save_dir_reasoning_scoring = save_dir_reasoning_scoring

    os.makedirs(save_dir_reasoning, exist_ok=True)
    os.makedirs(save_dir_reasoning_scoring, exist_ok=True)

    diagnostic_task_list = ['aortic_knob_enlargement', 'ascending_aorta_enlargement',
                            'cardiomegaly', 'carina_angle', 'descending_aorta_enlargement',
                            'descending_aorta_tortuous', 'inclusion', 'inspiration',
                            'mediastinal_widening', 'projection', 'rotation', 'trachea_deviation']
    model, processor = load_model_n_prosessor(args, args.model_id, args.model_path)
    for diagnostic_task in tqdm(diagnostic_task_list, total=len(diagnostic_task_list)):
        save_dir_per_dx = os.path.join(save_dir_reasoning, diagnostic_task)
        save_dir_per_dx_scoring = os.path.join(save_dir_reasoning_scoring, diagnostic_task)
        os.makedirs(save_dir_per_dx, exist_ok=True)
        os.makedirs(save_dir_per_dx_scoring, exist_ok=True)
    
        with open(f"{save_dir_reasoning}/config.json", "w") as file:
            json.dump(args_dict, file, indent=4)
    
        # ====================================================================
        #                       Evaluation
        # ====================================================================
        with open(args.dx_by_dicoms_file, "r") as file:
            dx_by_dicoms = json.load(file)
        dicom_lst = dx_by_dicoms[diagnostic_task]
        saved_dicom_lst = [path.split('/')[-1].split('.')[0] for path in glob(f"{save_dir_per_dx}/*.json")]
        remained_dicom_lst = list(set(dicom_lst).difference(saved_dicom_lst))
    
        for dicom in tqdm(remained_dicom_lst, total=len(remained_dicom_lst), desc=f'{diagnostic_task}/{args.model_id}'):
            chat_history = []
            if 'HealthGPT' in args.model_id:
                from llava import conversation as conversation_lib
                args.conv = conversation_lib.conv_templates["phi4_instruct"].copy()
            # ===========================================================================
            #                       Init Question
            # ===========================================================================
            qa_init_file = f'{args.qa_base_dir}/{diagnostic_task}/path1/init/basic/{dicom}.json'
            with open(qa_init_file, "r") as file:
                qa_init = json.load(file)
            answer_per_stage = {'init': qa_init['answer']}
            measured_value = ''  # qa_init['measured_value']
            qa_init_img_path = [f'{args.mimic_cxr_base}/{path}' for path in qa_init['img_path']]
            response_init, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                query=qa_init['question'],
                                                                                img_path_lst=qa_init_img_path,
                                                                                system_message=system_message,
                                                                                chat_history=chat_history)
    
            score_init, res_idx_init = return_scoring_result(args.model_id4scoring, stage_by_sysmsg['init'],
                                                             [qa_init['question']], qa_init['answer'], response_init)
    
            if res_idx_init is not None:
                chat_history[-1][-1] = chat_history[-1][-1][res_idx_init]
    
            scoring_per_stage = {'init': score_init}
            sysmsg4scoring_per_stage = {'init': stage_by_sysmsg['init']}
            if score_init != -1:
                # ===========================================================================
                #                       Criteria
                # ===========================================================================
                qa_criteria_file = glob(f'{args.qa_base_dir}/{diagnostic_task}/path1/stage1/*/{dicom}.json')[0]
                with open(qa_criteria_file, "r") as file:
                    qa_criteria = json.load(file)
    
                score_criteria_lst = []
                for idx, (q_criteria, a_criteria) in enumerate(zip(qa_criteria['question'], qa_criteria['answer'])):
                    response_criteria, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                            query=q_criteria,
                                                                                            img_path_lst=[],
                                                                                            system_message=system_message,
                                                                                            chat_history=chat_history)
                    answer_per_stage[f'criteria_{idx}'] = a_criteria
    
                    score_criteria_sub, res_idx_criteria_sub = return_scoring_result(args.model_id4scoring,
                                                                                     stage_by_sysmsg['criteria'],
                                                                                     q_criteria, a_criteria,
                                                                                     response_criteria)
    
                    if res_idx_criteria_sub is not None:
                        chat_history[-1][-1] = chat_history[-1][-1][res_idx_criteria_sub]
    
                    score_criteria_lst.append(score_criteria_sub)
    
                    scoring_per_stage[f'criteria_{idx}'] = score_criteria_sub
                    sysmsg4scoring_per_stage[f'criteria_{idx}'] = stage_by_sysmsg['criteria']
                    if score_criteria_sub != 1:
                        break
    
                if len(score_criteria_lst) == sum(score_criteria_lst):
                    # ===========================================================================
                    #                   Custom Criteria
                    # ===========================================================================
                    if os.path.isdir(f'{args.qa_base_dir}/{diagnostic_task}/path1/stage1.5'):
                        qa_c_criteria_file = f'{args.qa_base_dir}/{diagnostic_task}/path1/stage1.5/basic/{dicom}.json'
                        with open(qa_c_criteria_file, "r") as file:
                            qa_c_criteria = json.load(file)
    
                        response_custom_criteria, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                                       query=qa_c_criteria['question'],
                                                                                                       img_path_lst=[],
                                                                                                       system_message=system_message,
                                                                                                       chat_history=chat_history)
                        answer_per_stage[f'custom_criteria'] = qa_c_criteria['answer']
    
                        score_custom_criteria, res_idx_custom = return_scoring_result(args.model_id4scoring,
                                                                                      stage_by_sysmsg['custom_criteria'],
                                                                                      qa_c_criteria['question'],
                                                                                      qa_c_criteria['answer'],
                                                                                      response_custom_criteria)
    
                        if res_idx_custom is not None:
                            chat_history[-1][-1] = chat_history[-1][-1][res_idx_custom]
    
                        scoring_per_stage[f'custom_criteria'] = score_custom_criteria
                        sysmsg4scoring_per_stage[f'custom_criteria'] = stage_by_sysmsg['custom_criteria']
    
                    else:
                        score_custom_criteria = 1
    
                    if score_custom_criteria == 1:
                        # ===========================================================================
                        #               Body Part
                        # ===========================================================================
                        qa_bodypart_file = glob(f'{args.qa_base_dir}/{diagnostic_task}/path1/stage2/*/{dicom}.json')[0]
                        with open(qa_bodypart_file, "r") as file:
                            qa_bodypart = json.load(file)
    
                        score_bodypart_lst = []
                        for idx, q_bodypart in enumerate(qa_bodypart['question']):
                            img_path_lst_bodypart = [f'{args.segmask_base_dir}{path}' for path in qa_bodypart['img_path'][idx]]
    
                            response_bodypart, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                                    query=q_bodypart,
                                                                                                    img_path_lst=img_path_lst_bodypart,
                                                                                                    system_message=system_message,
                                                                                                    chat_history=chat_history)
                            answer_per_stage[f'bodypart_{idx}'] = qa_bodypart['answer'][idx]
    
                            sys_msg_bodypart = stage_by_sysmsg['bodypart_all'] if diagnostic_task in dx_task_multi_bodyparts else stage_by_sysmsg['bodypart_one']
    
                            score_bodypart_sub, res_idx_body = return_scoring_result(args.model_id4scoring, sys_msg_bodypart,
                                                                                     q_bodypart, qa_bodypart['answer'][idx], response_bodypart)
                            if res_idx_body is not None:
                                chat_history[-1][-1] = chat_history[-1][-1][res_idx_body]
    
                            score_bodypart_lst.append(score_bodypart_sub)
    
                            scoring_per_stage[f'bodypart_{idx}'] = score_bodypart_sub
                            sysmsg4scoring_per_stage[f'bodypart_{idx}'] = sys_msg_bodypart
    
                            if score_bodypart_sub != 1:
                                break
    
                        if len(score_bodypart_lst) == sum(score_bodypart_lst):
                            # ===========================================================================
                            #               Measurement
                            # ===========================================================================
                            qa_measurement_file = f'{args.qa_base_dir}/{diagnostic_task}/path1/stage3/basic/{dicom}.json'
                            with open(qa_measurement_file, "r") as file:
                                qa_measurement = json.load(file)
    
                            response_measurement, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                                       query=qa_measurement['question'],
                                                                                                       img_path_lst=[],
                                                                                                       system_message=system_message,
                                                                                                       chat_history=chat_history)
                            answer_per_stage['measurement'] = qa_measurement['answer']
                            sys_msg_measurment = stage_by_sysmsg['measurement_projection'] if diagnostic_task in ['projection'] else stage_by_sysmsg['measurement']
                            score_measurement, res_idx_measure = return_scoring_result(args.model_id4scoring, sys_msg_measurment,
                                                                                       qa_measurement['question'],
                                                                                       qa_measurement['answer'],
                                                                                       response_measurement)
    
                            if res_idx_measure is not None:
                                chat_history[-1][-1] = chat_history[-1][-1][res_idx_measure]
    
                            scoring_per_stage[f'measurement'] = score_measurement
                            sysmsg4scoring_per_stage[f'measurement'] = sys_msg_measurment
    
                            if score_measurement == 1:
                                # ===========================================================================
                                #               Final
                                # ===========================================================================
                                qa_final_file = f'{args.qa_base_dir}/{diagnostic_task}/path1/stage4/basic/{dicom}.json'
                                with open(qa_final_file, "r") as file:
                                    qa_final = json.load(file)
                                response_final, chat_history = inference_vllms(args.model_id)(args, model, processor,
                                                                                                     query=qa_final['question'],
                                                                                                     img_path_lst=[],
                                                                                                     system_message=system_message,
                                                                                                     chat_history=chat_history)
                                answer_per_stage['final'] = qa_final['answer']
    
                                score_final, res_idx_final = return_scoring_result(args.model_id4scoring,
                                                                                   stage_by_sysmsg['final'],
                                                                                   [qa_final['question']],
                                                                                   qa_final['answer'],
                                                                                   response_final)
                                if res_idx_final is not None:
                                    chat_history[-1][-1] = chat_history[-1][-1][res_idx_final]
                                    response_final = response_final[res_idx_final]
    
                                scoring_per_stage[f'final'] = score_final
                                sysmsg4scoring_per_stage[f'final'] = stage_by_sysmsg['final'],
    
                                sysmsg_value_extraction = stage_by_sysmsg['extract_value_projection'] if diagnostic_task in ['projection'] else stage_by_sysmsg['extract_value']
                                model_measured_value = return_measured_value_result(args.model_id4scoring,sysmsg_value_extraction, response_final) if diagnostic_task in dx_task_measurement else ''
    
                                scoring_per_stage[f'measured_value'] = model_measured_value
                                sysmsg4scoring_per_stage[f'measured_value'] = sysmsg_value_extraction
    
            # ====================================================================
            #                       SAVE Result
            # ====================================================================
            result = {'dicom': dicom,
                      'system_message': system_message,
                      'cxr_path': qa_init['img_path'][0],
                      'measured_value': measured_value}
    
            assert len(answer_per_stage) == len(chat_history)
            for stage, history in zip(answer_per_stage, chat_history):
                hist = {f'stage-{stage}': {'query': history[0],
                                           'img_path': history[1],
                                           'response': history[-1],
                                           'answer': answer_per_stage[stage]}}
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
    
            if scoring_per_stage['init'] == -1:
                with open(f"{save_dir_per_dx_scoring}/dicom_lst_idk_init.jsonl", "a", encoding="utf-8") as f:
                    json.dump(dicom, f, ensure_ascii=False)
                    f.write("\n")
    
            if 'custom_criteria' in scoring_per_stage:
                if scoring_per_stage['custom_criteria'] == -1:
                    with open(f"{save_dir_per_dx_scoring}/dicom_lst_idk_custom.jsonl", "a", encoding="utf-8") as f:
                        json.dump(dicom, f, ensure_ascii=False)
                        f.write("\n")
    
            if 'final' in scoring_per_stage:
                if scoring_per_stage[f'final'] == 1:
                    with open(f"{save_dir_per_dx_scoring}/dicom_lst_correct_final.jsonl", "a", encoding="utf-8") as f:
                        json.dump(dicom, f, ensure_ascii=False)
                        f.write("\n")
