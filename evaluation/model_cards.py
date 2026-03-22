import base64
from PIL import Image
from io import BytesIO

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from transformers import Qwen2_5_VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration

from google import genai
import google.genai.types as types

from openai import AzureOpenAI

def resize_image_based_on_wider_side(image_path, new_size):
    img = Image.open(image_path).convert('RGB')
    original_width, original_height = img.size

    if ((original_width >= original_height) and original_width > new_size) \
            or ((original_width < original_height) and original_height > new_size):
        # Determine which side is wider
        if original_width > original_height:
            # Resize based on width
            aspect_ratio = original_height / original_width
            new_width = new_size
            new_height = int(new_width * aspect_ratio)
        else:
            # Resize based on height
            aspect_ratio = original_width / original_height
            new_height = new_size
            new_width = int(new_height * aspect_ratio)
        img_resized = img.resize((new_width, new_height))
        return img_resized
    else:
        return img

def load_model_n_prosessor(args, model_id, model_path):
    if 'HealthGPT' in model_id:
        import os, sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        import copy
        from dataclasses import dataclass, field
        import json
        import logging
        import pathlib
        from typing import Dict, Optional, Sequence, List
        # import torch
        import transformers
        import tokenizers
        # from llava.model import *
        from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig
        from PIL import Image
        from packaging import version
        IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
        from utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights

        model_dtype = torch.float16

        model = LlavaPhiForCausalLM.from_pretrained(
            pretrained_model_name_or_path="microsoft/Phi-4",
            attn_implementation=None,
            torch_dtype=model_dtype,
        )

        from llava.peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.0,
            bias='none',
            task_type="CAUSAL_LM",
            lora_nums=4,
        )
        model = get_peft_model(model, lora_config)

        processor = transformers.AutoTokenizer.from_pretrained(
            "microsoft/Phi-4",
            padding_side="right",
            use_fast=False,
        )
        num_new_tokens = add_special_tokens_and_resize_model(processor, model, 8192)
        print(f"Number of new tokens added for unified task: {num_new_tokens}")

        from utils import com_vision_args
        com_vision_args.model_name_or_path = "microsoft/Phi-4"
        com_vision_args.vision_tower = "openai/clip-vit-large-patch14-336"
        com_vision_args.version = "phi4_instruct"

        model.get_model().initialize_vision_modules(model_args=com_vision_args)
        model.get_vision_tower().to(dtype=model_dtype)

        model = load_weights(model, f"{model_path}/com_hlora_weights_phi4.bin")
        model.eval()
        model.to(model_dtype).cuda()

    elif model_id in ["KrauthammerLab/RadVLM"]:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_path,
                                                                       torch_dtype=torch.float16,
                                                                       low_cpu_mem_usage=True).to('cuda')
        processor = AutoProcessor.from_pretrained(model_path)

    elif model_id in ['google/medgemma-4b-it', 'google/medgemma-27b-it']:
        model = AutoModelForImageTextToText.from_pretrained(model_path,
                                                            torch_dtype=torch.bfloat16,
                                                            device_map="auto")
        processor = AutoProcessor.from_pretrained(model_path)

    elif 'gemini' in model_id:
        model = genai.Client(http_options=types.HttpOptions(api_version="v1"))
        processor = model_id

    elif "gpt-4.1" in model_id:
        model = AzureOpenAI(api_version=args.gpt_api_version,
                            azure_endpoint=args.gpt_endpoint,
                            api_key=args.gpt_api_key)
        processor = model_id

    else:
        from vllm import LLM
        from vllm.sampling_params import SamplingParams
        model = LLM(model=model_path,
                    max_model_len=30000,
                    tensor_parallel_size=args.tensor_parallel_size,
                    limit_mm_per_prompt={"image": 20},
                    disable_log_stats=True)

        if args.shot is not None:
            processor = SamplingParams(max_tokens=8192, temperature=0.7, top_p=0.95, n=args.shot)
        else:
            processor = SamplingParams(max_tokens=8192, temperature=0.0)

    return model, processor

def inference_vllm(args, model, sampling_params, query, img_path_lst, system_message, chat_history):
    def load_image(image_path):
        if isinstance(image_path, str):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            buff = BytesIO()
            image_path.save(buff, format="JPEG")
            return base64.b64encode(buff.getvalue()).decode('utf-8')

    conversation = []
    map_extension = {'jpg': 'jpeg', 'png': 'png'}
    # append system message
    if system_message is not None:
        conversation.append({"role": "system", "content": system_message,})

    # append chat history, if applicable
    for idx, (query_t, query_i, response_t) in enumerate(chat_history):
        conversation.extend([
            {"role": "user",
             "content": [{"type": "text", "text": query_t}] + [{"type": "image_url", "image_url": {"url": f"data:image/{map_extension[img_path.split('.')[-1]]};base64,{load_image(img_path)}"}} for img_path in query_i]},
            {"role": "assistant",
             "content": [{"type": "text", "text": response_t}]}
        ])

    # append current query
    conversation.append(
        {"role": "user",
         "content": [{"type": "text", "text": query}] + [{"type": "image_url", "image_url": {"url": f"data:image/{map_extension[img_path.split('.')[-1]]};base64,{load_image(img_path)}"}} for img_path in img_path_lst]}
    )

    with torch.inference_mode():
        outputs = model.chat(conversation, sampling_params=sampling_params, use_tqdm=False)

    if args.shot is not None:
        response = [outputs[0].outputs[idx].text for idx in range(args.shot)]
    else:
        response = outputs[0].outputs[0].text

    chat_history.append([query, img_path_lst, response])

    return response, chat_history

def inference_healthgpt(args, model, processor, query, img_path_lst, system_message, chat_history):
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    conv = args.conv
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    if img_path_lst:
        query = f"{DEFAULT_IMAGE_TOKEN * len(img_path_lst)}" + '\n' + query

    conv.system = system_message
    if chat_history:
        conv.append_message(conv.roles[1], chat_history[-1][-1])
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    img_path_lst_hist = []
    for idx, (query_t, query_i, response_t) in enumerate(chat_history):
        img_path_lst_hist.extend(query_i)

    img_tensor_lst = []
    for img_path in img_path_lst_hist + img_path_lst:
        image = Image.open(img_path).convert('RGB')
        image = expand2square(image, tuple(int(x * 255) for x in model.get_vision_tower().image_processor.image_mean))
        image_tensor = model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0)
        img_tensor_lst.append(image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True))

    input_ids = tokenizer_image_token(prompt, processor, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda().unsqueeze_(0)

    model.eval()
    with torch.inference_mode():
        if args.shot is not None:
            response = []
            for _ in range(args.shot):
                output_ids = model.base_model.model.generate(
                    input_ids,
                    images=img_tensor_lst if img_tensor_lst else None,
                    image_sizes=image.size if img_tensor_lst else None,
                    do_sample=True, temperature=0.7, top_p=0.95, num_beams=1, max_new_tokens=1024, use_cache=True)
                res = processor.decode(output_ids[0], skip_special_tokens=True)
                response.append(res)
        else:
            output_ids = model.base_model.model.generate(
                input_ids,
                images=img_tensor_lst if img_tensor_lst else None,
                image_sizes=image.size if img_tensor_lst else None,
                do_sample=False, temperature=0.0, top_p=None, num_beams=1, max_new_tokens=1024, use_cache=True)
            response = processor.decode(output_ids[0], skip_special_tokens=True)
    conv.messages.pop()
    chat_history.append([query, img_path_lst, response])

    return response, chat_history

def inference_hf(args, model, processor, query, img_path_lst, system_message, chat_history):
    conversation = []

    # append system message
    if system_message is not None:
        conversation.append({"role": "system", "content": system_message})

    # append chat history, if applicable
    img_path_lst_hist = []
    for idx, (query_t, query_i, response_t) in enumerate(chat_history):
        img_path_lst_hist.extend(query_i)
        conversation.extend([
            {"role": "user", "content": [{"type": "text", "text": query_t}] + [{"type": "image"} for _ in query_i]},
            {"role": "assistant",
             "content": [{"type": "text", "text": response_t}]}
        ])

    # append current query
    conversation.append({"role": "user",
                         "content": [{"type": "text", "text": query}] + [{"type": "image"} for _ in img_path_lst]
                         })

    img_lst = []
    for img_path in img_path_lst_hist + img_path_lst:
        img = resize_image_based_on_wider_side(img_path, new_size=args.img_size)
        img_lst.append(img)

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    if img_lst:
        inputs = processor(images=img_lst, text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    else:
        inputs = processor(text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)

    with torch.inference_mode():
        if args.shot is not None:
            output = model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=0.7, top_p=0.95, num_return_sequences=args.shot)
            response = processor.batch_decode(output[:, inputs['input_ids'].size(1):-1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            output = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            response = processor.decode(output[0][inputs['input_ids'].size(1):-1])
    chat_history.append([query, img_path_lst, response])

    return response, chat_history

def inference_gemini(args, client, model_id, query, img_path_lst, system_message, chat_history):
    history = []
    for (query_hist, img_path_lst_hist, response_hist) in chat_history:
        img_lst = []
        for img_path in img_path_lst_hist:
            img = resize_image_based_on_wider_side(img_path, new_size=args.img_size)
            img_lst.append(img)
        history.append(types.UserContent([query_hist] + img_lst))
        history.append(types.ModelContent(response_hist))

    if model_id in ['gemini-2.5-pro']:
        think = 1024
    else:
        think = 0
    if args.shot is not None:
        chat = client.chats.create(
            model=model_id,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                thinking_config=types.ThinkingConfig(thinking_budget=think),
                temperature=0.7, top_p=0.95, candidate_count=args.shot),
            history=history if len(history) else None
        )

    else:
        chat = client.chats.create(
            model=model_id,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                thinking_config=types.ThinkingConfig(thinking_budget=think),
                temperature=0.0),
            history=history if len(history) else None
        )


    current_message = [query]
    for img_path in img_path_lst:
        img = resize_image_based_on_wider_side(img_path, new_size=args.img_size)
        current_message.append(img)
    output = chat.send_message(current_message)

    if args.shot is not None:
        response = [candidate.content.parts[0].text for candidate in output.candidates]
    else:
        response = output.text

    chat_history.append([query, img_path_lst, response])
    return response, chat_history

def inference_gpt(args, client, model_id, query, img_path_lst, system_message, chat_history):
    import io
    import base64
    from mimetypes import guess_type

    def load_image(img_path, img_size):
        # Guess the MIME type of the image
        mime_type, _ = guess_type(img_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        img = resize_image_based_on_wider_side(img_path, img_size)

        # Save the image to a buffer in the same format as original
        buffered = io.BytesIO()
        img_format = img.format if img.format else 'PNG'  # Fallback to PNG
        img.save(buffered, format=img_format)
        base64_encoded_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return f"data:{mime_type};base64,{base64_encoded_data}"

    conversation = []
    # append system message
    if system_message is not None:
        conversation.append({"role": "system", "content": system_message})

    # append chat history, if applicable
    for idx, (query_t, query_i, response_t) in enumerate(chat_history):
        conversation.extend([
            {"role": "user",
             "content": [{"type": "text", "text": query_t}] +
                        [{"type": "image_url", "image_url": {"url": load_image(img_path, args.img_size)}} for img_path in query_i]
             },
            {"role": "assistant", "content": [{"type": "text", "text": response_t}]}
        ])

    # append current query
    conversation.append(
        {"role": "user",
         "content": [{"type": "text", "text": query}] +
                    [{"type": "image_url", "image_url": {"url": load_image(img_path, args.img_size)}} for img_path in img_path_lst]
         })

    if args.shot is not None:
        output = client.chat.completions.create(messages=conversation, max_completion_tokens=8192,
                                                temperature=0.7, top_p=0.95, n=args.shot,
                                                frequency_penalty=0.0, presence_penalty=0.0, model=model_id)
        response = [choice.message.content for choice in output.choices]

    else:
        output = client.chat.completions.create(messages=conversation, max_completion_tokens=8192,
                                                temperature=0.0, frequency_penalty=0.0, presence_penalty=0.0, model=model_id)
        response = output.choices[0].message.content

    chat_history.append([query, img_path_lst, response])
    return response, chat_history


def inference_vllms(model_id):
    """Return the appropriate inference function based on the model_id."""
    model_id_lower = model_id.lower()

    model_map = {
        'gemini': inference_gemini,
        'gpt': inference_gpt,
        'healthgpt': inference_healthgpt,
        'medgemma': inference_hf,
        'radvlm': inference_hf,
    }

    for keyword, func in model_map.items():
        if keyword in model_id_lower:
            return func

    return inference_vllm
