from collections import Counter

from google import genai
import google.genai.types as types


def inference_llm4scoring(model_id4scoring, system_message, prompt):
    client = genai.Client(http_options=types.HttpOptions(api_version="v1"))
    chat = client.chats.create(model=model_id4scoring,
                               config=types.GenerateContentConfig(
                                   system_instruction=system_message,
                                   max_output_tokens=500,
                                   temperature=0.0
                               ))
    response = chat.send_message(prompt)
    return response


def return_scoring_result(model_id4scoring, sysmsg, question, answer, response):
    if isinstance(response, list):
        score_lst = []
        for res in response:
            if isinstance(question, list):
                query = f"- Question: {question[0]}" \
                        f"- Answer 1: {answer} " \
                        f"- Answer 2: {res}"
            else:
                query = f"- Answer 1: {answer} " \
                        f"- Answer 2: {res}"
            output = inference_llm4scoring(model_id4scoring, sysmsg, query)
            if 'true' in output.text.lower():
                score = 1
            elif 'idk' in output.text.lower():
                score = -1
            elif 'n/a' in output.text.lower():
                score = -2
            else:
                score = 0

            score_lst.append(score)
        most_common = Counter(score_lst).most_common()
        if len(most_common) < 2 or most_common[0][1] > most_common[1][1]:
            major_score = most_common[0][0]
        else:
            if -1 in score_lst:
                major_score = -1
            else:
                major_score = 0
        return major_score, score_lst.index(major_score)

    else:
        if isinstance(question, list):
            query = f"- Question: {question[0]}" \
                    f"- Answer 1: {answer} " \
                    f"- Answer 2: {response}"
        else:
            query = f"- Answer 1: {answer} " \
                    f"- Answer 2: {response}"
        output = inference_llm4scoring(model_id4scoring, sysmsg, query)

        if 'true' in output.text.lower():
            score = 1
        elif 'idk' in output.text.lower():
            score = -1
        elif 'n/a' in output.text.lower():
            score = -2
        else:
            score = 0

        return score, None

def return_measured_value_result(model_id4scoring, sysmsg, response):
    query = f"- Response: {response}"
    output = inference_llm4scoring(model_id4scoring, sysmsg, query)
    return output.text