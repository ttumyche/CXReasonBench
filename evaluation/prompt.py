system_message = "You are a well-trained radiologist answering multiple-choice questions about a chest X-ray. " \
                     "Base your assessment solely on the chest X-ray itself, without considering " \
                     "labels or annotations in the upper corners. " \
                     "Rely purely on anatomical and radiographic features. " \
                     "Provide your response in the format: FINAL ANSWER: [YOUR FINAL ANSWER], " \
                     "including the letter of your choice followed by the selected option " \
                     "(e.g., FINAL ANSWER: (A) Yes). " \
                     "Always state your selected option first, then explain your reasoning if needed."


stage_by_sysmsg = {
    'init': "Task: Decide if Answer 2 and Answer 1 indicate the same choice, "
            "even if the option letter is missing or the wording is slightly different."
            "Rules: If Answer 2 selects 'I don't know', return 'IDK'. "
            "Otherwise, return 'True' if both answers select the same option letter (case insensitive), "
            "or if the same option is selected even when the letter is missing or "
            "the wording is slightly different. "
            "Also, return 'True' if the meaning of Answer 2 aligns with the intent of the question and Answer 1, "
            "even if the phrasing differs."
            "Otherwise, return 'False'."
            "Input format: "
            "- Question: A multiple-choice question with options."
            "- Answer 1: '[ANSWER]' (e.g., (a) Yes)"
            "- Answer 2: May include explanation and a final answer (e.g., FINAL ANSWER: (a) Yes). "
            "If 'FINAL ANSWER:' is not present, extract the final answer using phrases like 'The answer is'."
            "Output Format: 'IDK', 'True', or 'False'. No explanations.",

    'criteria': "Task: Decide if Answer 2 and Answer 1 indicate the same choice, even if the option letter is missing or the wording is slightly different. "
                "Rules: If Answer 2 selects 'None of the above', return 'N/A'. "
                "Otherwise, return 'True' if both answers select the same option letter (case insensitive), "
                "or if the same option is selected even when the letter is missing or "
                "the wording is slightly different."
                " Otherwise, return 'False'."
                "Input format: "
                "- Answer 1: '[ANSWER]' (e.g., (a) Yes)"
                "- Answer 2: May include explanation and a final answer (e.g., FINAL ANSWER: (a) Yes). "
                "If 'FINAL ANSWER:' is not present, extract the final answer using phrases like 'The answer is'."
                "Output Format: 'N/A', 'True', or 'False'. No explanations.",

    'custom_criteria': "Task: Decide if Answer 2 and Answer 1 indicate the same choice, even if the option letter is missing or the wording is slightly different."
                       "Rules: Return 'True' if both answers select the same option letter (case insensitive) "
                       "or the same option, even if the letter is missing."
                       "otherwise, return 'IDK'. "
                       "Input format: "
                       "- Answer 1: '[ANSWER]' (e.g., (a) Yes) "
                       "- Answer 2: May include explanation and a final answer (e.g., FINAL ANSWER: (a) Yes). "
                       "If 'FINAL ANSWER:' is not present, extract the final answer using phrases like 'The answer is'."
                       "Output Format: 'True', or 'IDK'. No explanations.",

    'bodypart_all': "Task: Decide if Answer 2 and Answer 1 indicate the same choice, "
                    "even if the option letter is missing or the wording is slightly different. "
                    "Rules: "
                    "- If Answer 2 selects 'None of the above', return 'N/A'. "
                    "- Return 'True' if both answers exactly select the same option letters (case insensitive), "
                    "or if the same options are selected even when the letters are missing or the wording is slightly different. "
                    "- Return 'False' if Answer 2 includes any incorrect options or misses any correct options."
                    "- Return  'Partial (X)' if all selected options in Answer 2 are correct, but only a subset of the correct answers is included. X is the number of correctly selected options. "
                    "Input Format: "
                    "- Answer 1: A response in the format '[ANSWER]' (e.g., '(a) 1st image, (c) 3rd image'). "
                    "- Answer 2: May include explanation and a final answer  (e.g., 'FINAL ANSWER: (a) 1st image, (c) 3rd image')."
                    "If 'FINAL ANSWER:' is not present, extract the final answer using phrases like 'The answer is'."
                    "Output Format: 'N/A', 'True, 'Partial (X)', or 'False'. No explanations.",

    'bodypart_one': "Task: Decide if Answer 2 and Answer 1 indicate the same choice, "
                    "even if the option letter is missing or the wording is slightly different. "
                    "Rules: "
                    "- If Answer 2 selects 'None of the above', return 'N/A'. "
                    "- Return 'True' if both answers exactlyselect the same option letter (case insensitive), "
                    "or if the same option is selected even when the letter is missing or the wording is slightly different."
                    "- Return 'False' if Answer 2 includes any incorrect options."
                    "Input format: "
                    "- Answer 1: '[ANSWER]' (e.g., (a) 1st image) "
                    "- Answer 2: May include explanation and a final answer (e.g., FINAL ANSWER: (a) 1st image), "
                    "If 'FINAL ANSWER:' is not present, extract the final answer using phrases like 'The answer is'."
                    "Output Format: 'N/A', 'True', or 'False'. No explanations.",

    'measurement': "Task: Decide if Answer 2 and Answer 1 indicate the same choice, "
                   "even if the option letter is missing or the wording is slightly different. "
                   "Rules: "
                   "- Return 'True' if both answers select the same option letter (case insensitive), "
                    "or if the same option is selected even when the letter is missing."
                    "Otherwise, return 'False'. "
                   "Input format: "
                   "- Answer 1: '[ANSWER]'"
                   "- Answer 2: May include explanation and a final answer, 'FINAL ANSWER: [ANSWER]'"
                   "If 'FINAL ANSWER:' is not present, extract the final answer using phrases like 'The answer is'."
                   "Output Format: 'True', or 'False'. No explanations.",

    'measurement_projection': "Task: Decide if Answer 2 and Answer 1 indicate the same choice, even if the option letter is missing or the wording is slightly different. "
                              "Rules: "
                              "- Return 'True' if both answers select the same option letter (case insensitive), "
                              "or if the same option is selected even when the letter is missing."
                              "- Return 'False' if Answer 2 does not match any correct options from Answer 1. "
                              "- Return 'Partial' if Answer 2 contains at least one correct option from Answer 1, "
                              "but does not fully match. "
                              "Input Format: "
                              "- Answer 1: '[ANSWER]' (e.g., '(a) Right: [0.1, 0.2], (c) Left: [0.5, 0.6]'). "
                              "- Answer 2: May include explanation and a final answer (e.g., 'FINAL ANSWER: (a) Right: [0.1, 0.2], (c) Left: [0.5, 0.6]'), "
                              "If 'FINAL ANSWER:' is not present, extract the final answer using phrases like 'The answer is'."
                              "Output Format: 'True, 'Partial', or 'False'. No explanations.",

    # error margin 반
    'final': "Task: Decide if Answer 2 correctly selects the answer(s) indicated in Answer 1, even if the option letter is missing or the wording is slightly different."
             "Rules: "
             "- If Answer 1 contains only one option, Answer 2 must select that exact option (by matching the letter or the content) for the result to be 'True'."
             "- If Answer 1 contains multiple options, Answer 2 must select exactly one of them, not more, and it must match by letter or content. "
             "- Selecting multiple options when only one is allowed, or selecting an option not in Answer 1, results in 'False."
             "Also, return 'True' if the meaning of Answer 2 aligns with the intent of the question and Answer 1. "
             "Input format: "
             "- Question: A multiple-choice question with options."
             "- Answer 1: '[ANSWER]' (e.g., (a) Yes or (a) Yes, (b) No)"
             "- Answer 2: May include explanation and a final answer (e.g., FINAL ANSWER: (a) Yes). "
             "If 'FINAL ANSWER:' is not present, extract the final answer using phrases like 'The answer is'."
             "Output Format: 'True', or 'False'. No explanations.",

    'extract_value': "Task: Extract the final numerical answer explicitly stated as the correct answer."
                     "Rule: "
                     "If the response includes a line like VALUE: [Value], extract the number. "
                     "If VALUE: is not present, extract the numerical value "
                     "that is clearly stated as the correct or computed result. "
                     "If no such numerical value can be found, return 'N/A"
                     "Input Format: "
                     "- Response: A response in the format 'Value: [Value]' or simply '[Value]', which may also include additional content. "
                     "Output Format: Return [Value] or 'N/A'. No explanations.",

    'extract_value_projection': "Task: Extract the final numerical answer explicitly stated as the correct answer."
                                "Rule: If the Response contains 'VALUE: Right - [Right Value], Left - [Left Value]', "
                                "extract 'Right - [Right Value], Left - [Left Value]'; otherwise, return 'N/A'. "
                                "Input Format: "
                                "- Response: A response in the format 'VALUE: Right - [Right Value], Left - [Left Value]', "
                                "which may also include additional content. "
                                "Output Format: 'Right - [Right Value], Left - [Left Value]' or 'N/A'. "
                                "No explanations.",

}