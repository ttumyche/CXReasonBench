import os
import subprocess


def run_evaluation():
    eval_dir = os.path.dirname(os.path.abspath(__file__))

    # -------------------------
    #  1️⃣ evaluate_reasoning.py
    # -------------------------
    reasoning_args = [
        "python",
        os.path.join(eval_dir, "evaluate_reasoning.py"),
        "--model_id", "",
        "--model_path", "",
        "--tensor_parallel_size", "2",
        "--model_id4scoring", "gemini-2.0-flash",
        "--cxreasonbench_base_dir", "",
        "--mimic_cxr_base", "",
        "--save_base_dir", "result"
    ]

    print("Running evaluate_reasoning.py ...")
    subprocess.run(reasoning_args, check=True)

    # -------------------------
    #  2️⃣ evaluate_guidance.py
    # -------------------------
    save_base_dir = reasoning_args[reasoning_args.index("--save_base_dir") + 1]
    model_id = reasoning_args[reasoning_args.index("--model_id") + 1]
    config_path = os.path.join(save_base_dir, 'inference/reasoning', model_id, "config.json")

    guidance_args = [
        "python",
        os.path.join(eval_dir, "evaluate_guidance.py"),
        "--config_path", config_path,
    ]
    print("Running evaluate_guidance.py ...")
    subprocess.run(guidance_args, check=True)

    # -------------------------
    #      3️⃣ metric.py
    # -------------------------
    model_id4scoring = reasoning_args[reasoning_args.index("--model_id4scoring") + 1]
    for evaluation_path in ['reasoning', 'guidance']:
        saved_dir_inference = os.path.join(save_base_dir, 'inference', evaluation_path, model_id)
        saved_dir_scoring = os.path.join(save_base_dir, 'scoring', evaluation_path, model_id4scoring, model_id)
        metric_args = [
            "python",
            os.path.join(eval_dir, "metric.py"),
            "--saved_dir_inference", saved_dir_inference,
            "--saved_dir_scoring", saved_dir_scoring,
        ]
        print("Running metric.py ...")
        subprocess.run(metric_args, check=True)

if __name__ == "__main__":
    run_evaluation()