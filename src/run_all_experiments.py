import os

# =========================================================
# FULL EXPERIMENT SPACE (for paper)
# =========================================================

datasets = ["cancer", "parkinson"]

models = ["llama3.2:3b", "qwen3.5:9b", "qwen2.5:14b"]

prompts = ["generic_ml", "biomedical_general", "cancer_clinical", "strict_json"]

optimizers = ["pso", "ga", "aco", "woa", "bhho"]

inits = ["random", "llm", "filter"]

filters = ["mutual_info", "anova", "chi2"]


# =========================================================
# RUN ALL EXPERIMENTS
# =========================================================
for dataset in datasets:
    for opt in optimizers:
        for init in inits:
            for model in models:
                for prompt in prompts:

                    # filter only used when init=filter
                    filter_arg = "mutual_info"

                    if init == "filter":
                        for f in filters:

                            print("\n==============================")
                            print(f"{dataset} | {opt} | {init} | {f} | {model} | {prompt}")
                            print("==============================")

                            os.system(f"""
python -m src.main \
--dataset {dataset} \
--optimizer {opt} \
--init {init} \
--filter {f} \
--model {model} \
--prompt {prompt} \
--iters 30 \
--pop 20
""")

                    else:

                        print("\n==============================")
                        print(f"{dataset} | {opt} | {init} | {model} | {prompt}")
                        print("==============================")

                        os.system(f"""
python -m src.main \
--dataset {dataset} \
--optimizer {opt} \
--init {init} \
--model {model} \
--prompt {prompt} \
--iters 30 \
--pop 20
""")