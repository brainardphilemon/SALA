# configs/mappings.py

generalization_mapping = {
    "G1":  {"train": ["tqa"], "test": ["sciq"]},
    "G2":  {"train": ["tqa"], "test": ["nq_open"]},
    "G3":  {"train": ["tqa"], "test": ["triviaqa"]},
    "G4":  {"train": ["sciq"], "test": ["tqa"]},
    "G5":  {"train": ["sciq"], "test": ["nq_open"]},
    "G6":  {"train": ["sciq"], "test": ["triviaqa"]},
    "G7":  {"train": ["triviaqa"], "test": ["sciq"]},
    "G8":  {"train": ["triviaqa"], "test": ["tqa"]},
    "G9":  {"train": ["triviaqa"], "test": ["nq_open"]},
    "G10": {"train": ["nq_open"], "test": ["triviaqa"]},
    "G11": {"train": ["nq_open"], "test": ["sciq"]},
    "G12": {"train": ["nq_open"], "test": ["tqa"]},
    "G13": {"train": ["sciq", "nq_open", "triviaqa"], "test": ["tqa"]},
    "G14": {"train": ["tqa", "nq_open", "sciq"], "test": ["triviaqa"]},
    "G15": {"train": ["tqa", "sciq", "triviaqa"], "test": ["nq_open"]},
    "G16": {"train": ["tqa", "nq_open", "triviaqa"], "test": ["sciq"]},
}

generalization_mapping_math = {
    "G1":  {"train": ["SVAMP"], "test": ["math"]},
    "G2":  {"train": ["SVAMP"], "test": ["mgsm"]},
    "G3":  {"train": ["SVAMP"], "test": ["theoremqa"]},
    "G4":  {"train": ["math"], "test": ["SVAMP"]},
    "G5":  {"train": ["math"], "test": ["mgsm"]},
    "G6":  {"train": ["math"], "test": ["theoremqa"]},
    "G7":  {"train": ["mgsm"], "test": ["math"]},
    "G8":  {"train": ["mgsm"], "test": ["theoremqa"]},
    "G9":  {"train": ["mgsm"], "test": ["SVAMP"]},
    "G10": {"train": ["theoremqa"], "test": ["SVAMP"]},
    "G11": {"train": ["theoremqa"], "test": ["math"]},
    "G12": {"train": ["theoremqa"], "test": ["mgsm"]},
    "G13m": {"train": ["SVAMP", "math", "mgsm"], "test": ["theoremqa"]},
    "G14m": {"train": ["SVAMP", "math", "theoremqa"], "test": ["mgsm"]},
    "G15m": {"train": ["SVAMP", "theoremqa", "mgsm"], "test": ["math"]},
    "G16m": {"train": ["theoremqa", "math", "mgsm"], "test": ["SVAMP"]},
}

HF_NAMES = {
    "Llama3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    "qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "qwen2.5-14B": "Qwen/Qwen2.5-14B",
    "Llama3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Llama3.2-3B-Instruct": "meta-llama/Meta-Llama-3.2-3B-Instruct"
}