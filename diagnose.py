"""End-to-end 5G root-cause diagnosis.

Raw drive-test telemetry text in -> (computed measurements, model diagnosis) out.

Usage:
    from diagnose import Diagnostician
    d = Diagnostician(adapter="sub/submission_main_public_0.996/qwen3_rca_lora")
    result = d.diagnose(open("scenario.txt").read())
    print(result["measurements"])   # what the system computed
    print(result["diagnosis"])      # the model's stated root cause
"""
import re
import sys
from collections import Counter

sys.path.insert(0, "solution")
from feature_text import compact_question  # parser -> features -> verbalizer

SYS = ("You are a 5G network diagnosis expert. Analyze the data and answer with "
       "the most likely option, ending with \\boxed{<option>}.")


class Diagnostician:
    def __init__(self, adapter=None, model_name="Qwen/Qwen3-4B", device_map="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device_map)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.eval()
        model.config.use_cache = True
        self.model = model

    def _generate(self, prompt, n=5, max_new=160, seed=0):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = self.tok(text, return_tensors="pt",
                       truncation=True, max_length=2048).to(self.model.device)
        plen = inp["input_ids"].shape[1]
        outs = []
        self.torch.manual_seed(seed)
        with self.torch.no_grad():
            o = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=self.tok.eos_token_id)
            outs.append(self.tok.decode(o[0][plen:], skip_special_tokens=True))
            o = self.model.generate(**inp, max_new_tokens=max_new, do_sample=True,
                                    temperature=0.7, top_p=0.9,
                                    num_return_sequences=n - 1,
                                    pad_token_id=self.tok.eos_token_id)
            outs.extend(self.tok.decode(r[plen:], skip_special_tokens=True) for r in o)
        return outs

    def diagnose(self, scenario_text, votes=5):
        """scenario_text: a fault scenario as provided in the benchmark format
        (question + options + telemetry tables)."""
        prompt = compact_question(scenario_text)
        measurements = prompt.split("Computed diagnostic summary")[-1] \
            if "Computed diagnostic summary" in prompt else "(no telemetry tables found)"
        responses = self._generate(prompt, n=votes)
        answers = []
        for r in responses:
            m = re.findall(r"\\boxed\{([A-Za-z]{0,2}\d{0,2})\}", r)
            if m:
                answers.append(m[-1])
        verdict = Counter(answers).most_common(1)[0][0] if answers else None
        best_response = next((r for r in responses
                              if verdict and f"\\boxed{{{verdict}}}" in r), responses[0])
        return {
            "measurements": "Computed diagnostic summary" + measurements,
            "diagnosis": verdict,
            "explanation": best_response.strip(),
            "votes": dict(Counter(answers)),
        }
