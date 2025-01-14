import vllm

class vLLM:

    def __init__(self, model, 
        temperature=0.7, top_p=0.9, 
        dtype='half', gpu_memory_utilization=0.75, 
        num_gpus=1, 
    ):
        self.model = vllm.LLM(
            model, 
            dtype=dtype,
            enforce_eager=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization
        )
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            skip_special_tokens=False
        )

    def generate(self, x, max_tokens=1024, min_tokens=0, **kwargs):
        self.sampling_params.max_tokens = kwargs.pop('max_tokens', 256)
        self.sampling_params.min_tokens = kwargs.pop('min_tokens', 32)
        outputs = self.model.generate(x, self.sampling_params, use_tqdm=False)
        if len(outputs) > 1:
            return [o.outputs[0].text for o in outputs]
        else:
            return [outputs[0].outputs[0].text]

