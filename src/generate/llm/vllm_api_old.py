import argparse
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams
# cleaning
import contextlib
import gc
import torch
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel
)

def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()

async def iterate_over_output_for_one_prompt(output_iterator: AsyncStream) -> str:
    last_text = ""
    prompt = "???"

    async for output in output_iterator:
        prompt = output.prompt
        last_text = output.outputs[0].text

    return last_text

async def generate(
    engine: AsyncLLMEngine, 
    request_ids: list[str], 
    prompts: list[str], 
    sampling_params: SamplingParams, 
    **kwargs
) -> list[str]:

    output_iterators = [
        await engine.add_request(request_ids[i], prompt, sampling_params)\
                for i, prompt in enumerate(prompts)
    ]
    outputs = await asyncio.gather(*[iterate_over_output_for_one_prompt(output_iterator)
                                     for output_iterator in output_iterators])
    return list(outputs)

async def serve(engine, sampling_params, prompts):
    request_ids = [str(i) for i in range(len(prompts))]
    outputs = await generate(engine, request_ids, prompts, sampling_params)
    return outputs

class LLM:

    def __init__(self, 
        model, 
        temperature=0.7, top_p=0.9, 
        dtype='half', gpu_memory_utilization=0.75, 
        num_gpus=1, 
        enforce_eager=False,
        think_activated=False,
    ):
        args = AsyncEngineArgs(
            model=model,
            dtype=dtype,
            enforce_eager=enforce_eager,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=20480
        )
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs.from_cli_args(args))

        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            skip_special_tokens=False
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def generate(self, prompts, **kwargs):
        engine = self.engine
        sampling_params = self.sampling_params
        sampling_params.max_tokens = kwargs.pop('max_tokens', 5)
        sampling_params.min_tokens = kwargs.pop('min_tokens', 0)

        outputs = asyncio.run(
            serve(engine, sampling_params=sampling_params, prompts=prompts)
        )
        return outputs

    def __del__(self):
        cleanup()
        print('Destructor called. Clean up vllm server.')

    # def generate(self, prompts, **kwargs):
    #     engine = self.engine
    #     sampling_params = self.sampling_params
    #     sampling_params.max_tokens = kwargs.pop('max_tokens', 5)
    #     sampling_params.min_tokens = kwargs.pop('min_tokens', 0)
    #     
    #     # Run the async generation in the persistent event loop
    #     # return self.loop.run_until_complete(self._generate_async(prompts, sampling_params))
    #     outputs = self.loop.run_until_complete(
    #         asyncio.run(
    #             serve(engine, sampling_params=sampling_params, prompts=prompts)
    #         )
    #     )
    #     return outputs
