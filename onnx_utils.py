import onnxruntime_genai as og
from dataclasses import dataclass, asdict
from time import perf_counter

@dataclass
class SearchOptions:
    min_length: int = None # Min number of tokens to generate including the prompt
    max_length: int = 2048 # Max number of tokens to generate including the prompt
    do_sample: bool = False # Do random sampling. When false, greedy or beam search are used to generate the output.
    top_p: float  = None # Top p probability to sample with
    top_k: int = None # Top k tokens to sample from
    temperature: float = None # Temperature to sample with
    repetition_penalty: float = None # Repetition penalty to sample with

@dataclass
class GeneratedText:
    text: str
    stats: dict

class Model:
    def __init__(self, model_path: str, search_options: SearchOptions = None):
        self._search_options = {k:v for k,v in asdict(SearchOptions() if search_options is None else search_options).items() if v is not None}
        self._model = og.Model(model_path)
        self._tokenizer = og.Tokenizer(self._model)
        self._tokenizer_stream = self._tokenizer.create_stream()

    def generate(self, input: str) -> GeneratedText:
        time_start = perf_counter()

        input_tokens = self._tokenizer.encode(f'<|user|>\n{input} <|end|>\n<|assistant|>')

        time_tokenized = perf_counter()

        params = og.GeneratorParams(self._model)
        params.set_search_options(**self._search_options)
        params.input_ids = input_tokens
        generator = og.Generator(self._model, params)

        output = ''
        output_token_count = 0
        time_first_token = None
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            output_token = generator.get_next_tokens()[0]
            output += self._tokenizer_stream.decode(output_token)
            if time_first_token is None:
                time_first_token = perf_counter()
            output_token_count += 1

        time_generated = perf_counter()

        return GeneratedText(
            output,
            {
                'input_token_count': len(input_tokens),
                'output_token_count': output_token_count,
                'tokenization_time': time_tokenized - time_start,
                'time_to_first_token': time_first_token - time_start,
                'generation_time': time_generated - time_tokenized,
                'total_time': time_generated - time_start,
                'average_time_per_token': (time_generated - time_start) / (len(input_tokens) + output_token_count),
                'input_tokens_per_second': len(input_tokens) / (time_first_token - time_start),
                'output_tokens_per_second': output_token_count / (time_generated - time_first_token)
            })
