import dataclasses
from sglang import function, gen, Runtime
from sglang.srt.server_args import prepare_server_args, ServerArgs


@function
def text_gen(s, prompt):
    s += prompt + gen('response', max_tokens=128)

if __name__ == "__main__":
    server_args = prepare_server_args(["--config", "tidar_8b_config.yaml"])
    # Only pass dataclass fields (exclude computed attrs like model_config)
    _field_names = {f.name for f in dataclasses.fields(ServerArgs)}
    _kwargs = {k: getattr(server_args, k) for k in _field_names}
    runtime = Runtime(**_kwargs)

    while True:
        print('='*100)
        prompt = input("Enter a prompt: ")
        response = text_gen.run(prompt=prompt, runtime=runtime)
        print(response['response'])
        print('='*100)
