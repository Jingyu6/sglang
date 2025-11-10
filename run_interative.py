import dataclasses
from sglang import function, gen, Runtime
from sglang.srt.server_args import prepare_server_args, ServerArgs


MAX_TOKENS = 512

@function
def text_gen(s, prompt):
    s += prompt + gen(
        'response', 
        max_tokens=MAX_TOKENS, 
        ignore_eos=True, 
        stop=[],
        stop_token_ids=[],
        n=1
    )

if __name__ == "__main__":
    model_name = input("Enter the model name: ")
    assert model_name in ["tidar", "qwen3"], "Invalid model name"
    server_args = prepare_server_args(["--config", f"{model_name}_8b_config.yaml"])
    # Only pass dataclass fields (exclude computed attrs like model_config)
    _field_names = {f.name for f in dataclasses.fields(ServerArgs)}
    _kwargs = {k: getattr(server_args, k) for k in _field_names}
    runtime = Runtime(**_kwargs)

    while True:
        print('='*100)
        prompt = input("Enter a prompt: ")
        if prompt == "input": 
            print("Reading input from input.txt")
            with open("tmp.txt", "r") as f:
                prompt = f.read()
        response = text_gen.run(prompt=prompt, backend=runtime)
        print(response['response'])
        print('='*100)
