# generate code
python tidar_data/gen_lm_eval_sharegpt.py \
    --tasks humaneval,humaneval_plus \
    --output-path tidar_data/humaneval_all.json \
    --num-shots 0,0

python tidar_data/gen_lm_eval_sharegpt.py \
    --tasks mbpp,mbpp_plus \
    --output-path tidar_data/mbpp_all.json \
    --num-shots 3,3

# generate gsm8k
python tidar_data/gen_lm_eval_sharegpt.py \
    --tasks gsm8k_cot \
    --output-path tidar_data/gsm8k_cot.json \
    --num-shots 8

# generate minerva math
# python tidar_data/gen_lm_eval_sharegpt.py \
#     --tasks minerva_math_algebra,minerva_math_counting_and_prob,minerva_math_geometry,minerva_math_intermediate_algebra,minerva_math_num_theory,minerva_math_prealgebra,minerva_math_precalc \
#     --output-path tidar_data/minerva_math.json \
#     --num-shots 4,4,4,4,4,4,4
