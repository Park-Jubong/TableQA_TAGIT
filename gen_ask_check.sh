echo "1. generate prompt"
python generate_prompt.py \
--prompt_type pyagent \
--n_gram 1 \
--input_data table_sim_test \
--few_shot_num 1

echo "2. ask_llm"
python ask_llm.py \
--prompt_type pyagent  \
--num_samples 100 \
--output_file_name pyagent_test

echo "3. check answer"
python check_answer.py \
--prediction_file pyagent_test \
--answers_file pyagent_test

echo "----------------------------------------------------"
echo "1. generate prompt"
python generate_prompt.py \
--prompt_type code_sim \
--n_gram 1 \
--input_data pyagent_test \
--few_shot_num 1

echo "2. ask_llm"
python ask_llm.py \
--prompt_type code_sim  \
--num_samples 100 \
--output_file_name pyagent_code_test

echo "3. check answer"
python check_answer.py \
--prediction_file pyagent_code_test \
--answers_file pyagent_code_test

# echo "----------------------------------------------------"
# echo "----------------------------------------------------"

# echo "1. generate prompt"
# python generate_prompt.py \
# --prompt_type keyword_sim \
# --n_gram 1 \
# --input_data no_iter_prompting_nl_test \
# --few_shot_num 1

# echo "2. ask_llm"
# python ask_llm.py \
# --prompt_type keyword_sim  \
# --num_samples 100 \
# --output_file_name no_iter_prompting_key_test

# echo "3. check answer"
# python check_answer.py \
# --prediction_file no_iter_prompting_key_test \
# --answers_file no_iter_prompting_key_test

# echo "----------------------------------------------------"
# echo "1. generate prompt"
# python generate_prompt.py \
# --prompt_type code_sim \
# --n_gram 1 \
# --input_data no_iter_prompting_key_test \
# --few_shot_num 1

# echo "2. ask_llm"
# python ask_llm.py \
# --prompt_type code_sim  \
# --num_samples 100 \
# --output_file_name no_iter_prompting_key_code_test

# echo "3. check answer"
# python check_answer.py \
# --prediction_file no_iter_prompting_key_code_test \
# --answers_file no_iter_prompting_key_code_test

# echo "----------------------------------------------------"
# echo "----------------------------------------------------"

# echo "1. generate prompt"
# python generate_prompt.py \
# --prompt_type pyagent \
# --n_gram 1 \
# --input_data no_iter_prompting_nl_test \
# --few_shot_num 1

# echo "2. ask_llm"
# python ask_llm.py \
# --prompt_type pyagent  \
# --num_samples 100 \
# --output_file_name no_iter_prompting_pyagent_test

# echo "3. check answer"
# python check_answer.py \
# --prediction_file no_iter_prompting_pyagent_test \
# --answers_file no_iter_prompting_pyagent_test

# echo "----------------------------------------------------"
# echo "1. generate prompt"
# python generate_prompt.py \
# --prompt_type code_sim \
# --n_gram 1 \
# --input_data no_iter_prompting_pyagent_test \
# --few_shot_num 1

# echo "2. ask_llm"
# python ask_llm.py \
# --prompt_type code_sim  \
# --num_samples 100 \
# --output_file_name no_iter_prompting_pyagent_code_test

# echo "3. check answer"
# python check_answer.py \
# --prediction_file no_iter_prompting_pyagent_code_test \
# --answers_file no_iter_prompting_pyagent_code_test