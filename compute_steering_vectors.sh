models=("sd21-turbo" "sdxl-turbo")
races=("white" "black" "latino" "asian" "indian")

for model in "${models[@]}"; do
    if [ "$model" = "sd21-turbo" ] || [ "$model" = "sdxl-turbo" ]; then
        CUDA_VISIBLE_DEVICES=1 python -m diffusion.compute_steering_vectors --model $model \
        --mode file --prompts_pos_file exp/prompts/steer_male.txt \
        --prompts_neg_file exp/prompts/steer_female.txt  --output_dir ../outputs_${model%%-*}_male_female/
        
        for race1 in "${races[@]}"; do
            # for race2 in "${races[@]}"; do
                if [ "$race1" != "$race2" ]; then
                    CUDA_VISIBLE_DEVICES=1 python -m diffusion.compute_steering_vectors --model $model \
                    --mode file --prompts_pos_file exp/prompts/steer_$race1.txt \
                    --prompts_neg_file exp/prompts/steer_without_race.txt  --output_dir "../outputs_${model%%-*}_${race1}/"
                fi
            # done
        done
    else
        CUDA_VISIBLE_DEVICES=1 python -m diffusion.compute_steering_vectors --model $model \
        --mode file --prompts_pos_file exp/prompts/steer_male.txt \
        --prompts_neg_file exp/prompts/steer_female.txt  --output_dir ../outputs_${model}_male_female/

        for race1 in "${races[@]}"; do
            # for race2 in "${races[@]}"; do
                if [ "$race1" != "$race2" ]; then
                    CUDA_VISIBLE_DEVICES=1 python -m diffusion.compute_steering_vectors --model $model \
                    --mode file --prompts_pos_file exp/prompts/steer_$race1.txt \
                    --prompts_neg_file exp/prompts/steer_without_race.txt  --output_dir "../outputs_${model}_${race1}/"
                fi
            # done
        done
    fi
done