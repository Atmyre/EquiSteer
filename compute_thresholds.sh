concepts=("driver" "photographer" "builder" "shopkeeper" "sprinter" "singer")
models=("sdxl")
prefix=/data/akshit/fairsteer

for model in "${models[@]}"; do
    for concept in "${concepts[@]}"; do
        # concept
        CUDA_VISIBLE_DEVICES=1 python -m diffusion.run_with_steering --model_name $model \
            --generate_concept "$concept"  --num_images_per_prompt 50  --steering_method casteer \
            --steering_strength 1  --intermediate_clipping --output_dir ../outputs_${model}_${concept}/ translate \
            --source_concept_path ../outputs_${model}/pos_means_30.pickle \
            --target_concept_path ../outputs_${model}/neg_means_30.pickle \
            --save_vectors --save_vectors_path ${prefix}/${concept}_${model}/

        # male concept
        CUDA_VISIBLE_DEVICES=1 python -m diffusion.run_with_steering --model_name $model \
        --generate_concept "male ${concept}"  --num_images_per_prompt 50  --steering_method casteer \
        --steering_strength 1  --intermediate_clipping --output_dir ../outputs_${model}_male_${concept}/ translate \
        --source_concept_path ../outputs_${model}/pos_means_30.pickle \
        --target_concept_path ../outputs_${model}/neg_means_30.pickle \
        --save_vectors --save_vectors_path ${prefix}/male_${concept}_${model}/
    done
    python thresholds.py --prefix $prefix --model $model --concepts "${concepts[@]}" --attribute gender --output_path ./thresholds

done
