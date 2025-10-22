
models=("sd21" "sdxl")
prefix=/data/akshit/fairsteer


male_concepts=("driver" "photographer" "builder")
for model in "${models[@]}"; do
    for concept in "${male_concepts[@]}"; do
        # concept
        CUDA_VISIBLE_DEVICES=1 python -m diffusion.run_with_steering --model_name $model \
            --generate_concept "$concept"  --num_images_per_prompt 5  --steering_method casteer \
            --steering_strength 1 --output_dir ../outputs_${model}_${concept}/ translate \
            --source_concept_path ../outputs_${model}/pos_means_60.pickle \
            --target_concept_path ../outputs_${model}/neg_means_60.pickle \
            --save_vectors --save_vectors_path ${prefix}/${concept}_${model}/

        # male concept
        CUDA_VISIBLE_DEVICES=1 python -m diffusion.run_with_steering --model_name $model \
        --generate_concept "male ${concept}"  --num_images_per_prompt 5  --steering_method casteer \
        --steering_strength 1 --output_dir ../outputs_${model}_male_${concept}/ translate \
        --source_concept_path ../outputs_${model}/pos_means_60.pickle \
        --target_concept_path ../outputs_${model}/neg_means_60.pickle \
        --save_vectors --save_vectors_path ${prefix}/male_${concept}_${model}/
    done
    python thresholds.py --prefix $prefix --model $model --concepts "${male_concepts[@]}" \
        --attribute gender --output_path ./thresholds_male/ --statistics max --subgroup male

done

female_concepts=("nutritionist" "receptionist")
for model in "${models[@]}"; do
    for concept in "${female_concepts[@]}"; do
        # concept
        CUDA_VISIBLE_DEVICES=1 python -m diffusion.run_with_steering --model_name $model \
            --generate_concept "$concept"  --num_images_per_prompt 5  --steering_method casteer \
            --steering_strength 1 --output_dir ../outputs_${model}_${concept}/ translate \
            --source_concept_path ../outputs_${model}/pos_means_60.pickle \
            --target_concept_path ../outputs_${model}/neg_means_60.pickle \
            --save_vectors --save_vectors_path ${prefix}/${concept}_${model}/

        # female concept
        CUDA_VISIBLE_DEVICES=1 python -m diffusion.run_with_steering --model_name $model \
        --generate_concept "female ${concept}"  --num_images_per_prompt 5  --steering_method casteer \
        --steering_strength 1 --output_dir ../outputs_${model}_female_${concept}/ translate \
        --source_concept_path ../outputs_${model}/pos_means_60.pickle \
        --target_concept_path ../outputs_${model}/neg_means_60.pickle \
        --save_vectors --save_vectors_path ${prefix}/female_${concept}_${model}/
    done
    python thresholds.py --prefix $prefix --model $model --concepts "${female_concepts[@]}" --attribute gender \
        --output_path ./thresholds_female/ --statistics min --subgroup female

done
