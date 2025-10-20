# the concepts here will be the ones to generate
concepts=("CEO" "doctor" "pilot" "technician" "fashion designer" "librarian" "teacher" "nurse")
models=("sdxl")

for model in "${models[@]}"; do
    for concept in "${concepts[@]}"; do
        # concept
        CUDA_VISIBLE_DEVICES=1 python -m diffusion.run_with_steering --model_name $model \
        --generate_concept $concept  --num_images_per_prompt 20  --steering_method casteer \
        --steering_strength 1  --intermediate_clipping --output_dir ../outputs_${model}_${concept}/ translate  \
        --source_concept_path ../outputs_sdxl/pos_means_30.pickle \
        --target_concept_path ../outputs_sdxl/neg_means_30.pickle \
        --attribute gender

    done

done
