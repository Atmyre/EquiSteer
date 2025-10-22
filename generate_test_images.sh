# the concepts here will be the ones to generate
concepts=("CEO" "doctor" "pilot" "technician" "fashion designer" "librarian" "teacher" "nurse")
models=("sdxl")
debias_methods=("continuous" "discrete")

for model in "${models[@]}"; do
    for concept in "${concepts[@]}"; do
        for debias_method in "${debias_methods[@]}"; do
            # concept
            CUDA_VISIBLE_DEVICES=1 python -m diffusion.run_with_steering --model_name $model \
            --generate_concept $concept  --num_images_per_prompt 100  --steering_method casteer \
            --steering_strength 1 --output_dir ../outputs_${model}_${concept}_${debias_method}/ translate  \
            --source_concept_path ../outputs_${model}/pos_means_60.pickle \
            --target_concept_path ../outputs_${model}/neg_means_60.pickle \
            --attribute gender --debias_type $debias_method
        done
    done
done
