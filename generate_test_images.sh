# the concepts here will be the ones to generate
concepts=("CEO")
models=("sdxl")
debias_methods=("continuous")

for model in "${models[@]}"; do
    for concept in "${concepts[@]}"; do
        for debias_method in "${debias_methods[@]}"; do
            # concept
            CUDA_VISIBLE_DEVICES=1 python -m diffusion.run_with_steering --model_name $model \
            --generate_concept "${concept}"  --num_images_per_prompt 100  --steering_method casteer \
            --steering_strength 1 --output_dir "../outputs_${model}_${concept}_${debias_method}/" translate  \
            --source_concept_path ../outputs_${model}_white/pos_means_60.pickle  ../outputs_${model}_black/pos_means_60.pickle  ../outputs_${model}_latino/pos_means_60.pickle ../outputs_${model}_asian/pos_means_60.pickle ../outputs_${model}_indian/pos_means_60.pickle \
            --target_concept_path ../outputs_${model}_white/neg_means_60.pickle  ../outputs_${model}_black/neg_means_60.pickle ../outputs_${model}_latino/neg_means_60.pickle ../outputs_${model}_asian/neg_means_60.pickle ../outputs_${model}_indian/neg_means_60.pickle \
            --attribute race --debias_type $debias_method
        done
    done
done
