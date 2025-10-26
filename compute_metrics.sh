concepts=("CEO") # "doctor" "pilot" "technician" "fashion designer" "librarian" "teacher" "nurse")
models=("sdxl")
debias_methods=("discrete" "continuous")


for model in "${models[@]}"; do
    for concept in "${concepts[@]}"; do
        for debias_method in "${debias_methods[@]}"; do
            # concept
            python3 -u compute_metrics.py --images_path "../outputs_${model}_${concept}_${debias_method}/"  \
                --attribute race --results_path ./results/ --approach ${debias_method} --concept "${concept}"
        done
    done
done

