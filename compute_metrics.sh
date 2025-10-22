concepts=("CEO" "doctor" "pilot" "technician" "fashion designer" "librarian" "teacher" "nurse")
models=("sd21" "sdxl")
debias_methods=("continuous" "discrete")


for model in "${models[@]}"; do
    for concept in "${concepts[@]}"; do
        for debias_method in "${debias_methods[@]}"; do
            # concept
            python3 -u compute_metrics.py --images_path ../outputs_${model}_${concept}_${debias_method}/ \
                --attribute gender --results_path ./results/
        done
    done
done

