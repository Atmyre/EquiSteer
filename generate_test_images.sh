#! /usr/bin/env bash
#SBATCH --partition=camera-long
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=./exp/logs/slurm-%x-%j.out
#SBATCH --error=./exp/logs/slurm-%x-%j.err
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_rt=120:00:00
#$ -l h_vmem=7.5G
#$ -l gpu=1
#$ -l cluster=apocrita

set -eoux pipefail

nvidia-smi

export JOB_NAME=$SLURM_JOB_NAME
export JOB_ID=$SLURM_JOB_ID

run_cmd="srun --gpus=1 -N1 --exclusive"

export PYTHONPATH=.
python=../miniconda3/bin/python

# gender, race, glasses, coco
attribute="gender"
# multiplier, discrete
debias_type="multiplier"

models=("sana15")
concepts=("CEO" "doctor" "pilot" "technician" "librarian" "teacher" "nurse" "fashion_designer")

if [ "$attribute" == "gender" ]; then
    for model in "${models[@]}"; do
        for concept in "${concepts[@]}"; do
            # # orig
            # $run_cmd $python run_with_steering.py --model_name $model \
            # --generate_concept $concept  --num_images_per_prompt 100  --steering_method casteer \
            # --steering_strength 1  --output_dir ./results_choose_profession/gender/${model}/${concept}_orig/ --renormalize_after_steering \
            # translate  \
            # --debias_type none 

            # concept
            $run_cmd $python run_with_steering.py --model_name $model \
            --generate_concept $concept  --num_images_per_prompt 1000 --steering_method casteer \
            --steering_strength 1  --output_dir ./test/${debias_type}/${attribute}/${model}/${concept}/ --renormalize_after_steering \
            translate  \
            --attribute gender --debias_type $debias_type 

            for gender in "female" "male"; do
                $run_cmd $python run_with_steering.py --model_name $model \
                --generate_concept "$gender $concept"  --num_images_per_prompt 300  --steering_method casteer \
                --steering_strength 1  --output_dir ./results/${debias_type}/${attribute}/${model}/${concept}_${gender}/ --renormalize_after_steering \
                translate  \
                --attribute $attribute --debias_type $debias_type 
            done
        done
    done
fi


if [ "$attribute" == "race" ]; then
    for model in "${models[@]}"; do
        for concept in "${concepts[@]}"; do
            # # orig
            # $run_cmd $python run_with_steering.py --model_name $model \
            # --generate_concept $concept  --num_images_per_prompt 1000  --steering_method casteer \
            # --steering_strength 1  --output_dir ./results/${debias_type}/${attribute}/${model}/${concept}_orig/ --renormalize_after_steering \
            # translate  \
            # --debias_type none

            # concept
            $run_cmd $python run_with_steering.py --model_name $model \
            --generate_concept $concept  --num_images_per_prompt 1000  --steering_method casteer \
            --steering_strength 1  --output_dir ./results/${debias_type}/${attribute}/${model}/${concept}_5cls/ --renormalize_after_steering \
            translate  \
            --attribute $attribute --debias_type $debias_type 

            # for race in "white" "black" "asian" "latino" "indian"; do
            #     $run_cmd $python run_with_steering.py --model_name $model \
            #         --generate_concept "$concept of $race race"  --num_images_per_prompt 50  --steering_method casteer \
            #         --steering_strength 1  --output_dir ./results/${debias_type}/${attribute}/${race}_${concept}/ --renormalize_after_steering \
            #         translate  \
            #         --attribute gender --debias_type $debias_type 
            # done
        done
    done
fi

if [ "$attribute" == "coco" ]; then
    for model in "${models[@]}"; do
        # # orig
        # $run_cmd $python run_with_steering.py --model_name $model \
        # --generate_concept $concept   --steering_method casteer \
        # --steering_strength 1  --output_dir ./results/${debias_type}/coco_gender/${model}_orig/ --renormalize_after_steering \
        # translate  \
        # --debias_type none 

        # concept
        $run_cmd $python run_with_steering.py --model_name $model \
        --generate_concept $concept   --steering_method casteer \
        --steering_strength 1  --output_dir ./results/${debias_type}/coco_gender/${model}/ --renormalize_after_steering \
        translate  \
        --attribute gender --debias_type $debias_type 
    done
fi


if [ "$attribute" == "glasses" ]; then
    for model in "${models[@]}"; do
        for concept in "${concepts[@]}"; do
            # # orig
            # $run_cmd $python run_with_steering.py --model_name $model \
            # --generate_concept $concept  --num_images_per_prompt 1000  --steering_method casteer \
            # --steering_strength 1  --output_dir ./results/${debias_type}/${attribute}${model}/${concept}_orig/ --renormalize_after_steering \
            # translate  \
            # --debias_type none 

            # concept
            $run_cmd $python run_with_steering.py --model_name $model \
            --generate_concept $concept  --num_images_per_prompt 1000  --steering_method casteer \
            --steering_strength 1  --output_dir ./results/${debias_type}/${attribute}/${model}/${concept}_debiased/ --renormalize_after_steering \
            translate  \
            --attribute glasses --debias_type discrete 
        done
    done
<<<<<<< Updated upstream
fi  
=======
fi  
>>>>>>> Stashed changes
