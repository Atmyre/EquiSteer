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

models=("sana15")
attribute="gender"

if [ "$attribute" == "gender" ]; then
    for model in "${models[@]}"; do
        prefix=./vectors/"${model}"

        if [ "$model" == "sdxl" ]; then
            concepts=("councelor" "cleaner")
        else
            concepts=("councelor" "baker")
        fi

        for gender in "male_female" "female_male"; do
            
            for concept in "${concepts[@]}"; do
                if [ "$gender" == "male_female" ]; then
                    gender_to_generate="male ${concept}"
                else
                    gender_to_generate="female ${concept}"
                fi
                # concept
                $run_cmd $python run_with_steering.py --model_name $model \
                    --generate_concept "$concept"  --num_images_per_prompt 10  --steering_method casteer \
                    --steering_strength 1 --output_dir ./tmp_gender_sdxl30/${gender}/outputs_${model}_${concept}/ \
                    translate \
                    --source_concept_path ./steering_vectors/${model}_gender/${gender}/pos_means_126.pickle \
                    --target_concept_path ./steering_vectors/${model}_gender/${gender}/neg_means_126.pickle \
                    --save_vectors --save_vectors_path ${prefix}/${gender}/${concept}/

                # male concept
                $run_cmd $python run_with_steering.py --model_name $model \
                    --generate_concept "${gender_to_generate}"  --num_images_per_prompt 10  --steering_method casteer \
                    --steering_strength 1  --output_dir ./tmp_gender_sdxl30/${gender}/outputs_${model}_${gender}_${concept}/ \
                    translate \
                    --source_concept_path ./steering_vectors/${model}_gender/${gender}/pos_means_126.pickle \
                    --target_concept_path ./steering_vectors/${model}_gender/${gender}/neg_means_126.pickle \
                    --save_vectors --save_vectors_path ${prefix}/${gender}/pos_${concept}/

            done
            $run_cmd $python thresholds.py --prefix $prefix --model $model --concepts "${concepts[@]}" --attribute gender --subgroup $gender --statistics max --output_path ./thresholds_human/${model}/${gender}/
        done
    done
fi

if [ "$attribute" == "race" ]; then
    concepts=("man" "woman")
    races=("White" "Black" "Asian" "Latino" "Indian")

    for model in "${models[@]}"; do
        prefix=./vectors_race/"${model}"
        for race in "${races[@]}"; do
            for concept in "${concepts[@]}"; do
                # concept
                $run_cmd $python run_with_steering.py --model_name $model \
                    --generate_concept "$concept"  --num_images_per_prompt 10  --steering_method casteer \
                    --steering_strength 1 --output_dir ./tmp_race1/${race}/outputs_${model}_${concept}/ \
                    translate \
                    --source_concept_path ./steering_vectors/${model}_race/${race}/pos_means_210.pickle,\
                    --target_concept_path ./steering_vectors/${model}_race/${race}/neg_means_210.pickle,\
                    --save_vectors --save_vectors_path ${prefix}/${race}/${concept}/

                # race concept
                $run_cmd $python run_with_steering.py --model_name $model \
                    --generate_concept "a ${race} ${concept}"  --num_images_per_prompt 10  --steering_method casteer \
                    --steering_strength 1  --output_dir ./tmp_race1/${race}/outputs_${model}_pos_${concept}/ \
                    translate \
                    --source_concept_path ./steering_vectors/${model}_race/${race}/pos_means_210.pickle,\
                    --target_concept_path ./steering_vectors/${model}_race/${race}/neg_means_210.pickle,\
                    --save_vectors --save_vectors_path ${prefix}/${race}/pos_${concept}/
            done
            $run_cmd $python thresholds.py --prefix $prefix --model $model --concepts "${concepts[@]}" --attribute race --subgroup ${race} --statistics max --output_path ./thresholds_race/${model}/${race}/
        done
    done
fi

if [ "$attribute" == "glasses" ]; then
    concepts=("man" "woman")
    attrs=("eyeglasses")
    models=("sdxl")
    path="eyeglasses"

    for model in "${models[@]}"; do
        prefix=./vectors/"${model}"
        for attr in "${attrs[@]}"; do
            for concept in "${concepts[@]}"; do
                # concept
                $run_cmd $python run_with_steering.py --model_name $model \
                    --generate_concept "$concept"  --num_images_per_prompt 10  --steering_method casteer \
                    --steering_strength 1 --output_dir ./tmp_${path}/${attr}/outputs_${model}_${concept}/ \
                    translate \
                    --source_concept_path ./steering_vectors/${model}_${path}/${attr}/pos_means_210.pickle,\
                    --target_concept_path ./steering_vectors/${model}_${path}/${attr}/neg_means_210.pickle,\
                    --save_vectors --save_vectors_path ${prefix}/${attr}/${concept}/

                # race concept
                if [ "$attr" == "eyeglasses" ]; then
                    prompt_to_generate="${concept}_wearing_eyeglasses"
                else
                    prompt_to_generate=""
                fi
                $run_cmd $python run_with_steering.py --model_name $model \
                    --generate_concept ${prompt_to_generate}  --num_images_per_prompt 10  --steering_method casteer \
                    --steering_strength 1  --output_dir ./tmp_${path}/${attr}/outputs_${model}_pos_${concept}/ \
                    translate \
                    --source_concept_path ./steering_vectors/${model}_${path}/${attr}/pos_means_210.pickle,\
                    --target_concept_path ./steering_vectors/${model}_${path}/${attr}/neg_means_210.pickle,\
                    --save_vectors --save_vectors_path ${prefix}/${attr}/pos_${concept}/
            done
            $run_cmd $python thresholds.py --prefix $prefix --model $model --concepts "${concepts[@]}" --attribute concepts --subgroup ${attr} --statistics max --output_path ./thresholds_${path}/${model}/${attr}/
        done
    done
fi
