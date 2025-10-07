## Cross-Attention-Based Steering for De-biasing in Diffusion Models
This project explores cross-attention-based steering techniques to mitigate bias in diffusion models. By analyzing and modifying cross-attention representations, we compute steering vectors that can guide generative behavior toward more balanced and unbiased outcomes.

### Installation

  - **Create a Conda environment**
   ```bash
   conda create -n steer python=3.12
   conda activate steer
   ```
  - **Install dependencies**
  ```
  pip install -r requirements.txt
  ```

### Usage
  - **Compute Steering Vectors**
    Run the following command to compute steering vectors from reference positive and negative prompts:
    ```
    CUDA_VISIBLE_DEVICES=1 python -m diffusion.compute_steering_vectors --model sd21 --mode file --prompts_pos_file exp/prompts/professions_male.txt --prompts_neg_file exp/prompts/professions_female.txt  --output_dir ../outputssd21/
    ```
  - **Apply Steering Vectors**
    Once computed, apply the steering vectors during diffusion model generation:
    ```
    python -m diffusion.run_with_steering --model_name sd21 --generate_concept teacher  --steering_method casteer --steering_strength 2  --intermediate_clipping --output_dir ../outputs/ translate --source_concept_path ../outputssd21/pos_means_30.pickle --target_concept_path ../outputssd21/neg_means_30.pickle
    ```    
