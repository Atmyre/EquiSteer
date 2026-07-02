import pickle
import os
import argparse
import numpy as np
import copy
from tqdm import tqdm 

def main(args):
    prefix = args.prefix
    model = args.model
    concepts = list(args.concepts)
    output_path = args.output_path

    tracker = {}
    keys = ["diffusion_step", "place_in_unet", "block_index"]

    for concept in concepts:
        f1 = os.path.join(prefix, f'{args.subgroup}/{concept}') # concept
        f2 = os.path.join(prefix, f'{args.subgroup}/pos_{concept}') # male/female concept
        f1_files = sorted(os.listdir(f1))
        f2_files = sorted(os.listdir(f2))

        for f1_file, f2_file in tqdm(zip(f1_files, f2_files)):
            if not f1_file == f2_file:
                print(f1_file, f2_file)
            assert f1_file == f2_file

            with open(os.path.join(f1, f1_file), 'rb') as h:
                obj1 = pickle.load(h)
            with open(os.path.join(f2, f2_file), 'rb') as h:
                obj2 = pickle.load(h)

            if (obj1[keys[0]], obj1[keys[1]], obj1[keys[2]]) not in tracker.keys():
                tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])] = {}
                if 'stats_max' not in tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])].keys():
                    tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])]['stats_max'] = []
                    tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])]['mean_stats'] = []

            tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])]['mean_stats'] += [[obj1['scores'].mean(), obj2['scores'].mean()]]
            if args.statistics == "min":
                tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])]['stats_max'] += [[obj1['scores'].min(), obj2['scores'].min()]]
            elif args.statistics == "max":
                tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])]['stats_max'] += [[obj1['scores'].max(), obj2['scores'].max()]]
    
    # now aggregate for all concepts
    final_thresholds = copy.deepcopy(tracker)
    for k,v in tracker.items():
        final_concept_stats = np.array(v['stats_max'])[:, 0]
        final_male_stats = np.array(v['stats_max'])[:, 1]
        mean_stat = (final_male_stats.mean() + final_concept_stats.mean()) / 2
        final_thresholds[k]['stats'] = mean_stat

        final_thresholds[k]['mean_normal_max'] = np.mean(final_concept_stats) 
        final_thresholds[k]['std_normal_max'] = np.std(final_concept_stats) 
        final_thresholds[k]['mean_male_max'] = np.mean(final_male_stats) 
        final_thresholds[k]['std_male_max'] = np.std(final_male_stats) 

        final_concept_mean_stats = np.array(v['mean_stats'])[:, 0]
        final_male_mean_stats = np.array(v['mean_stats'])[:, 1]

        final_thresholds[k]['mean_normal'] = np.mean(final_concept_mean_stats) 
        final_thresholds[k]['std_normal'] = np.std(final_concept_mean_stats) 
        final_thresholds[k]['mean_male'] = np.mean(final_male_mean_stats) 
        final_thresholds[k]['std_male'] = np.std(final_male_mean_stats) 
        
        if k[0] == 0:
            print(k)
            print(np.min(final_male_stats), np.mean(final_male_stats), np.std(final_male_stats), np.max(final_male_stats))
            print(np.min(final_concept_stats), np.mean(final_concept_stats), np.std(final_concept_stats), np.max(final_concept_stats))
            print(np.min(final_male_mean_stats), np.mean(final_male_mean_stats), np.std(final_male_mean_stats), np.max(final_male_mean_stats))
            print(np.min(final_concept_mean_stats), np.mean(final_concept_mean_stats), np.std(final_concept_mean_stats), np.max(final_concept_mean_stats))
            print(mean_stat)

    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, f'{args.attribute}_{args.subgroup}_{model}.pkl'), 'wb') as handle:
        pickle.dump(final_thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--attribute", type=str)
    parser.add_argument("--statistics", type=str, default="max")
    parser.add_argument("--concepts", nargs="+", help="List of concepts")
    parser.add_argument("--subgroup", type=str)
    args = parser.parse_args()
    main(args)