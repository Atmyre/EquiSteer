import pickle
import os
import argparse
import numpy as np
import copy

def main(args):
    prefix = args.prefix
    model = args.model
    concepts = list(args.concepts)
    output_path = args.output_path

    tracker = {}
    keys = ["diffusion_step", "place_in_unet", "block_index"]

    for concept in concepts:
        f1 = os.path.join(prefix, f'{concept}_{model}') # concept
        f2 = os.path.join(prefix, f'male_{concept}_{model}') # male concept
        f1_files = os.listdir(f1)
        f2_files = os.listdir(f2)

        rows = []
        for f1_file, f2_file in zip(f1_files, f2_files):
            assert f1_file == f2_file

            with open(os.path.join(f1, f1_file), 'rb') as h:
                obj1 = pickle.load(h)
            with open(os.path.join(f2, f2_file), 'rb') as h:
                obj2 = pickle.load(h)

            if (obj1[keys[0]], obj1[keys[1]], obj1[keys[2]]) not in tracker.keys():
                tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])] = {}
                if 'mean' not in tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])].keys():
                    tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])]['mean'] = []
                    tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])]['std'] = []

            tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])]['mean'] += [(obj1['scores'] - obj2['scores']).mean()]
            tracker[(obj1[keys[0]], obj1[keys[1]], obj1[keys[2]])]['std'] += [(obj1['scores'] - obj2['scores']).std()]
    
    # now aggregate for all concepts
    final_thresholds = copy.deepcopy(tracker)
    for k,v in tracker.items():
        final_thresholds[k]['mean'] = np.array(v['mean']).mean()
        final_thresholds[k]['std'] = np.array(v['std']).mean()

    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, f'{args.attribute}.pkl'), 'wb') as handle:
        pickle.dump(final_thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--attribute", type=str)
    parser.add_argument("--concepts", nargs="+", help="List of concepts")
    args = parser.parse_args()
    main(args)