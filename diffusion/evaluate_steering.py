import argparse

from core.eval.fid import compute_fid
from core.eval.clip import compute_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=['clip', 'fid'], required=True, help='Metric type to compute')
    parser.add_argument('--clip_path', type=str, help='Path to folder with images')
    parser.add_argument('--clip_fname', type=str, default=None, help='Name of the file to consider for clip score')
    parser.add_argument('--clip_concept', type=str, default=None, help='Concept name to compare with (for CLIP score)')
    parser.add_argument('--fid_first_path', type=str, default=None, help='Path to the first folder containing files (for FID score)')
    parser.add_argument('--fid_first_fname', type=str, default=None, help='Regex of the first file to match (for FID score)')
    parser.add_argument('--fid_second_path', type=str, default=None, help='Path to the second folder containing files (for FID score)')
    parser.add_argument('--fid_second_fname', type=str, default=None, help='Regex of the second file to match (for FID score)')
    args = parser.parse_args()

    if args.metric == 'fid':
        fid_value = compute_fid(
            first_path=args.fid_first_path,
            first_fname=args.fid_first_fname,
            second_path=args.fid_second_path,
            second_fname=args.fid_second_fname,
        )
        print(f'FID score: {fid_value}')
    elif args.metric == 'clip':
        clip_score, clip_accuracy = compute_clip(
            path=args.clip_path,
            fname=args.clip_fname,
            concept=args.clip_concept,
        )
        print(f'CLIP score: {clip_score}, CLIP accuracy: {clip_accuracy}')



if __name__ == "__main__":
    main()