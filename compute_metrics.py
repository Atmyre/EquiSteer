# script to compute fairness metrics
import argparse
import os
from deepface import DeepFace

def absolute_file_paths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

gender_map = {'female': 0, 'male': 1}
race_map = {'black': 0, 'white': 1}

def main(args):
    images_path = args.images_path
    attribute = args.attribute

    assert attribute in ["gender", "race"], "Only gender and race supported for now!"

    assert os.path.exists(images_path), "The path is invalid!"
    image_paths = absolute_file_paths(images_path)

    predictions = []

    for image_path in image_paths:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=attribute,
            detector_backend='retinaface',
            enforce_detection=False,
        )

        print(result)

        if attribute == 'gender':
            output = result[0]['dominant_gender']
            if output.lower() == 'man':
                predictions += [gender_map['male']]
            elif output.lower() == 'woman':
                predictions += [gender_map['female']]
            else:
                raise ValueError(f"Prediction neither male or female!")
        else:
            raise ValueError("Only gender metrics can be computed for the moment")


if __name__=="__main__":
    parser = argparse.ArgumentParser(usage="script to compute fairness metrics on the generated images")
    parser.add_argument("--images_path", type=str, help="path of the directory where generated images are located")
    parser.add_argument("--attribute", type=str, help="attribute to choose the classifier -- gender/race")
    parser.add_argument("--results_path", type=str, help="path to store the fairness metrics")
    args = parser.parse_args()
    main(args)