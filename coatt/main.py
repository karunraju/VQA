import argparse
from coatt.simple_baseline_experiment_runner import SimpleBaselineExperimentRunner
from coatt.coattention_experiment_runner import CoattentionNetExperimentRunner


if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, choices=['simple', 'coattention'], default='simple')
    parser.add_argument('--train_image_dir', type=str, default='/home/ubuntu/hw3_release/data/train2014')
    parser.add_argument('--train_question_path', type=str, default='/home/ubuntu/hw3_release/data/OpenEnded_mscoco_train2014_questions.json')
    parser.add_argument('--train_annotation_path', type=str, default='/home/ubuntu/hw3_release/data/mscoco_train2014_annotations.json')
    parser.add_argument('--test_image_dir', type=str, default='/home/ubuntu/hw3_release/data/val2014')
    parser.add_argument('--test_question_path', type=str, default='/home/ubuntu/hw3_release/data/OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--test_annotation_path', type=str, default='/home/ubuntu/hw3_release/data/mscoco_val2014_annotations.json')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_data_loader_workers', type=int, default=10)
    args = parser.parse_args()

    if args.model == "simple":
        experiment_runner_class = SimpleBaselineExperimentRunner
    elif args.model == "coattention":
        experiment_runner_class = CoattentionNetExperimentRunner
    else:
        raise ModuleNotFoundError()

    experiment_runner = experiment_runner_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers)
    experiment_runner.train()
