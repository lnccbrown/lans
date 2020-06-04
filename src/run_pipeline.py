import config, os
from train_detector import train_model, test_model_eval
import argparse
from reverse_model import train_reverse_model, test_rev_model_eval

def main(train_mode=False,
         test_video=False,
	 rev_train=False,
	 rev_test=False):
    """
    :param train_mode: if we want to train the model from scratch/fine_tune
    :param test_video:  run the model to get predictions
    :return:
    """

    """
    bypassing the model name and train/evaluate the inverse model
    """
    cfg = config.Config()
    cfg.model_name = 'rev_' + cfg.model_name
    #cfg.model_output = os.path.join(cfg.base_dir, 'models', cfg.model_name+'_'+cfg.model_suffix) 
    cfg.model_output = os.path.join(cfg.base_dir, 'models', cfg.model_name+'_training_data_binned_{}_nbins_{}_n_{}'.format(int(cfg.isBinned),cfg.nBins,cfg.nDatapoints))
    
    if rev_train:
	train_reverse_model(cfg)
	os._exit(0)
    elif rev_test:
	test_rev_model_eval(cfg)
	os._exit(0)

    """
    else, see if we want to train the model or run the evaluation pipeline.
    Note: Evalutaion means that we have labels to calculate accuracy as well and these 
    have been written into tfrecords
    """
    if train_mode:
        train_model(config.Config())
    else:
        test_model_eval(config.Config())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        dest="train_mode",
        action="store_true",
        help='flag to train the model')

    parser.add_argument(
        "--test",
        dest="test_video",
        action="store_true",
        help='flag to run model on a new video')

    parser.add_argument(
	"--revtrain",
	dest="rev_train",
	action="store_true",
	help='flag to run the inverse training model')
    parser.add_argument(
	"--revtest",
	dest="rev_test",
	action="store_true",
	help='flag to run the inverse training model')

    args = parser.parse_args()
    main(**vars(args))
