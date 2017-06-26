import tensorflow as tf
import re

def _safe_define(name, default_value, helper):
	try:
		if isinstance(default_value, bool):
			tf.app.flags.DEFINE_bool(name, default_value, helper)
		elif isinstance(default_value, int):
			tf.app.flags.DEFINE_integer(name, default_value, helper)
		elif isinstance(default_value, float):
			tf.app.flags.DEFINE_float(name, default_value, helper)
		elif isinstance(default_value, str):
			tf.app.flags.DEFINE_string(name, default_value, helper)
	except:
		pass


_safe_define('train_batch_size', 128, "Number of samples to process in a training batch.")
_safe_define('test_batch_size', 0, "Number of samples to process in a test batch(0 = all).")
_safe_define('BN_DECAY', 0.9, "Normalization decay used for BatchNorm.")
_safe_define('INIT_FACTOR', 1.0, 'Initialization factor used to scale ALL weight variables initializations.')
_safe_define('RESULTS_FILE', './results/{DATASET_NAME}/results.txt', 'Save results to file.')
_safe_define('NO_FOLDS', 10, 'Number of folds used on CV (default 10).')



_safe_define('optimizer', 'adam', 'Adam (default) or Momentum.')
_safe_define('starter_learning_rate', 0.01, 'Starter learning rate.')
_safe_define('learning_rate_exp', 0.1, 'Learning rate step size.')
_safe_define('learning_rate_step', 1000, 'Learning rate step period.')

_safe_define('display_iter', 5, 'Display stats period.')
_safe_define('save_checkpoints', True, 'Should create checkpoints based on checkpoint rules (True/False).')
_safe_define('snapshot_iter', 1000, 'Snapshot period.')
_safe_define('iterations_per_test', 5, 'Training iterations per test iteration.')


_safe_define('summary_save', True, 'Should save summaries (True/False).')
_safe_define('summary_period', 5, 'Save summaries after x iterations (Default 5).')


_safe_define('silent', False, 'Silent mode.')



FLAGS = tf.app.flags.FLAGS

def get_regex_flag(s):
	pattern = re.compile('|'.join([ '{' + k + '}' for k in FLAGS.__flags.keys()]))
	return pattern.sub(lambda x: FLAGS.__flags[x.group()[1:-1]], FLAGS.__flags[s])