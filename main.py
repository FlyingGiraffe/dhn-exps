from absl import app
from absl import flags
from ml_collections.config_flags import config_flags


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_enum(
  'mode', None, ['train', 'extract', 'generate'],
  'Running mode: train or eval')
flags.DEFINE_enum('dataset_split', 'train', ['train', 'test'], 'Dataset split: train or test.')
flags.DEFINE_string('work_subdir', None, 'Sub-directory for inference.')
flags.mark_flags_as_required(['config'])


def main(argv):
  
  if FLAGS.mode == 'train':
    from trainers.trainer import Trainer
    trainer = Trainer(FLAGS.config)
    trainer.train_and_eval()
  
  elif FLAGS.mode == 'extract':
    from trainers.extractor import Extractor
    extract_dir = FLAGS.work_subdir if FLAGS.work_subdir is not None else 'extract'
    extractor = Extractor(FLAGS.config, extract_dir=extract_dir)
    extractor.train_and_eval()
  
  elif FLAGS.mode == 'generate':
    from trainers.generator import Generator
    generate_dir = FLAGS.work_subdir if FLAGS.work_subdir is not None else 'gen_sequence'
    generator = Generator(FLAGS.config, generate_dir=generate_dir, dataset_split=FLAGS.dataset_split)
    generator.generate()

if __name__ == '__main__':
  app.run(main)  