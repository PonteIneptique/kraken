#
# Copyright 2015 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
import os
import glob
import uuid
import click
import logging
import unicodedata

from PIL import Image
from rich.traceback import install
from bidi.algorithm import get_display

from typing import cast, Set, List, IO, Any, Dict

from kraken.lib import log
from kraken.lib.progress import KrakenProgressBar
from kraken.lib.exceptions import KrakenCairoSurfaceException
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.default_specs import (SEGMENTATION_HYPER_PARAMS,
                                      RECOGNITION_HYPER_PARAMS,
                                      SEGMENTATION_SPEC,
                                      RECOGNITION_SPEC)

APP_NAME = 'kraken'

logging.captureWarnings(True)
logger = logging.getLogger('kraken')

# install rich traceback handler
install(suppress=[click])

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


def message(msg, **styles):
    if logger.getEffectiveLevel() >= 30:
        click.secho(msg, **styles)


@click.group()
@click.version_option()
@click.pass_context
@click.option('-v', '--verbose', default=0, count=True)
@click.option('-s', '--seed', default=None, type=click.INT,
              help='Seed for numpy\'s and torch\'s RNG. Set to a fixed value to '
                   'ensure reproducible random splits of data')
def cli(ctx, verbose, seed):
    if seed:
        import numpy.random
        numpy.random.seed(seed)
        from torch import manual_seed
        manual_seed(seed)

    ctx.meta['verbose'] = verbose
    log.set_logger(logger, level=30 - min(10 * verbose, 20))


def _validate_manifests(ctx, param, value):
    images = []
    for manifest in value:
        for entry in manifest.readlines():
            im_p = entry.rstrip('\r\n')
            if os.path.isfile(im_p):
                images.append(im_p)
            else:
                logger.warning('Invalid entry "{}" in {}'.format(im_p, manifest.name))
    return images


def _expand_gt(ctx, param, value):
    images = []
    for expression in value:
        images.extend([x for x in glob.iglob(expression, recursive=True) if os.path.isfile(x)])
    return images


def _validate_merging(ctx, param, value):
    """
    Maps baseline/region merging to a dict of merge structures.
    """
    if not value:
        return None
    merge_dict = {}  # type: Dict[str, str]
    try:
        for m in value:
            k, v = m.split(':')
            merge_dict[v] = k  # type: ignore
    except Exception:
        raise click.BadParameter('Mappings must be in format target:src')
    return merge_dict


@cli.command('segtrain')
@click.pass_context
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('-s', '--spec', show_default=True,
              default=SEGMENTATION_SPEC,
              help='VGSL spec of the baseline labeling network')
@click.option('--line-width',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['line_width'],
              help='The height of each baseline in the target after scaling')
@click.option('-i', '--load', show_default=True, type=click.Path(exists=True,
              readable=True), help='Load existing file to continue training')
@click.option('-F', '--freq', show_default=True, default=SEGMENTATION_HYPER_PARAMS['freq'], type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['quit'],
              type=click.Choice(['early',
                                 'dumb']),
              help='Stop condition for training. Set to `early` for early stopping or `dumb` for fixed number of epochs')
@click.option('-N',
              '--epochs',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['epochs'],
              help='Number of epochs to train for')
@click.option('--min-epochs',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['min_epochs'],
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--lag',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['lag'],
              help='Number of evaluations (--report frequence) to wait before stopping training without improvement')
@click.option('--min-delta',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['min_delta'],
              type=click.FLOAT,
              help='Minimum improvement between epochs to reset early stopping. By default it scales the delta by the best loss')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--optimizer',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['optimizer'],
              type=click.Choice(['Adam',
                                 'SGD',
                                 'RMSprop',
                                 'Lamb']),
              help='Select optimizer')
@click.option('-r', '--lrate', show_default=True, default=SEGMENTATION_HYPER_PARAMS['lrate'], help='Learning rate')
@click.option('-m', '--momentum', show_default=True, default=SEGMENTATION_HYPER_PARAMS['momentum'], help='Momentum')
@click.option('-w', '--weight-decay', show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['weight_decay'], help='Weight decay')
@click.option('--schedule',
              show_default=True,
              type=click.Choice(['constant',
                                 '1cycle',
                                 'exponential',
                                 'cosine',
                                 'step',
                                 'reduceonplateau']),
              default=SEGMENTATION_HYPER_PARAMS['schedule'],
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--step-size` option.')
@click.option('-g',
              '--gamma',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['gamma'],
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['step_size'],
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['rop_patience'],
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['cos_t_max'],
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('-p', '--partition', show_default=True, default=0.9,
              help='Ground truth data partition ratio between train/validation set')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--workers', show_default=True, default=1, help='Number of OpenMP threads and workers when running on CPU.')
@click.option('--load-hyper-parameters/--no-load-hyper-parameters', show_default=True, default=False,
              help='When loading an existing model, retrieve hyper-parameters from the model')
@click.option('--force-binarization/--no-binarization', show_default=True,
              default=False, help='Forces input images to be binary, otherwise '
              'the appropriate color format will be auto-determined through the '
              'network specification. Will be ignored in `path` mode.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page']), default='xml',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information.')
@click.option('--suppress-regions/--no-suppress-regions', show_default=True,
              default=False, help='Disables region segmentation training.')
@click.option('--suppress-baselines/--no-suppress-baselines', show_default=True,
              default=False, help='Disables baseline segmentation training.')
@click.option('-vr', '--valid-regions', show_default=True, default=None, multiple=True,
              help='Valid region types in training data. May be used multiple times.')
@click.option('-vb', '--valid-baselines', show_default=True, default=None, multiple=True,
              help='Valid baseline types in training data. May be used multiple times.')
@click.option('-mr',
              '--merge-regions',
              show_default=True,
              default=None,
              help='Region merge mapping. One or more mappings of the form `$target:$src` where $src is merged into $target.',
              multiple=True,
              callback=_validate_merging)
@click.option('-mb',
              '--merge-baselines',
              show_default=True,
              default=None,
              help='Baseline type merge mapping. Same syntax as `--merge-regions`',
              multiple=True,
              callback=_validate_merging)
@click.option('-br', '--bounding-regions', show_default=True, default=None, multiple=True,
              help='Regions treated as boundaries for polygonization purposes. May be used multiple times.')
@click.option('--augment/--no-augment',
              show_default=True,
              default=SEGMENTATION_HYPER_PARAMS['augment'],
              help='Enable image augmentation')
@click.option('--resize', show_default=True, default='fail', type=click.Choice(['add', 'both', 'fail']),
              help='Output layer resizing option. If set to `add` new classes will be '
                   'added, `both` will set the layer to match exactly '
                   'the training data classes, `fail` will abort if training data and model '
                   'classes do not match.')
@click.option('-tl', '--topline', 'topline', show_default=True, flag_value='topline',
              help='Switch for the baseline location in the scripts. '
                   'Set to topline if the data is annotated with a hanging baseline, as is '
                   'common with Hebrew, Bengali, Devanagari, etc. Set to '
                   ' centerline for scripts annotated with a central line.')
@click.option('-cl', '--centerline', 'topline', flag_value='centerline')
@click.option('-bl', '--baseline', 'topline', flag_value='baseline', default='baseline')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def segtrain(ctx, output, spec, line_width, load, freq, quit, epochs, min_epochs,
             lag, min_delta, device, optimizer, lrate, momentum, weight_decay,
             schedule, gamma, step_size, sched_patience, cos_max, partition,
             training_files, evaluation_files, workers, load_hyper_parameters,
             force_binarization, format_type, suppress_regions,
             suppress_baselines, valid_regions, valid_baselines, merge_regions,
             merge_baselines, bounding_regions, augment, resize, topline, ground_truth):
    """
    Trains a baseline labeling model for layout analysis
    """
    import shutil

    from kraken.lib.train import SegmentationModel, KrakenTrainer

    if resize != 'fail' and not load:
        raise click.BadOptionUsage('resize', 'resize option requires loading an existing model')

    if not (0 <= freq <= 1) and freq % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if augment:
        try:
            import albumentations
        except ImportError:
            raise click.BadOptionUsage('augment', 'augmentation needs the `albumentations` package installed.')

    logger.info('Building ground truth set from {} document images'.format(len(ground_truth) + len(training_files)))

    # populate hyperparameters from command line args
    hyper_params = SEGMENTATION_HYPER_PARAMS.copy()
    hyper_params.update({'line_width': line_width,
                         'freq': freq,
                         'quit': quit,
                         'epochs': epochs,
                         'min_epochs': min_epochs,
                         'lag': lag,
                         'min_delta': min_delta,
                         'optimizer': optimizer,
                         'lrate': lrate,
                         'momentum': momentum,
                         'weight_decay': weight_decay,
                         'schedule': schedule,
                         'augment': augment,
                         'gamma': gamma,
                         'step_size': step_size,
                         'rop_patience': sched_patience,
                         'cos_t_max': cos_max})

    # disable automatic partition when given evaluation set explicitly
    if evaluation_files:
        partition = 1
    ground_truth = list(ground_truth)

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    loc = {'topline': True,
           'baseline': False,
           'centerline': None}

    topline = loc[topline]

    if device == 'cpu':
        device = None
    elif device.startswith('cuda'):
        device = [int(device.split(':')[-1])]

    if hyper_params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(hyper_params['freq'])}
    else:
        val_check_interval = {'val_check_interval': hyper_params['freq']}

    model = SegmentationModel(hyper_params,
                              output=output,
                              spec=spec,
                              model=load,
                              training_data=ground_truth,
                              evaluation_data=evaluation_files,
                              partition=partition,
                              num_workers=workers,
                              load_hyper_parameters=load_hyper_parameters,
                              force_binarization=force_binarization,
                              format_type=format_type,
                              suppress_regions=suppress_regions,
                              suppress_baselines=suppress_baselines,
                              valid_regions=valid_regions,
                              valid_baselines=valid_baselines,
                              merge_regions=merge_regions,
                              merge_baselines=merge_baselines,
                              bounding_regions=bounding_regions,
                              resize=resize,
                              topline=topline)

    message('Training line types:')
    for k, v in model.train_set.dataset.class_mapping['baselines'].items():
        message(f'  {k}\t{v}\t{model.train_set.dataset.class_stats["baselines"][k]}')
    message('Training region types:')
    for k, v in model.train_set.dataset.class_mapping['regions'].items():
        message(f'  {k}\t{v}\t{model.train_set.dataset.class_stats["regions"][k]}')

    if len(model.train_set) == 0:
        raise click.UsageError('No valid training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    trainer = KrakenTrainer(gpus=device,
                            max_epochs=hyper_params['epochs'] if hyper_params['quit'] == 'dumb' else -1,
                            min_epochs=hyper_params['min_epochs'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            **val_check_interval)

    trainer.fit(model)

    if quit == 'early':
        message('Moving best model {0}_{1}.mlmodel ({2}) to {0}_best.mlmodel'.format(
            output, trainer.stopper.best_epoch, trainer.stopper.best_loss))
        logger.info('Moving best model {0}_{1}.mlmodel ({2}) to {0}_best.mlmodel'.format(
            output, trainer.stopper.best_epoch, trainer.stopper.best_loss))
        shutil.copy(f'{output}_{trainer.stopper.best_epoch}.mlmodel', f'{output}_best.mlmodel')


@cli.command('train')
@click.pass_context
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=RECOGNITION_HYPER_PARAMS['batch_size'], help='batch sample size')
@click.option('--pad', show_default=True, type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('-s', '--spec', show_default=True, default=RECOGNITION_SPEC,
              help='VGSL spec of the network to train. CTC layer will be added automatically.')
@click.option('-a', '--append', show_default=True, default=None, type=click.INT,
              help='Removes layers before argument and then appends spec. Only works when loading an existing model')
@click.option('-i', '--load', show_default=True, type=click.Path(exists=True,
              readable=True), help='Load existing file to continue training')
@click.option('-F', '--freq', show_default=True, default=RECOGNITION_HYPER_PARAMS['freq'], type=click.FLOAT,
              help='Model saving and report generation frequency in epochs '
                   'during training. If frequency is >1 it must be an integer, '
                   'i.e. running validation every n-th epoch.')
@click.option('-q',
              '--quit',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['quit'],
              type=click.Choice(['early',
                                 'dumb']),
              help='Stop condition for training. Set to `early` for early stooping or `dumb` for fixed number of epochs')
@click.option('-N',
              '--epochs',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['epochs'],
              help='Number of epochs to train for')
@click.option('--min-epochs',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['min_epochs'],
              help='Minimal number of epochs to train for when using early stopping.')
@click.option('--lag',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['lag'],
              help='Number of evaluations (--report frequence) to wait before stopping training without improvement')
@click.option('--min-delta',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['min_delta'],
              type=click.FLOAT,
              help='Minimum improvement between epochs to reset early stopping. Default is scales the delta by the best loss')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--optimizer',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['optimizer'],
              type=click.Choice(['Adam',
                                 'SGD',
                                 'RMSprop',
                                 'Lamb']),
              help='Select optimizer')
@click.option('-r', '--lrate', show_default=True, default=RECOGNITION_HYPER_PARAMS['lrate'], help='Learning rate')
@click.option('-m', '--momentum', show_default=True, default=RECOGNITION_HYPER_PARAMS['momentum'], help='Momentum')
@click.option('-w', '--weight-decay', show_default=True, type=float,
              default=RECOGNITION_HYPER_PARAMS['weight_decay'], help='Weight decay')
@click.option('--schedule',
              show_default=True,
              type=click.Choice(['constant',
                                 '1cycle',
                                 'exponential',
                                 'cosine',
                                 'step',
                                 'reduceonplateau']),
              default=RECOGNITION_HYPER_PARAMS['schedule'],
              help='Set learning rate scheduler. For 1cycle, cycle length is determined by the `--epoch` option.')
@click.option('-g',
              '--gamma',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['gamma'],
              help='Decay factor for exponential, step, and reduceonplateau learning rate schedules')
@click.option('-ss',
              '--step-size',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['step_size'],
              help='Number of validation runs between learning rate decay for exponential and step LR schedules')
@click.option('--sched-patience',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['rop_patience'],
              help='Minimal number of validation runs between LR reduction for reduceonplateau LR schedule.')
@click.option('--cos-max',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['cos_t_max'],
              help='Epoch of minimal learning rate for cosine LR scheduler.')
@click.option('-p', '--partition', show_default=True, default=0.9,
              help='Ground truth data partition ratio between train/validation set')
@click.option('--fixed-splits/--ignore-fixed-split', show_default=True, default=False,
              help='Whether to honor fixed splits in binary datasets.')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=RECOGNITION_HYPER_PARAMS['normalization'], help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace', show_default=True,
              default=RECOGNITION_HYPER_PARAMS['normalize_whitespace'], help='Normalizes unicode whitespace')
@click.option('-c', '--codec', show_default=True, default=None, type=click.File(mode='r', lazy=True),
              help='Load a codec JSON definition (invalid if loading existing model)')
@click.option('--resize', show_default=True, default='fail', type=click.Choice(['add', 'both', 'fail']),
              help='Codec/output layer resizing option. If set to `add` code '
                   'points will be added, `both` will set the layer to match exactly '
                   'the training data, `fail` will abort if training data and model '
                   'codec do not match.')
@click.option('--reorder/--no-reorder', show_default=True, default=True, help='Reordering of code points to display order')
@click.option('--base-dir', show_default=True, default='auto',
              type=click.Choice(['L', 'R', 'auto']), help='Set base text '
              'direction.  This should be set to the direction used during the '
              'creation of the training data. If set to `auto` it will be '
              'overridden by any explicit value given in the input files.')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('--workers', show_default=True, default=1, help='Number of OpenMP threads and workers when running on CPU.')
@click.option('--load-hyper-parameters/--no-load-hyper-parameters', show_default=True, default=False,
              help='When loading an existing model, retrieve hyperparameters from the model')
@click.option('--repolygonize/--no-repolygonize', show_default=True,
              default=False, help='Repolygonizes line data in ALTO/PageXML '
              'files. This ensures that the trained model is compatible with the '
              'segmenter in kraken even if the original image files either do '
              'not contain anything but transcriptions and baseline information '
              'or the polygon data was created using a different method. Will '
              'be ignored in `path` mode. Note that this option will be slow '
              'and will not scale input images to the same size as the segmenter '
              'does.')
@click.option('--force-binarization/--no-binarization', show_default=True,
              default=False, help='Forces input images to be binary, otherwise '
              'the appropriate color format will be auto-determined through the '
              'network specification. Will be ignored in `path` mode.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page', 'binary']), default='path',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both line definitions and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with `.gt.txt` text files '
              'containing the transcription. In binary mode files are datasets '
              'files containing pre-extracted text lines.')
@click.option('--augment/--no-augment',
              show_default=True,
              default=RECOGNITION_HYPER_PARAMS['augment'],
              help='Enable image augmentation')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def train(ctx, batch_size, pad, output, spec, append, load, freq, quit, epochs,
          min_epochs, lag, min_delta, device, optimizer, lrate, momentum,
          weight_decay, schedule, gamma, step_size, sched_patience, cos_max,
          partition, fixed_splits, normalization, normalize_whitespace, codec,
          resize, reorder, base_dir, training_files, evaluation_files, workers,
          load_hyper_parameters, repolygonize, force_binarization, format_type,
          augment, ground_truth):
    """
    Trains a model from image-text pairs.
    """
    if not load and append:
        raise click.BadOptionUsage('append', 'append option requires loading an existing model')

    if resize != 'fail' and not load:
        raise click.BadOptionUsage('resize', 'resize option requires loading an existing model')

    if not (0 <= freq <= 1) and freq % 1.0 != 0:
        raise click.BadOptionUsage('freq', 'freq needs to be either in the interval [0,1.0] or a positive integer.')

    if augment:
        try:
            import albumentations
        except ImportError:
            raise click.BadOptionUsage('augment', 'augmentation needs the `albumentations` package installed.')

    import json
    import shutil
    from kraken.lib.train import RecognitionModel, KrakenTrainer

    hyper_params = RECOGNITION_HYPER_PARAMS.copy()
    hyper_params.update({'freq': freq,
                         'pad': pad,
                         'batch_size': batch_size,
                         'quit': quit,
                         'epochs': epochs,
                         'min_epochs': min_epochs,
                         'lag': lag,
                         'min_delta': min_delta,
                         'optimizer': optimizer,
                         'lrate': lrate,
                         'momentum': momentum,
                         'weight_decay': weight_decay,
                         'schedule': schedule,
                         'gamma': gamma,
                         'step_size': step_size,
                         'rop_patience': sched_patience,
                         'cos_t_max': cos_max,
                         'normalization': normalization,
                         'normalize_whitespace': normalize_whitespace,
                         'augment': augment})

    # disable automatic partition when given evaluation set explicitly
    if evaluation_files:
        partition = 1
    ground_truth = list(ground_truth)

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    if reorder and base_dir != 'auto':
        reorder = base_dir

    if codec:
        logger.debug(f'Loading codec file from {codec}')
        codec = json.load(codec)

    if device == 'cpu':
        device = None
    elif device.startswith('cuda'):
        device = [int(device.split(':')[-1])]

    if hyper_params['freq'] > 1:
        val_check_interval = {'check_val_every_n_epoch': int(hyper_params['freq'])}
    else:
        val_check_interval = {'val_check_interval': hyper_params['freq']}

    model = RecognitionModel(hyper_params=hyper_params,
                             output=output,
                             spec=spec,
                             append=append,
                             model=load,
                             reorder=reorder,
                             training_data=ground_truth,
                             evaluation_data=evaluation_files,
                             partition=partition,
                             binary_dataset_split=fixed_splits,
                             num_workers=workers,
                             load_hyper_parameters=load_hyper_parameters,
                             repolygonize=repolygonize,
                             force_binarization=force_binarization,
                             format_type=format_type,
                             codec=codec,
                             resize=resize)

    trainer = KrakenTrainer(gpus=device,
                            max_epochs=hyper_params['epochs'] if hyper_params['quit'] == 'dumb' else -1,
                            min_epochs=hyper_params['min_epochs'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            **val_check_interval)
    try:
        trainer.fit(model)
    except KrakenInputException as e:
        if e.args[0].startswith('Training data and model codec alphabets mismatch') and resize == 'fail':
            raise click.BadOptionUsage('resize', 'Mismatched training data for loaded model. Set option `--resize` to `add` or `both`')
        else:
            raise e

    if quit == 'early':
        message('Moving best model {0}_{1}.mlmodel ({2}) to {0}_best.mlmodel'.format(
            output, model.best_epoch, model.best_metric))
        logger.info('Moving best model {0}_{1}.mlmodel ({2}) to {0}_best.mlmodel'.format(
            output, model.best_epoch, model.best_metric))
        shutil.copy(f'{output}_{model.best_epoch}.mlmodel', f'{output}_best.mlmodel')


@cli.command('test')
@click.pass_context
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=RECOGNITION_HYPER_PARAMS['batch_size'], help='Batch sample size')
@click.option('-m', '--model', show_default=True, type=click.Path(exists=True, readable=True),
              multiple=True, help='Model(s) to evaluate')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--pad', show_default=True, type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('--workers', show_default=True, default=1, help='Number of OpenMP threads when running on CPU.')
@click.option('--reorder/--no-reorder', show_default=True, default=True, help='Reordering of code points to display order')
@click.option('--base-dir', show_default=True, default='auto',
              type=click.Choice(['L', 'R', 'auto']), help='Set base text '
              'direction.  This should be set to the direction used during the '
              'creation of the training data. If set to `auto` it will be '
              'overridden by any explicit value given in the input files.')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=None, help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace',
              show_default=True, default=True, help='Normalizes unicode whitespace')
@click.option('--repolygonize/--no-repolygonize', show_default=True,
              default=False, help='Repolygonizes line data in ALTO/PageXML '
              'files. This ensures that the trained model is compatible with the '
              'segmenter in kraken even if the original image files either do '
              'not contain anything but transcriptions and baseline information '
              'or the polygon data was created using a different method. Will '
              'be ignored in `path` mode. Note, that this option will be slow '
              'and will not scale input images to the same size as the segmenter '
              'does.')
@click.option('--force-binarization/--no-binarization', show_default=True,
              default=False, help='Forces input images to be binary, otherwise '
              'the appropriate color format will be auto-determined through the '
              'network specification. Will be ignored in `path` mode.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page', 'binary']), default='path',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information. In `binary` mode files are '
              'collections of pre-extracted text line images.')
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def test(ctx, batch_size, model, evaluation_files, device, pad, workers,
         reorder, base_dir, normalization, normalize_whitespace, repolygonize,
         force_binarization, format_type, test_set):
    """
    Evaluate on a test set.
    """
    if not model:
        raise click.UsageError('No model to evaluate given.')

    import numpy as np
    from torch.utils.data import DataLoader

    from kraken.serialization import render_report
    from kraken.lib import models
    from kraken.lib.xml import preparse_xml_data
    from kraken.lib.dataset import (global_align, compute_confusions,
                                    PolygonGTDataset, GroundTruthDataset,
                                    ImageInputTransforms,
                                    ArrowIPCRecognitionDataset,
                                    collate_sequences)

    logger.info('Building test set from {} line images'.format(len(test_set) + len(evaluation_files)))

    nn = {}
    for p in model:
        message('Loading model {}\t'.format(p), nl=False)
        nn[p] = models.load_any(p)
        message('\u2713', fg='green')

    test_set = list(test_set)

    # set number of OpenMP threads
    next(iter(nn.values())).nn.set_num_threads(1)

    if evaluation_files:
        test_set.extend(evaluation_files)

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    if format_type in ['xml', 'page', 'alto']:
        if repolygonize:
            message('Repolygonizing data')
        test_set = preparse_xml_data(test_set, format_type, repolygonize)
        valid_norm = False
        DatasetClass = PolygonGTDataset
    elif format_type == 'binary':
        DatasetClass = ArrowIPCRecognitionDataset
        if repolygonize:
            logger.warning('Repolygonization enabled in `binary` mode. Will be ignored.')
        test_set = [{'file': file} for file in test_set]
        valid_norm = False
    else:
        DatasetClass = GroundTruthDataset
        if force_binarization:
            logger.warning('Forced binarization enabled in `path` mode. Will be ignored.')
            force_binarization = False
        if repolygonize:
            logger.warning('Repolygonization enabled in `path` mode. Will be ignored.')
        test_set = [{'image': img} for img in test_set]
        valid_norm = True

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    if reorder and base_dir != 'auto':
        reorder = base_dir

    acc_list = []
    for p, net in nn.items():
        algn_gt: List[str] = []
        algn_pred: List[str] = []
        chars = 0
        error = 0
        message('Evaluating {}'.format(p))
        logger.info('Evaluating {}'.format(p))
        batch, channels, height, width = net.nn.input
        ts = ImageInputTransforms(batch, height, width, channels, pad, valid_norm, force_binarization)
        ds = DatasetClass(normalization=normalization,
                          whitespace_normalization=normalize_whitespace,
                          reorder=reorder,
                          im_transforms=ts)
        for line in test_set:
            try:
                ds.add(**line)
            except KrakenInputException as e:
                logger.info(e)
        # don't encode validation set as the alphabets may not match causing encoding failures
        ds.no_encode()
        ds_loader = DataLoader(ds,
                               batch_size=batch_size,
                               num_workers=workers,
                               pin_memory=True,
                               collate_fn=collate_sequences)

        with KrakenProgressBar() as progress:
            batches = len(ds_loader)
            pred_task = progress.add_task('Evaluating', total=batches, visible=True if not ctx.meta['verbose'] else False)

            for batch in ds_loader:
                im = batch['image']
                text = batch['target']
                lens = batch['seq_lens']
                try:
                    pred = net.predict_string(im, lens)
                    for x, y in zip(pred, text):
                        chars += len(y)
                        c, algn1, algn2 = global_align(y, x)
                        algn_gt.extend(algn1)
                        algn_pred.extend(algn2)
                        error += c
                except FileNotFoundError as e:
                    batches -= 1
                    progress.update(pred_task, total=batches)
                    logger.warning('{} {}. Skipping.'.format(e.strerror, e.filename))
                except KrakenInputException as e:
                    batches -= 1
                    progress.update(pred_task, total=batches)
                    logger.warning(str(e))
                progress.update(pred_task, advance=1)

        acc_list.append((chars - error) / chars)
        confusions, scripts, ins, dels, subs = compute_confusions(algn_gt, algn_pred)
        rep = render_report(p, chars, error, confusions, scripts, ins, dels, subs)
        logger.info(rep)
        message(rep)
    logger.info('Average accuracy: {:0.2f}%, (stddev: {:0.2f})'.format(np.mean(acc_list) * 100, np.std(acc_list) * 100))
    message('Average accuracy: {:0.2f}%, (stddev: {:0.2f})'.format(np.mean(acc_list) * 100, np.std(acc_list) * 100))


@cli.command('extract')
@click.pass_context
@click.option('-b', '--binarize/--no-binarize', show_default=True, default=True,
              help='Binarize color/grayscale images')
@click.option('-u', '--normalization', show_default=True,
              type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']), default=None,
              help='Normalize ground truth')
@click.option('-s', '--normalize-whitespace/--no-normalize-whitespace',
              show_default=True, default=True, help='Normalizes unicode whitespace')
@click.option('-n', '--reorder/--no-reorder', default=False, show_default=True,
              help='Reorder transcribed lines to display order')
@click.option('-r', '--rotate/--no-rotate', default=True, show_default=True,
              help='Skip rotation of vertical lines')
@click.option('-o', '--output', type=click.Path(), default='training', show_default=True,
              help='Output directory')
@click.option('--format',
              default='{idx:06d}',
              show_default=True,
              help='Format for extractor output. valid fields are `src` (source file), `idx` (line number), and `uuid` (v4 uuid)')
@click.argument('transcriptions', nargs=-1, type=click.File(lazy=True))
def extract(ctx, binarize, normalization, normalize_whitespace, reorder,
            rotate, output, format, transcriptions):
    """
    Extracts image-text pairs from a transcription environment created using
    ``ketos transcribe``.
    """
    import regex
    import base64

    from io import BytesIO
    from PIL import Image
    from lxml import html, etree

    from kraken import binarization

    try:
        os.mkdir(output)
    except Exception:
        pass

    text_transforms = []
    if normalization:
        text_transforms.append(lambda x: unicodedata.normalize(normalization, x))
    if normalize_whitespace:
        text_transforms.append(lambda x: regex.sub(r'\s', ' ', x))
    if reorder:
        text_transforms.append(get_display)

    idx = 0
    manifest = []

    with KrakenProgressBar() as progress:
        read_task = progress.add_task('Reading transcriptions', total=len(transcriptions), visible=True if not ctx.meta['verbose'] else False)

        for fp in transcriptions:
            logger.info('Reading {}'.format(fp.name))
            doc = html.parse(fp)
            etree.strip_tags(doc, etree.Comment)
            td = doc.find(".//meta[@itemprop='text_direction']")
            if td is None:
                td = 'horizontal-lr'
            else:
                td = td.attrib['content']

            im = None
            dest_dict = {'output': output, 'idx': 0, 'src': fp.name, 'uuid': str(uuid.uuid4())}
            for section in doc.xpath('//section'):
                img = section.xpath('.//img')[0].get('src')
                fd = BytesIO(base64.b64decode(img.split(',')[1]))
                im = Image.open(fd)
                if not im:
                    logger.info('Skipping {} because image not found'.format(fp.name))
                    break
                if binarize:
                    im = binarization.nlbin(im)
                for line in section.iter('li'):
                    if line.get('contenteditable') and (not u''.join(line.itertext()).isspace() and u''.join(line.itertext())):
                        dest_dict['idx'] = idx
                        dest_dict['uuid'] = str(uuid.uuid4())
                        logger.debug('Writing line {:06d}'.format(idx))
                        l_img = im.crop([int(x) for x in line.get('data-bbox').split(',')])
                        if rotate and td.startswith('vertical'):
                            im.rotate(90, expand=True)
                        l_img.save(('{output}/' + format + '.png').format(**dest_dict))
                        manifest.append((format + '.png').format(**dest_dict))
                        text = u''.join(line.itertext()).strip()
                        for func in text_transforms:
                            text = func(text)
                        with open(('{output}/' + format + '.gt.txt').format(**dest_dict), 'wb') as t:
                            t.write(text.encode('utf-8'))
                        idx += 1

            progress.update(read_task, advance=1)

    logger.info('Extracted {} lines'.format(idx))
    with open('{}/manifest.txt'.format(output), 'w') as fp:
        fp.write('\n'.join(manifest))


@cli.command('transcribe')
@click.pass_context
@click.option('-d', '--text-direction', default='horizontal-lr',
              type=click.Choice(['horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl']),
              help='Sets principal text direction', show_default=True)
@click.option('--scale', default=None, type=click.FLOAT)
@click.option('--bw/--orig', default=True, show_default=True,
              help="Put nonbinarized images in output")
@click.option('-m', '--maxcolseps', default=2, type=click.INT, show_default=True)
@click.option('-b/-w', '--black_colseps/--white_colseps', default=False, show_default=True)
@click.option('-f', '--font', default='',
              help='Font family to use')
@click.option('-fs', '--font-style', default=None,
              help='Font style to use')
@click.option('-p', '--prefill', default=None,
              help='Use given model for prefill mode.')
@click.option('--pad', show_default=True, type=(int, int), default=(0, 0),
              help='Left and right padding around lines')
@click.option('-l', '--lines', type=click.Path(exists=True), show_default=True,
              help='JSON file containing line coordinates')
@click.option('-o', '--output', type=click.File(mode='wb'), default='transcription.html',
              help='Output file', show_default=True)
@click.argument('images', nargs=-1, type=click.File(mode='rb', lazy=True))
def transcription(ctx, text_direction, scale, bw, maxcolseps,
                  black_colseps, font, font_style, prefill, pad, lines, output,
                  images):
    """
    Creates transcription environments for ground truth generation.
    """
    import json

    from PIL import Image

    from kraken import rpred
    from kraken import pageseg
    from kraken import transcribe
    from kraken import binarization

    from kraken.lib import models

    ti = transcribe.TranscriptionInterface(font, font_style)

    if len(images) > 1 and lines:
        raise click.UsageError('--lines option is incompatible with multiple image files')

    if prefill:
        logger.info('Loading model {}'.format(prefill))
        message('Loading ANN', nl=False)
        prefill = models.load_any(prefill)
        message('\u2713', fg='green')

    with KrakenProgressBar() as progress:
        read_task = progress.add_task('Reading images', total=len(images), visible=True if not ctx.meta['verbose'] else False)
        for fp in images:
            logger.info('Reading {}'.format(fp.name))
            im = Image.open(fp)
            if im.mode not in ['1', 'L', 'P', 'RGB']:
                logger.warning('Input {} is in {} color mode. Converting to RGB'.format(fp.name, im.mode))
                im = im.convert('RGB')
            logger.info('Binarizing page')
            im_bin = binarization.nlbin(im)
            im_bin = im_bin.convert('1')
            logger.info('Segmenting page')
            if not lines:
                res = pageseg.segment(im_bin, text_direction, scale, maxcolseps, black_colseps, pad=pad)
            else:
                with click.open_file(lines, 'r') as fp:
                    try:
                        fp = cast(IO[Any], fp)
                        res = json.load(fp)
                    except ValueError as e:
                        raise click.UsageError('{} invalid segmentation: {}'.format(lines, str(e)))
            if prefill:
                it = rpred.rpred(prefill, im_bin, res.copy())
                preds = []
                logger.info('Recognizing')
                for pred in it:
                    logger.debug('{}'.format(pred.prediction))
                    preds.append(pred)
                ti.add_page(im, res, records=preds)
            else:
                ti.add_page(im, res)
            fp.close()
            progress.update(read_task, advance=1)

    logger.info('Writing transcription to {}'.format(output.name))
    message('Writing output ', nl=False)
    ti.write(output)
    message('\u2713', fg='green')


@cli.command('linegen')
@click.pass_context
@click.option('-f', '--font', default='sans',
              help='Font family to render texts in.')
@click.option('-n', '--maxlines', type=click.INT, default=0,
              help='Maximum number of lines to generate')
@click.option('-e', '--encoding', default='utf-8',
              help='Decode text files with given codec.')
@click.option('-u', '--normalization',
              type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']), default=None,
              help='Normalize ground truth')
@click.option('-ur', '--renormalize',
              type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']), default=None,
              help='Renormalize text for rendering purposes.')
@click.option('--reorder/--no-reorder', default=False, help='Reorder code points to display order')
@click.option('-fs', '--font-size', type=click.INT, default=32,
              help='Font size to render texts in.')
@click.option('-fw', '--font-weight', type=click.INT, default=400,
              help='Font weight to render texts in.')
@click.option('-l', '--language',
              help='RFC-3066 language tag for language-dependent font shaping')
@click.option('-ll', '--max-length', type=click.INT, default=None,
              help="Discard lines above length (in Unicode codepoints).")
@click.option('--strip/--no-strip', help="Remove whitespace from start and end "
              "of lines.")
@click.option('-D', '--disable-degradation', is_flag=True, help='Dont degrade '
              'output lines.')
@click.option('-a', '--alpha', type=click.FLOAT, default=1.5,
              help="Mean of folded normal distribution for sampling foreground pixel flip probability")
@click.option('-b', '--beta', type=click.FLOAT, default=1.5,
              help="Mean of folded normal distribution for sampling background pixel flip probability")
@click.option('-d', '--distort', type=click.FLOAT, default=1.0,
              help='Mean of folded normal distribution to take distortion values from')
@click.option('-ds', '--distortion-sigma', type=click.FLOAT, default=20.0,
              help='Mean of folded normal distribution to take standard deviations for the '
              'Gaussian kernel from')
@click.option('--legacy/--no-legacy', default=False,
              help='Use ocropy-style degradations')
@click.option('-o', '--output', type=click.Path(), default='training_data',
              help='Output directory')
@click.argument('text', nargs=-1, type=click.Path(exists=True))
def line_generator(ctx, font, maxlines, encoding, normalization, renormalize,
                   reorder, font_size, font_weight, language, max_length, strip,
                   disable_degradation, alpha, beta, distort, distortion_sigma,
                   legacy, output, text):
    """
    Generates artificial text line training data.
    """
    import errno
    import numpy as np

    from kraken import linegen
    from kraken.lib.util import make_printable

    lines: Set[str] = set()
    if not text:
        return
    with KrakenProgressBar() as progress:
        read_task = progress.add_task('Reading texts', total=len(text), visible=True if not ctx.meta['verbose'] else False)
        for t in text:
            with click.open_file(t, encoding=encoding) as fp:
                logger.info('Reading {}'.format(t))
                for line in fp:
                    lines.add(line.rstrip('\r\n'))
            progress.update(read_task, advance=1)

    if normalization:
        lines = set([unicodedata.normalize(normalization, line) for line in lines])
    if strip:
        lines = set([line.strip() for line in lines])
    if max_length:
        lines = set([line for line in lines if len(line) < max_length])
    logger.info('Read {} lines'.format(len(lines)))
    message('Read {} unique lines'.format(len(lines)))
    if maxlines and maxlines < len(lines):
        message('Sampling {} lines\t'.format(maxlines), nl=False)
        llist = list(lines)
        lines = set(llist[idx] for idx in np.random.randint(0, len(llist), maxlines))
        message('\u2713', fg='green')
    try:
        os.makedirs(output)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # calculate the alphabet and print it for verification purposes
    alphabet: Set[str] = set()
    for line in lines:
        alphabet.update(line)
    chars = []
    combining = []
    for char in sorted(alphabet):
        k = make_printable(char)
        if k != char:
            combining.append(k)
        else:
            chars.append(k)
    message('Σ (len: {})'.format(len(alphabet)))
    message('Symbols: {}'.format(''.join(chars)))
    if combining:
        message('Combining Characters: {}'.format(', '.join(combining)))
    lg = linegen.LineGenerator(font, font_size, font_weight, language)
    with KrakenProgressBar() as progress:
        gen_task = progress.add_task('Writing images', total=len(lines), visible=True if not ctx.meta['verbose'] else False)
        for idx, line in enumerate(lines):
            logger.info(line)
            try:
                if renormalize:
                    im = lg.render_line(unicodedata.normalize(renormalize, line))
                else:
                    im = lg.render_line(line)
            except KrakenCairoSurfaceException as e:
                logger.info('{}: {} {}'.format(e.message, e.width, e.height))
                continue
            if not disable_degradation and not legacy:
                im = linegen.degrade_line(im, alpha=alpha, beta=beta)
                im = linegen.distort_line(im, abs(np.random.normal(distort)), abs(np.random.normal(distortion_sigma)))
            elif legacy:
                im = linegen.ocropy_degrade(im)
            im.save('{}/{:06d}.png'.format(output, idx))
            with open('{}/{:06d}.gt.txt'.format(output, idx), 'wb') as fp:
                if reorder:
                    fp.write(get_display(line).encode('utf-8'))
                else:
                    fp.write(line.encode('utf-8'))
            progress.update(gen_task, advance=1)

@cli.command('publish')
@click.pass_context
@click.option('-i', '--metadata', show_default=True,
              type=click.File(mode='r', lazy=True), help='Metadata for the '
              'model. Will be prompted from the user if not given')
@click.option('-a', '--access-token', prompt=True, help='Zenodo access token')
@click.argument('model', nargs=1, type=click.Path(exists=False, readable=True, dir_okay=False))
def publish(ctx, metadata, access_token, model):
    """
    Publishes a model on the zenodo model repository.
    """
    import json
    import pkg_resources

    from functools import partial
    from jsonschema import validate
    from jsonschema.exceptions import ValidationError

    from kraken import repo
    from kraken.lib import models

    with pkg_resources.resource_stream(__name__, 'metadata.schema.json') as fp:
        schema = json.load(fp)

    nn = models.load_any(model)

    if not metadata:
        author = click.prompt('author')
        affiliation = click.prompt('affiliation')
        summary = click.prompt('summary')
        description = click.edit('Write long form description (training data, transcription standards) of the model here')
        accuracy_default = None
        # take last accuracy measurement in model metadata
        if 'accuracy' in nn.nn.user_metadata and nn.nn.user_metadata['accuracy']:
            accuracy_default = nn.nn.user_metadata['accuracy'][-1][1] * 100
        accuracy = click.prompt('accuracy on test set', type=float, default=accuracy_default)
        script = [
            click.prompt(
                'script',
                type=click.Choice(
                    sorted(
                        schema['properties']['script']['items']['enum'])),
                show_choices=True)]
        license = click.prompt(
            'license',
            type=click.Choice(
                sorted(
                    schema['properties']['license']['enum'])),
            show_choices=True)
        metadata = {
            'authors': [{'name': author, 'affiliation': affiliation}],
            'summary': summary,
            'description': description,
            'accuracy': accuracy,
            'license': license,
            'script': script,
            'name': os.path.basename(model),
            'graphemes': ['a']
        }
        while True:
            try:
                validate(metadata, schema)
            except ValidationError as e:
                message(e.message)
                metadata[e.path[-1]] = click.prompt(e.path[-1], type=float if e.schema['type'] == 'number' else str)
                continue
            break

    else:
        metadata = json.load(metadata)
        validate(metadata, schema)
    metadata['graphemes'] = [char for char in ''.join(nn.codec.c2l.keys())]
    oid = repo.publish_model(model, metadata, access_token, partial(message, '.', nl=False))
    message('\nmodel PID: {}'.format(oid))


@cli.command('compile')
@click.pass_context
@click.option('-o', '--output', show_default=True, type=click.Path(), default='model', help='Output model file')
@click.option('--workers', show_default=True, default=1, help='Number of parallel workers for text line extraction.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page']), default='xml', show_default=True,
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information.')
@click.option('--random-split', type=float, nargs=3, default=None, show_default=True,
              help='Creates a fixed random split of the input data with the '
              'proportions (train, validation, test). Overrides the save split option.')
@click.option('--force-type', type=click.Choice(['bbox', 'baseline']), default=None, show_default=True,
              help='Forces the dataset type to a specific value. Can be used to '
                   '"convert" a line strip-type collection to a baseline-style '
                   'dataset, e.g. to disable centerline normalization.')
@click.option('--save-splits/--ignore-splits', show_default=True, default=True,
              help='Whether to serialize explicit splits contained in XML '
                   'files. Is ignored in `path` mode.')
@click.option('--recordbatch-size', show_default=True, default=100,
              help='Minimum number of records per RecordBatch written to the '
                   'output file. Larger batches require more transient memory '
                   'but slightly improve reading performance.')
@click.argument('ground_truth', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def compile(ctx, output, workers, format_type, random_split, force_type, save_splits, recordbatch_size, ground_truth):
    """
    Precompiles a binary dataset from a collection of XML files.
    """
    if not ground_truth:
        raise click.UsageError('No training data was provided to the compile command. Use the `ground_truth` argument.')

    from kraken.lib import arrow_dataset

    force_type = {'bbox': 'kraken_recognition_bbox',
                  'baseline': 'kraken_recognition_baseline',
                  None: None}[force_type]

    with KrakenProgressBar() as progress:
        extract_task = progress.add_task('Extracting lines', total=0, start=False, visible=True if not ctx.meta['verbose'] else False)

        arrow_dataset.build_binary_dataset(ground_truth,
                                           output,
                                           format_type,
                                           workers,
                                           save_splits,
                                           random_split,
                                           force_type,
                                           recordbatch_size,
                                           lambda advance, total: progress.update(extract_task, total=total, advance=advance))

    message(f'Output file written to {output}')


@cli.command('segtest')
@click.pass_context
@click.option('-m', '--model', show_default=True, type=click.Path(exists=True, readable=True),
              multiple=False, help='Model(s) to evaluate')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--workers', show_default=True, default=1, help='Number of OpenMP threads when running on CPU.')
@click.option('--force-binarization/--no-binarization', show_default=True,
              default=False, help='Forces input images to be binary, otherwise '
              'the appropriate color format will be auto-determined through the '
              'network specification. Will be ignored in `path` mode.')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page']), default='xml',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information.')
@click.option('--suppress-regions/--no-suppress-regions', show_default=True,
              default=False, help='Disables region segmentation training.')
@click.option('--suppress-baselines/--no-suppress-baselines', show_default=True,
              default=False, help='Disables baseline segmentation training.')
@click.option('-vr', '--valid-regions', show_default=True, default=None, multiple=True,
              help='Valid region types in training data. May be used multiple times.')
@click.option('-vb', '--valid-baselines', show_default=True, default=None, multiple=True,
              help='Valid baseline types in training data. May be used multiple times.')
@click.option('-mr',
              '--merge-regions',
              show_default=True,
              default=None,
              help='Region merge mapping. One or more mappings of the form `$target:$src` where $src is merged into $target.',
              multiple=True,
              callback=_validate_merging)
@click.option('-mb',
              '--merge-baselines',
              show_default=True,
              default=None,
              help='Baseline type merge mapping. Same syntax as `--merge-regions`',
              multiple=True,
              callback=_validate_merging)
@click.option('-br', '--bounding-regions', show_default=True, default=None, multiple=True,
              help='Regions treated as boundaries for polygonization purposes. May be used multiple times.')
@click.option("--threshold", type=click.FloatRange(.01, .99), default=.3, show_default=True,
              help="Threshold for heatmap binarization. Training threshold is .3, prediction is .5")
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def segtest(
    ctx, model, evaluation_files, device, workers, threshold,
    force_binarization, format_type, test_set,
    suppress_regions, suppress_baselines,
    valid_regions, valid_baselines,
    merge_regions, merge_baselines,
    bounding_regions
):
    """
    Evaluate on a test set.
    """
    if not model:
        raise click.UsageError('No model to evaluate given.')

    import numpy as np
    from torch.utils.data import DataLoader
    import torch
    import torch.nn.functional as F

    import tabulate
    from kraken.lib.train import BaselineSet, ImageInputTransforms, Subset
    from kraken.lib.vgsl import TorchVGSLModel

    logger.info('Building test set from {} line images'.format(len(test_set) + len(evaluation_files)))

    message('Loading model {}\t'.format(model), nl=False)
    nn = TorchVGSLModel.load_model(model)
    message('\u2713', fg='green')

    test_set = list(test_set)
    if evaluation_files:
        test_set.extend(evaluation_files)

    if len(test_set) == 0:
        raise click.UsageError(
            'No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.'
        )
    # set number of OpenMP threads
    # next(iter(nn.values())).nn.set_num_threads(1)

    _batch, _channels, _height, _width = nn.input
    transforms = ImageInputTransforms(
        _batch,
        _height, _width, _channels, 0,
        valid_norm=False, force_binarization=force_binarization
    )
    if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
        logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
        torch.multiprocessing.set_sharing_strategy('file_system')

    if not valid_regions:
        valid_regions = None
    if not valid_baselines:
        valid_baselines = None

    if suppress_regions:
        valid_regions = []
        merge_regions = None
    if suppress_baselines:
        valid_baselines = []
        merge_baselines = None

    test_set = BaselineSet(
        test_set,
        line_width=nn.user_metadata["hyper_params"]["line_width"],
        im_transforms=transforms,
        mode=format_type,
        augmentation=False,
        valid_baselines=valid_baselines,
        merge_baselines=merge_baselines,
        valid_regions=valid_regions,
        merge_regions=merge_regions
    )

    # test_set.num_classes = len(nn.user_metadata["class_mapping"]["baselines"])
    #test_set.class_mapping = len(nn.user_metadata["class_mapping"]["regions"])

    test_set.class_mapping = nn.user_metadata["class_mapping"]
    test_set.num_classes = sum([
        len(classDict)
        for classDict in test_set.class_mapping.values()
    ])

    baselines_diff = set(test_set.class_stats["baselines"].keys()).difference(
        test_set.class_mapping["baselines"].keys()
    )
    regions_diff = set(test_set.class_stats["regions"].keys()).difference(
        test_set.class_mapping["regions"].keys()
    )
    if baselines_diff:
        message("Unknown baseline type by the model in the test set: %s" % ", ".join(
            sorted(list(baselines_diff))
        ))

    if regions_diff:
        message("Unknown regions type by the model in the test set: %s" % ", ".join(
            sorted(list(regions_diff))
        ))

    if device == 'cpu':
        device = None
    elif device.startswith('cuda'):
        device = [int(device.split(':')[-1])]

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    message('Evaluating {}'.format(model))
    logger.info('Evaluating {}'.format(model))
    ds_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=workers,
        pin_memory=True
    )

    nn.to(device)
    nn.eval()
    nn.set_num_threads(1)
    pages = []

    lines_idx = list(test_set.class_mapping["baselines"].values())
    regions_idx = list(test_set.class_mapping["regions"].values())

    def _compute_iou_overlap(classes_idx: List[int], y: torch.BoolTensor, pred: torch.BoolTensor):
        return {
            "intersections": torch.stack([
                (
                        (
                            y[line_idx].unsqueeze(0).repeat(len(classes_idx) - 1, 1)
                        ) & (
                            pred[[subline_idx for subline_idx in classes_idx if subline_idx != line_idx]]
                        )
                ).sum(dim=1, dtype=torch.double)
                for line_idx in classes_idx
                ],
                dim=0
            ),
            "unions": torch.stack([
                (
                        (
                            y[line_idx].squeeze(0).repeat(len(classes_idx) - 1, 1)
                        ) | (
                            pred[[subline_idx for subline_idx in classes_idx if subline_idx != line_idx]]
                        )
                ).sum(dim=1, dtype=torch.double)
                for line_idx in classes_idx
                ],
                dim=0
            )
        }

    with KrakenProgressBar() as progress:
        batches = len(ds_loader)
        pred_task = progress.add_task('Evaluating', total=batches, visible=True if not ctx.meta['verbose'] else False)
        for batch in ds_loader:
            x, y = batch['image'], batch['target']
            try:
                # [BatchSize, NumClasses, Width, Height ?]
                pred, _ = nn.nn(x)
                # scale target to output size
                y = F.interpolate(y, size=(pred.size(2), pred.size(3))).squeeze(0).bool()
                pred = pred.squeeze() > threshold
                pred = pred.view(pred.size(0), -1)
                y = y.view(y.size(0), -1)
                pages.append({
                    'intersections': (y & pred).sum(dim=1, dtype=torch.double),
                    'unions': (y | pred).sum(dim=1, dtype=torch.double),
                    'corrects': torch.eq(y, pred).sum(dim=1, dtype=torch.double),
                    'cls_cnt': y.sum(dim=1, dtype=torch.double),
                    'all_n': torch.tensor(y.size(1), dtype=torch.double, device=device)
                })
                if lines_idx:
                    y_baselines = y[lines_idx].sum(dim=0, dtype=torch.bool)
                    pred_baselines = pred[lines_idx].sum(dim=0, dtype=torch.bool)
                    pages[-1]["baselines"] = {
                        'intersections': (y_baselines & pred_baselines).sum(dim=0, dtype=torch.double),
                        'unions': (y_baselines | pred_baselines).sum(dim=0, dtype=torch.double),
                    }
                    # Confusion
                    if len(lines_idx) > 1:
                        pages[-1]["baselines"]["confusion"] = _compute_iou_overlap(lines_idx, y=y, pred=pred)
                if regions_idx:
                    y_regions_idx = y[regions_idx].sum(dim=0, dtype=torch.bool)
                    pred_regions_idx = pred[regions_idx].sum(dim=0, dtype=torch.bool)
                    pages[-1]["regions"] = {
                        'intersections': (y_regions_idx & pred_regions_idx).sum(dim=0, dtype=torch.double),
                        'unions': (y_regions_idx | pred_regions_idx).sum(dim=0, dtype=torch.double),
                    }
                    if len(regions_idx) > 1:
                        pages[-1]["regions"]["confusion"] = _compute_iou_overlap(regions_idx, y=y, pred=pred)

            except FileNotFoundError as e:
                batches -= 1
                progress.update(pred_task, total=batches)
                logger.warning('{} {}. Skipping.'.format(e.strerror, e.filename))
            except KrakenInputException as e:
                batches -= 1
                progress.update(pred_task, total=batches)
                logger.warning(str(e))
            progress.update(pred_task, advance=1)

    # Accuracy / pixel
    corrects = torch.stack([x['corrects'] for x in pages], -1).sum(dim=-1)
    all_n = torch.stack([x['all_n'] for x in pages]).sum()  # Number of pixel for all pages

    # Pixel Accuracy / Class
    class_pixel_accuracy = corrects / all_n
    # Global Mean Accuracy
    #    Macro, not per page. Which means small pages are less counted ?
    mean_accuracy = torch.mean(class_pixel_accuracy)

    intersections = torch.stack([x['intersections'] for x in pages], -1).sum(dim=-1)
    unions = torch.stack([x['unions'] for x in pages], -1).sum(dim=-1)
    smooth = torch.finfo(torch.float).eps
    class_iu = (intersections + smooth) / (unions + smooth)
    mean_iu = torch.mean(class_iu)

    cls_cnt = torch.stack([x['cls_cnt'] for x in pages]).sum()
    freq_iu = torch.sum(cls_cnt / cls_cnt.sum() * class_iu.sum())

    message(f"Mean accuracy: {mean_accuracy.item():.3f}")
    message(f"Mean Intersection / Union: {mean_iu.item():.3f}")
    message(f"Freq Intersection / Union: {freq_iu.item():.3f}")

    class_iu = class_iu.tolist()
    class_pixel_accuracy = class_pixel_accuracy.tolist()
    out = []
    for (cat, class_name), iu, pix_acc in zip(
        [(cat, key) for (cat, subcategory) in test_set.class_mapping.items() for key in subcategory],
        class_iu,
        class_pixel_accuracy
    ):
        # Only do those with supports
        out.append({"Category": cat, "Class name": class_name,
                    "Pixel Accuracy": f"{pix_acc:.3f}",
                    "Intersection / Union": f"{iu:.3f}",
                    "Support": test_set.class_stats[cat][class_name] if cat != "aux" else "N/A"})

    # Region accuracies
    inv_class_names = dict([
        (val, key) for (cat, subcategory) in test_set.class_mapping.items() for key, val in subcategory.items()
    ])

    def _compute_overlap_iou(
        classes: List[int],
        category: str = "baselines",
        smooth: torch.FloatType = torch.finfo(torch.float).eps
    ):
        # Compute Overlap between classes
        ov_intersections = torch.stack(
            [x[category]["confusion"]['intersections'] for x in pages],
            dim=-1
        ).sum(dim=-1)
        ov_unions = torch.stack(
            [x[category]["confusion"]['unions'] for x in pages],
            dim=-1
        ).sum(dim=-1)
        ov_iu = ((ov_intersections + smooth) / (ov_unions + smooth)).tolist()
        table = []
        for row_idx, row_class_idx in enumerate(classes):
            table.append({
                "Source": inv_class_names[row_class_idx],
                **{
                    inv_class_names[secondary_line_idx]: (
                        f"{ov_iu[row_idx][pred_idx]:.3f}" if ov_iu[row_idx][pred_idx] != 1 else ""
                    )
                    for pred_idx, secondary_line_idx in enumerate([
                        _x
                        for _x in classes
                        if _x != row_class_idx
                    ])

                }
            })
        return table

    if lines_idx:
        line_intersections = torch.stack([x["baselines"]['intersections'] for x in pages]).sum()
        line_unions = torch.stack([x["baselines"]['unions'] for x in pages]).sum()
        line_iu = (line_intersections + smooth) / (line_unions + smooth)
        message(f"Class-independent baselines Intersection / Union: {line_iu.item():.3f}")

        if len(lines_idx) > 1:
            # Compute Overlap between classes
            message(
                "\n Line Intersection / Union Overlap " + tabulate.tabulate(
                    _compute_overlap_iou(lines_idx, category="baselines"),
                    headers="keys",
                    tablefmt="markdown"
                )
            )

    # Region accuracies
    if regions_idx:
        region_intersections = torch.stack([x["regions"]['intersections'] for x in pages]).sum()
        region_unions = torch.stack([x["regions"]['unions'] for x in pages]).sum()
        region_iu = (region_intersections + smooth) / (region_unions + smooth)
        message(f"Class-independent regions Intersection / Union: {region_iu.item():.3f}")
        if len(regions_idx):
            message(
                "\n Region Intersection / Union Overlap " + tabulate.tabulate(
                    _compute_overlap_iou(regions_idx, category="regions"),
                    headers="keys",
                    tablefmt="markdown"
                )
            )

    message("\nPer category metrics\n" + tabulate.tabulate(out, headers="keys", tablefmt="markdown"))


if __name__ == '__main__':
    cli()
