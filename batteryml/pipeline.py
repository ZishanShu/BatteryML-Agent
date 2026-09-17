# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

from __future__ import annotations

import os
import torch
import pickle
import random
import inspect
import shutil
import hashlib
import numpy as np

from pathlib import Path
from datetime import datetime

from batteryml.task import Task
from batteryml.data import DataBundle
from batteryml.builders import MODELS
from batteryml.utils import import_config
from batteryml.models.base import BaseModel
from batteryml import training_checkpoint as _resume_schema_module


class Pipeline:
    def __init__(self, config_path: Path | str, workspace: Path | str):

        self.config_path = config_path
        self.config = load_config(config_path, workspace)

    def train(self,
              seed: int = 0,
              epochs: int | None = None,
              device: torch.device | str = 'cpu',
              ckpt_to_resume: str | None = None,
              skip_if_executed: bool = True,
              dataset: DataBundle | None = None):
        set_seed(seed)

        if skip_if_executed and (
            self.config['workspace'] is not None
            and any(Path(self.config['workspace'])
                    .glob(f'*seed_{seed}_*.ckpt'))
        ):
            # TODO: maybe we should add a logging util for handling infos
            print(f'Skip training for {self.config["workspace"]} '
                  'as the checkpoint already exists.')
            return

        # Prepare dataset
        if dataset is None:
            dataset, raw_data = build_dataset(self.config, device)
            self.raw_data = raw_data
        # Number of epochs override
        if epochs is not None:
            original_epochs = self.config['model'].get('epochs')
            self.config['model']['epochs'] = epochs

        # A *resumable* checkpoint continues training; a weights-only one is
        # the pre-existing warm start and keeps its old path untouched.
        resume_state = _load_resume_state(ckpt_to_resume)

        # Prepare model
        model = self._prepare_model(
            None if resume_state is not None else ckpt_to_resume, device)
        if resume_state is not None:
            _load_model_state(model, resume_state.get('model_state'))
            _restore_rng_state(resume_state.get('rng_state'))
        ts = timestamp()

        # Make a copy of the config in the workspace
        if model.workspace is not None:
            shutil.copyfile(
                self.config_path,
                model.workspace / f'config_{ts}.yaml')

        # Passed only to a fit() that accepts them, so a model with its own
        # training loop (BatLiNet) is called exactly as before.
        extra = _supported_fit_kwargs(
            model,
            resume_state=resume_state,
            on_epoch_end=_resume_checkpoint_writer(self, model, seed, ts),
        )
        model.fit(dataset, timestamp=ts, **extra)

        # Restore the origianl epochs in config
        if epochs is not None:
            self.config['model']['epochs'] = original_epochs


        return model, dataset

    def evaluate(self,
                 seed: int = 0,
                 device: torch.device | str = 'cpu',
                 metric: list | str = 'RMSE',
                 model: BaseModel | None = None,
                 dataset: DataBundle | None = None,
                 ckpt_to_resume: str | None = None,
                 skip_if_executed: bool = True):
        set_seed(seed)

        if skip_if_executed and (
            self.config['workspace'] is not None
            and any(Path(self.config['workspace'])
                    .glob(f'predictions_seed_{seed}_*.pkl'))
        ):
            print(f'Skip evaluation for {self.config["workspace"]} '
                  'as the prediction exists.')
            return

        if dataset is None:
            dataset, raw_data = build_dataset(self.config, device)
            self.raw_data = raw_data
        if model is None:
            model = self._prepare_model(ckpt_to_resume, device)

        prediction = model.predict(dataset)

        if isinstance(metric, str):
            metric = [metric]

        scores = {
            m: dataset.evaluate(prediction, m) for m in metric
        }
        print(scores)
        ts = timestamp()

        if self.config['workspace'] is not None:
            obj = {
                'prediction': prediction,
                'scores': scores,
                'data': dataset.to('cpu'),
                'seed': seed,
            }
            filename = f'predictions_seed_{seed}_{ts}.pkl'
            with open(Path(self.config['workspace']) / filename, 'wb') as f:
                pickle.dump(obj, f)

    def _prepare_model(self,
                       ckpt_to_resume: str | None = None,
                       device: torch.device | None = 'cpu'
                       ) -> BaseModel:
        model = MODELS.build(self.config['model'])
        if model.workspace is None:
            model.workspace = self.config['workspace']
        if ckpt_to_resume is not None:
            model.load_checkpoint(ckpt_to_resume)

        if torch.__version__ >= '2' and isinstance(model, torch.nn.Module):
            model = torch.compile(model)

        model = model.to(device)
        return model


CONFIG_FIELDS = [
    'model',
    'train_test_split',
    'feature',
    'label',
    'feature_transformation',
    'label_transformation'
]


def load_config(config_path: str,
                workspace: str | None,
                config_fields: list | None = None
                ) -> dict:
    config_path = Path(config_path)
    config_fields = config_fields or CONFIG_FIELDS
    configs = import_config(config_path, config_fields)

    # Determine the workspace
    if configs['model'].get('workspace') is not None:
        workspace = Path(configs['model'].get('workspace'))
    elif workspace is not None:
        if workspace.strip().lower() == 'none':
            workspace = None
        else:
            workspace = Path(workspace)
    else:
        # workspace = Path.cwd() / 'workspaces' / config_path.stem
        workspace = Path.cwd() / 'workspaces' / config_path.relative_to('configs').with_suffix('')  

    if workspace is not None and workspace.exists():
        assert workspace.is_dir(), workspace

    if workspace is not None and not workspace.exists():
        os.makedirs(workspace)

    configs['workspace'] = workspace
    print(workspace)
    return configs


def build_dataset(configs: dict,
                  device: str,
                  config_fields: list | None = None):
    strings = []
    config_fields = config_fields or CONFIG_FIELDS[1:]
    for field in config_fields:
        strings.append(recursive_dump_string(configs[field]))
    filename = hash_string('+'.join(strings))
    cache_dir = Path('cache')
    if not cache_dir.exists():
        cache_dir.mkdir()
    cache_file = Path(cache_dir / f'battery_cache_{filename}.pkl')

    if cache_file.exists():
        print(f'Load datasets from cache {str(cache_file)}.')
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            dataset = data['dataset']
            raw_data = data['raw_data']
    else:
        task = Task(
            label_annotator=configs['label'],
            feature_extractor=configs['feature'],
            train_test_splitter=configs['train_test_split'],
            feature_transformation=configs['feature_transformation'],
            label_transformation=configs['label_transformation'])

        dataset = task.build()
        train_cells, test_cells = task.get_raw_data()
        data = {'dataset':dataset,
                'raw_data':{
                    'train_cells': train_cells,
                    'test_cells': test_cells,
                }}
        raw_data = data['raw_data']
        # store cache
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    return dataset.to(device), raw_data


def set_seed(seed: int):
    print(f'Seed is set to {seed}.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def recursive_dump_string(data):
    if isinstance(data, list):
        return '_'.join([recursive_dump_string(x) for x in data])
    if isinstance(data, dict):
        return '_'.join([
            recursive_dump_string(data[key])
            for key in sorted(data.keys())
        ])
    return str(data)


def hash_string(string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(string.encode('utf-8'))
    hash_value = sha256_hash.hexdigest()
    truncated_hash = hash_value[:32]
    return truncated_hash


def timestamp(marker: bool = False):
    template = '%Y-%m-%d %H:%M:%S' if marker else '%Y%m%d%H%M%S'
    return datetime.now().strftime(template)


# --- resumable checkpoints -------------------------------------------------
#
# The schema lives in batteryml.training_checkpoint — the training layer,
# beside the code that writes and reads these files. It used to live in the web
# layer, which meant core BatteryML had to import the web application to resume
# its own training; that dependency is gone. The web layer re-exports the same
# module, so there is still one schema, one serializer and one validator.
# Recorded in docs/UPSTREAM_PATCHES.md (UP-02).


def _unwrap(model):
    """The real module beneath torch.compile / DDP wrappers.

    ``torch.compile`` is applied in ``_prepare_model``, and its state_dict
    keys carry an ``_orig_mod.`` prefix. Saving that would make a checkpoint
    fail to load into an uncompiled model — the same interoperability trap a
    ``module.`` prefix is for DDP — so both are peeled off before saving and
    before loading.
    """
    for attribute in ('_orig_mod', 'module'):
        inner = getattr(model, attribute, None)
        if inner is not None:
            return _unwrap(inner)
    return model


def _load_resume_state(path):
    """The resume payload at ``path``, or None if it is not a resumable one."""
    if path is None:
        return None
    schema = _resume_schema_module
    verdict = schema.classify_checkpoint(path)
    if verdict['kind'] != schema.KIND_RESUMABLE:
        # Weights-only, corrupt or from a newer schema: not a continuation.
        # The caller's existing warm-start path handles the first case.
        return None
    return schema.load_raw(path)


def _load_model_state(model, state):
    if state:
        _unwrap(model).load_state_dict(state)


def _restore_rng_state(state):
    if not state:
        return
    for warning in _resume_schema_module.restore_rng_state(state):
        print(f'[resume] {warning}', flush=True)


def _supported_fit_kwargs(model, **candidates):
    """Only the kwargs this model's ``fit`` actually accepts.

    A model that defines its own training loop keeps its own signature, so
    nothing here changes how it is called.
    """
    try:
        parameters = inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        return {}
    return {
        name: value for name, value in candidates.items()
        if name in parameters and value is not None
    }


def _resume_checkpoint_writer(self, model, seed, ts):
    """A per-epoch callback that writes a resumable checkpoint, or None.

    Cadence follows the model's own ``checkpoint_freq``, so this adds no
    extra writes beyond the ones training already performed — it makes those
    writes carry optimizer, epoch and RNG state as well as the weights.
    """
    schema = _resume_schema_module
    workspace = getattr(model, 'workspace', None)
    frequency = getattr(model, 'checkpoint_freq', None)
    if workspace is None or not frequency:
        return None

    def on_epoch_end(epoch, context):
        if (epoch + 1) % frequency != 0:
            return
        inner = _unwrap(context.get('model', model))
        optimizer = context.get('optimizer')
        payload = schema.build_checkpoint(
            model_state=inner.state_dict(),
            optimizer_state=optimizer.state_dict() if optimizer is not None else None,
            epoch=epoch + 1,
            target_total_epochs=context.get('total_epochs'),
            # What this layer can state truthfully. The dataset and task are
            # the web layer's vocabulary, not the pipeline's, so they are not
            # invented here; validate_resume reports any field it could not
            # compare rather than assuming a match.
            training_identity={
                'seed': str(seed),
                # The registered class, recorded under its own name. It is
                # deliberately NOT called model_key: that is the web layer's
                # vocabulary ("cnn"), and this is the config's
                # ("CNNRULPredictor"). Storing one under the other's name
                # would make a legitimate resume look like a mismatch.
                'model_class': str(self.config['model'].get('name', '')),
                'config_fingerprint': schema.config_fingerprint(self.config),
            },
        )
        schema.save_checkpoint(
            payload,
            Path(workspace) / f'{ts}_seed_{seed}_epoch_{epoch + 1}.resume.ckpt',
        )

    return on_epoch_end
