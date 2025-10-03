# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains functionality shared across all PyRIID models."""
import json
import os
from pathlib import Path
import uuid
from abc import abstractmethod
from enum import Enum

import numpy as np
import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.losses import cosine_similarity, categorical_crossentropy

import riid
from riid import SampleSet, SpectraState
from riid.data.labeling import label_to_index_element
from riid.losses import mish
from riid.losses.sparsemax import sparsemax
from riid.metrics import multi_f1, single_f1, APE_score
from sklearn.metrics import accuracy_score, f1_score

from riid.models.functions import (
    zscore, add_channel, extract_patches, add_sinusoidal_pos, 
    make_positions, take_cls_token_fn
)
from riid.models.layers import (
    ReduceSumLayer, ReduceMaxLayer, DivideLayer, ExpandDimsLayer,
    ClipByValueLayer, PoissonLogProbabilityLayer, SeedLayer, L1NormLayer,
    ClassToken, GradientReversalLayer
)

get_custom_objects().update({
    "multi_f1": multi_f1,
    "single_f1": single_f1,
    "mish": mish,
    "sparsemax": sparsemax,
    "APE_score": APE_score,
    # Custom functions
    "zscore": zscore,
    "add_channel": add_channel,
    "extract_patches": extract_patches,
    "add_sinusoidal_pos": add_sinusoidal_pos,
    "make_positions": make_positions,
    "take_cls_token_fn": take_cls_token_fn,
    # Custom layers
    "ReduceSumLayer": ReduceSumLayer,
    "ReduceMaxLayer": ReduceMaxLayer,
    "DivideLayer": DivideLayer,
    "ExpandDimsLayer": ExpandDimsLayer,
    "ClipByValueLayer": ClipByValueLayer,
    "PoissonLogProbabilityLayer": PoissonLogProbabilityLayer,
    "SeedLayer": SeedLayer,
    "L1NormLayer": L1NormLayer,
    "ClassToken": ClassToken,
    "GradientReversalLayer": GradientReversalLayer,
})


class ModelInput(int, Enum):
    """Enumerates the potential input sources for a model."""
    GrossSpectrum = 0
    BackgroundSpectrum = 1
    ForegroundSpectrum = 2


class PyRIIDModel:
    """Base class for PyRIID models."""

    def __init__(self, *args, **kwargs):
        self._info = {}
        self._temp_file_path = "temp_model.json"
        self._custom_objects = {}
        self._initialize_info()

    @property
    def seeds(self):
        return self._info["seeds"]

    @seeds.setter
    def seeds(self, value):
        self._info["seeds"] = value

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    @property
    def target_level(self):
        return self._info["target_level"]

    @target_level.setter
    def target_level(self, value):
        if value in SampleSet.SOURCES_MULTI_INDEX_NAMES:
            self._info["target_level"] = value
        else:
            msg = (
                f"Target level '{value}' is invalid.  "
                f"Acceptable levels: {SampleSet.SOURCES_MULTI_INDEX_NAMES}"
            )
            raise ValueError(msg)

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, value: Model):
        self._model = value

    @property
    def model_id(self):
        return self._info["model_id"]

    @model_id.setter
    def model_id(self, value):
        self._info["model_id"] = value

    @property
    def model_inputs(self):
        return self._info["model_inputs"]

    @model_inputs.setter
    def model_inputs(self, value):
        self._info["model_inputs"] = value

    @property
    def model_outputs(self):
        return self._info["model_outputs"]

    @model_outputs.setter
    def model_outputs(self, value):
        self._info["model_outputs"] = value

    def get_model_outputs_as_label_tuples(self):
        return [
            label_to_index_element(v, self.target_level) for v in self.model_outputs
        ]

    def _get_model_dict(self) -> dict:
        model_json = self.model.to_json()
        model_dict = json.loads(model_json)
        model_weights = self.model.get_weights()
        model_dict = {
            "info": self._info,
            "model": model_dict,
            "weights": model_weights,
            "history": self.history
        }
        return model_dict

    def _get_model_str(self) -> str:
        model_dict = self._get_model_dict()
        model_str = json.dumps(model_dict, indent=4, cls=PyRIIDModelJsonEncoder)
        return model_str

    def _initialize_info(self):
        init_info = {
            "model_id": str(uuid.uuid4()),
            "model_type": self.__class__.__name__,
            "normalization": SpectraState.Unknown,
            "pyriid_version": riid.__version__,
        }
        self._update_info(**init_info)

    def _update_info(self, **kwargs):
        self._info.update(kwargs)

    def _update_custom_objects(self, key, value):
        self._custom_objects.update({key: value})

    def load(self, model_path: str):
        """Load the model from a path.

        Args:
            model_path: path from which to load the model.
        """
        if not os.path.exists(model_path):
            raise ValueError("Model file does not exist.")

        with open(model_path) as fin:
            model = json.load(fin)

        model_str = json.dumps(model["model"])
        self.model = tf.keras.models.model_from_json(model_str, custom_objects=self._custom_objects)
        self.model.set_weights([np.array(x) for x in model["weights"]])
        self.info = model["info"]
        self.history = model.get("history", None)

    def save(self, model_path: str, overwrite=False):
        """Save the model to a path.

        Args:
            model_path: path at which to save the model.
            overwrite: whether to overwrite an existing file if it already exists.

        Raises:
            `ValueError` when the given path already exists
        """
        if os.path.exists(model_path) and not overwrite:
            raise ValueError("Model file already exists.")

        model_str = self._get_model_str()
        with open(model_path, "w") as fout:
            fout.write(model_str)

    def to_onnx(self, model_path, **tf2onnx_kwargs: dict):
        """Convert the model to an ONNX model.

        Args:
            model_path: path at which to save the model
            tf2onnx_kwargs: additional kwargs to pass to the conversion
        """
        model_path = Path(model_path)
        if not str(model_path).endswith(riid.ONNX_MODEL_FILE_EXTENSION):
            raise ValueError(f"ONNX file path must end with {riid.ONNX_MODEL_FILE_EXTENSION}")
        if model_path.exists():
            raise ValueError("Model file already exists.")

        tf2onnx.convert.from_keras(
            self.model,
            input_signature=[
                tf.TensorSpec(
                    shape=input_tensor.shape,
                    dtype=input_tensor.dtype,
                    name=input_tensor.name
                )
                for input_tensor in self.model.inputs
            ],
            output_path=str(model_path),
            **tf2onnx_kwargs
        )

    def to_tflite(self, model_path, quantize: bool = False, prune: bool = False):
        """Convert the model to a TFLite model and optionally applying quantization or pruning.

        Note: requires export to SavedModel format first, then conversion to TFLite occurs.

        Args:
            model_path: file path at which to save the model
            quantize: whether to apply quantization
            prune: whether to apply pruning
        """
        model_path = Path(model_path)
        if not str(model_path).endswith(riid.TFLITE_MODEL_FILE_EXTENSION):
            raise ValueError(f"TFLite file path must end with {riid.TFLITE_MODEL_FILE_EXTENSION}")
        if model_path.exists():
            raise ValueError("Model file already exists.")

        optimizations = []
        if quantize:
            optimizations.append(tf.lite.Optimize.DEFAULT)
        if prune:
            optimizations.append(tf.lite.Optimize.EXPERIMENTAL_SPARSITY)

        saved_model_dir = model_path.stem
        self.model.export(saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = optimizations
        tflite_model = converter.convert()

        with open(model_path, "wb") as fout:
            fout.write(tflite_model)

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def calc_APE_score(self, ss: SampleSet, target_level="Isotope", batch_size: int = 1000):
        """Calculate the prediction APE score on ss"""
        self.predict(ss, batch_size=batch_size)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        ape = APE_score(y_true, y_pred).numpy()
        return ape

    def calc_cosine_similarity(self, ss: SampleSet, target_level="Isotope", batch_size: int = 1000):
        """Calculate the prediction cosine similarity score on ss"""
        self.predict(ss, batch_size=batch_size)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        cosine_sim = cosine_similarity(y_true, y_pred)
        cosine_score = -tf.reduce_mean(cosine_sim).numpy()
        return cosine_score

    def calc_loss(self, ss: SampleSet, target_level="Isotope", batch_size: int = 1000):
        """Calculate the loss on ss"""
        self.predict(ss, batch_size=batch_size)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        loss = self.model.loss(y_true, y_pred).numpy()
        return loss

    def calc_crossentropy(self, ss: SampleSet, target_level="Isotope", batch_size: int = 1000):
        """Calculate the categorical crossentropy loss on ss (without label smoothing)"""
        self.predict(ss, batch_size=batch_size)
        y_true = ss.sources.T.groupby(target_level, sort=False).sum().T.values
        y_pred = ss.prediction_probas.T.groupby(target_level, sort=False).sum().T.values
        crossentropy = categorical_crossentropy(y_true, y_pred)
        crossentropy_loss = tf.reduce_mean(crossentropy).numpy()
        return crossentropy_loss

    def calc_accuracy(self, ss: SampleSet, target_level="Isotope", batch_size: int = 1000):
        """Calculate the accuracy on ss"""
        self.predict(ss, batch_size=batch_size)
        labels = ss.get_labels()
        predictions = ss.get_predictions()
        accuracy = accuracy_score(labels, predictions)
        return accuracy

    def calc_f1_score(self, ss: SampleSet, target_level="Isotope", average="weighted", batch_size: int = 1000):
        """Calculate the f1 score on ss"""
        self.predict(ss, batch_size=batch_size)
        labels = ss.get_labels()
        predictions = ss.get_predictions()
        score = f1_score(labels, predictions, average=average)
        return score

class PyRIIDModelJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder for saving models.
    """
    def default(self, o):
        """Converts certain types to JSON-compatible types.
        """
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.float32):
            return o.astype(float)

        return super().default(o)
