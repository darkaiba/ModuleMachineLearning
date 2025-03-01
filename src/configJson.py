from src import Aprendizado, TypeFile
import os
from pathlib import Path

class EvaluateJson:

    def __init__(self):
        pass

    def evaluate(self, json_job):
        self._validate_field(json_job, "input_file", [True, False, None])
        self._validate_field(json_job, "learning", [Aprendizado.NETWORK, Aprendizado.SUPERVISED, Aprendizado.UNSUPERVISED, Aprendizado.SEMISUPERVISED])
        self._validate_nested_field(json_job, "reading", self.validation_reading)
        self._validate_nested_field(json_job, "mode", self.validation_mode)
        self._validate_optional_field(json_job, "pipeline", self.validation_pipeline, [None, dict])
        self._validate_optional_field(json_job, "validations", self.validation_validations, [None, dict])
        return json_job

    def _validate_field(self, json_job, field, allowed_values):
        if field not in json_job:
            raise ValueError(f"Seu Json de entrada não tem o campo '{field}'. JSON: {json_job}")
        value = json_job[field]
        if value not in allowed_values:
            raise ValueError(f"O campo '{field}' não é um campo válido, só é aceito valor {allowed_values}. Seu campo: {value}")

    def _validate_field_type(self, json_job, field, allowed_values):
        if field not in json_job:
            raise ValueError(f"Seu Json de entrada não tem o campo '{field}'. JSON: {json_job}")
        value = json_job[field]
        if type(value) not in allowed_values:
            raise ValueError(f"O campo '{field}' não é um campo válido, só é aceito valor {allowed_values}. Seu campo: {value}")

    def _validate_nested_field(self, json_job, field, validation_func):
        if field not in json_job:
            raise ValueError(f"Seu Json de entrada não tem o campo '{field}'. JSON: {json_job}")
        validation_func(json_job[field])

    def _validate_optional_field(self, json_job, field, validation_func, allowed_types):
        if field in json_job:
            value = json_job[field]

            if value is None:
                if not None in allowed_types:
                    raise ValueError(
                        f"O campo '{field}' não é um campo válido, só é aceito valor {allowed_types}. Seu campo: {value}")
            else:
                if not any(isinstance(value, t) for t in allowed_types if t is not None):
                    raise ValueError(f"O campo '{field}' não é um campo válido, só é aceito valor {allowed_types}. Seu campo: {value}")

    def validation_reading(self, reading_json):
        self._validate_field(reading_json, "reading_mode", [TypeFile.JSON, TypeFile.PARQUET, TypeFile.CSV, TypeFile.DATABASE])
        if not os.path.isdir(reading_json["caminho"]):
            raise ValueError(f"O sub-campo 'caminho' não é um campo válido. Seu sub-campo: {reading_json['caminho']}")
        file_extension = Path(reading_json["nome_arquivo"]).suffix[1:]
        if file_extension not in [TypeFile.JSON, TypeFile.PARQUET, TypeFile.CSV, TypeFile.DATABASE]:
            raise ValueError(f"O sub-campo 'nome_arquivo' não é um campo válido. Seu sub-campo: {reading_json['nome_arquivo']}")
        self._validate_optional_field(reading_json, "host", None, [None, str])
        self._validate_optional_field(reading_json, "user", None, [None, str])
        self._validate_optional_field(reading_json, "password", None, [None, str])
        self._validate_optional_field(reading_json, "database", None, [None, str])
        self._validate_optional_field(reading_json, "type_database", None, [None, str])
        self._validate_optional_field(reading_json, "query", None, [None, str])

    def validation_pipeline(self, pipeline_json):
        self._validate_optional_field(pipeline_json, "numeric_features", None, [None, list])
        self._validate_optional_field(pipeline_json, "categorical_features", None, [None, list])
        self._validate_optional_field(pipeline_json, "normalize_method_num", None, [None, str])
        self._validate_optional_field(pipeline_json, "normalize_method_cat", None, [None, str])

    def validation_validations(self, validations_json):
        self._validate_optional_field(validations_json, "scoring", None, [None, str])
        self._validate_optional_field(validations_json, "folders", None, [None, int])
        self._validate_optional_field(validations_json, "grid", None, [None, bool])
        self._validate_optional_field(validations_json, "random", None, [None, bool])
        self._validate_optional_field(validations_json, "cross_validation", None, [None, bool])
        self._validate_optional_field(validations_json, "param", None, [None, dict])

    def validation_mode(self, mode_json):
        self._validate_field_type(mode_json, "params", [dict])
        self._validate_field_type(mode_json, "name_mode", [str])
        self._validate_field_type(mode_json, "algorithm", [str])
        self._validate_optional_field(mode_json, "target", None, [None, str])
        self._validate_optional_field(mode_json, "learning_network", None, [None, str])