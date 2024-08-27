import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import yaml  # type: ignore
from pydantic import BaseModel, ConfigDict
from torch.utils._pytree import tree_map


def overwrite_recursive(d: dict, d_new: dict) -> None:
    for k, v in list(d_new.items()):  # TODO nested overwrite
        if isinstance(v, dict):
            if k not in d:
                d[k] = dict()
            overwrite_recursive(d[k], d_new[k])
        else:
            d[k] = v


V = TypeVar("V")

TBaseConfig = TypeVar("TBaseConfig", bound="BaseConfig")


class BaseConfig(BaseModel):
    """
    Base config class providing general settings for non-mutability and json serialization options
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    def __post_init__(self) -> None:
        pass

    def as_dict(self) -> dict[Any, Any]:
        """
        return a json-serializable dict of self
        """
        self_dict: dict[Any, Any] = self.model_dump()

        def simplify(x: Path | Enum | V) -> str | V:
            if isinstance(x, Path):
                return str(x)
            elif isinstance(x, Enum):
                return x.value
            else:
                return x

        return tree_map(simplify, self_dict)

    @classmethod
    def from_dict(cls: type[TBaseConfig], d: dict, overwrite_values: Optional[dict] = None) -> TBaseConfig:
        if overwrite_values is not None:
            overwrite_recursive(d, overwrite_values)

        return cls(**d)

    def as_str(self) -> str:
        return json.dumps(self.as_dict())

    @classmethod
    def from_str(cls: type[TBaseConfig], s: str) -> TBaseConfig:
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_yaml(
        cls: type[TBaseConfig], yml_filename: Union[str, Path], overwrite_values: Optional[dict] = None
    ) -> TBaseConfig:
        with open(yml_filename) as conf_file:
            config_dict = yaml.load(conf_file, Loader=yaml.FullLoader)

        if overwrite_values is not None:
            overwrite_recursive(config_dict, overwrite_values)
        return cls.from_dict(config_dict)

    def save(self, out_file: Path, indent: int = 4) -> None:
        json.dump(self.as_dict(), open(out_file, "w", encoding="UTF-8"), indent=indent)

    @classmethod
    def get_template_str(cls, indent: int = 4, level: int = 1) -> str:
        """
        Computes a yaml template string of the config with explanatory comments.
        Note that the syntax uses the json subset of yaml, but we use yaml for its comment support.
        It works by iterating over the pydantic BaseModel schema (JSON schema) and accessing the
        model_fields (a dictionary of the model's fields).
        """

        def render_comment(text: str | None, indent: int, level: int) -> str:
            if text is None or text.strip() == "":
                return ""
            out = " " * indent * level + "# "
            out += text.replace("\n", "\n" + " " * indent * level + "# ")
            out += "\n"
            return out

        # get schema
        schema = cls.model_json_schema(by_alias=False)
        fields = cls.model_fields

        # save out
        result = ""
        result += "{\n"
        result += render_comment(cls.__name__, indent, level)
        result += render_comment(cls.__doc__, indent, level)

        for field_index, (field_name, _) in enumerate(schema["properties"].items()):
            field_info = fields[field_name]
            is_last = field_index == len(schema["properties"]) - 1

            # description and field name
            result += " " * level * indent + "\n"
            result += render_comment(field_info.description, indent, level)
            result += " " * level * indent
            result += f'"{field_name}": '

            # field value
            if isinstance(field_info.default, BaseConfig):
                result += field_info.default.get_template_str(indent=indent, level=level + 1)
            else:
                if isinstance(field_info.default, Enum):
                    result += f"{json.dumps(field_info.default.value)}"
                elif field_info.default is not None:
                    try:
                        result += f"{json.dumps(field_info.default)}"
                    except:  # noqa: E722
                        print(field_info)

            # comma in the end
            if not is_last:
                result += ","

            # finalize
            result += "\n"

        result += " " * (level - 1) * indent + "}"

        return result

    @classmethod
    def save_template(cls, out_file: Path, indent: int = 4) -> None:
        """
        Save a yaml config template with comments explaining the fields.
        This is useful as a starting point for creating a new config from scratch.
        Note that the syntax uses the json subset of yaml, but we use yaml for its comment support.
        """
        # convert to yaml str
        result = cls.get_template_str(indent=indent)

        # save out
        with open(out_file, "w", encoding="UTF-8") as f:
            f.write(result)
