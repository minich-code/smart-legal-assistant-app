import os
from box.exceptions import BoxValueError
import yaml
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, List
from backend.src.utils.exception import LegalRAGException
from backend.src.utils.logger import logger

def get_project_root() -> Path:
    """
    Returns the project root directory (case-search-RAG).
    """
    return Path(__file__).parent.parent.parent.parent

@ensure_annotations
def read_yaml(path_to_yaml: Path | str = "backend/config/config.yaml") -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path | str): Path to the YAML file, relative to the project root or absolute.

    Returns:
        ConfigBox: Parsed YAML content as a ConfigBox object, which allows attribute-style access.

    Raises:
        LegalRAGException: If the file is empty, not found, or an error occurs during reading.
    """
    try:
        # Convert string to Path and resolve relative to project root
        path_to_yaml = Path(path_to_yaml)
        if not path_to_yaml.is_absolute():
            path_to_yaml = get_project_root() / path_to_yaml
        if not path_to_yaml.exists():
            raise FileNotFoundError(f"No such file or directory: '{path_to_yaml}'")
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("YAML file is empty")
            logger.info(f"YAML file: {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError as e:
        raise LegalRAGException(
            error=e,
            error_type="EmptyYAML",
            context={"path": str(path_to_yaml)},
            status_code=400,
            log_immediately=True
        )
    except FileNotFoundError as e:
        raise LegalRAGException(
            error=e,
            error_type="FileNotFound",
            context={"path": str(path_to_yaml)},
            status_code=404,
            log_immediately=True
        )
    except Exception as e:
        raise LegalRAGException(
            error=e,
            error_type="YAMLReadError",
            context={"path": str(path_to_yaml)},
            status_code=500,
            log_immediately=True
        )

@ensure_annotations
def create_directories(path_to_directories: List[Path], verbose: bool = True):
    """
    Creates directories specified in the list if they do not exist.

    Args:
        path_to_directories (List[Path]): List of directory paths to create.
        verbose (bool): If True, logs the directory creation.

    Raises:
        LegalRAGException: If an error occurs while creating a directory.
    """
    for path in path_to_directories:
        try:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
        except Exception as e:
            raise LegalRAGException(
                error=e,
                error_type="DirectoryCreationError",
                context={"path": str(path)},
                status_code=500,
                log_immediately=True
            )

@ensure_annotations
def save_object(obj: Any, file_path: Path) -> None:
    """
    Saves a Python object to a file using joblib.

    Args:
        obj: The Python object to save.
        file_path (Path): The path where the object should be saved.

    Raises:
        LegalRAGException: If an error occurs during saving.
    """
    try:
        file_path = Path(file_path)
        dir_path = file_path.parent
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        raise LegalRAGException(
            error=e,
            error_type="ObjectSaveError",
            context={"path": str(file_path)},
            status_code=500,
            log_immediately=True
        )

@ensure_annotations
def load_object(file_path: Path) -> Any:
    """
    Loads a Python object from a file using joblib.

    Args:
        file_path (Path): Path of the file to load the object from.

    Returns:
        Any: The loaded Python object.

    Raises:
        LegalRAGException: If an error occurs during loading.
    """
    try:
        file_path = Path(file_path)
        with open(file_path, 'rb') as file_obj:
            obj = joblib.load(file_obj)
            logger.info(f"Object loaded from: {file_path}")
            return obj
    except Exception as e:
        raise LegalRAGException(
            error=e,
            error_type="ObjectLoadError",
            context={"path": str(file_path)},
            status_code=500,
            log_immediately=True
        )