import os
import sys
import json
import pytest
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class ModelFile(BaseModel):
    path: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(min_length=64, max_length=64)

class QuantizationInfo(BaseModel):
    scheme: str
    metadata_source: str

class LicenseInfo(BaseModel):
    spdx_id: str
    url: Optional[str] = None

class BackendTestInfo(BaseModel):
    backend: str
    backend_version: str
    status: str
    notes: Optional[str] = None

class ModelManifest(BaseModel):
    schema_version: str = "1.0"
    model_id: str
    display_name: str
    publisher: str
    revision: str
    format: str
    architecture: str
    modalities: List[str]
    files: List[ModelFile]
    quantization: QuantizationInfo
    license: LicenseInfo
    tested_backends: List[BackendTestInfo] = []

def test_valid_manifest_schema():
    sample_manifest = {
        "schema_version": "1.0",
        "model_id": "ZAQERIS/ZAQERIS_AXIOM",
        "display_name": "ZAQERIS AXIOM M3",
        "publisher": "ZAQERIS AI",
        "revision": "c0ffee1234567890",
        "format": "gguf",
        "architecture": "llama",
        "modalities": ["text"],
        "files": [
            {
                "path": "model.gguf",
                "size_bytes": 1048576,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            }
        ],
        "quantization": {
            "scheme": "Q4_K_M",
            "metadata_source": "gguf-header"
        },
        "license": {
            "spdx_id": "Apache-2.0",
            "url": "https://opensource.org/licenses/Apache-2.0"
        },
        "tested_backends": [
            {
                "backend": "qerlio-native-cpu",
                "backend_version": "0.3.1",
                "status": "experimental"
            }
        ]
    }
    manifest = ModelManifest.model_validate(sample_manifest)
    assert manifest.model_id == "ZAQERIS/ZAQERIS_AXIOM"
    assert manifest.files[0].size_bytes == 1048576

def test_invalid_hash_length_rejected():
    invalid_manifest = {
        "schema_version": "1.0",
        "model_id": "ZAQERIS/invalid",
        "display_name": "Invalid",
        "publisher": "ZAQERIS AI",
        "revision": "1",
        "format": "gguf",
        "architecture": "llama",
        "modalities": ["text"],
        "files": [
            {
                "path": "model.gguf",
                "size_bytes": 100,
                "sha256": "short_hash"  # Invalid length
            }
        ],
        "quantization": {"scheme": "Q4_0", "metadata_source": "header"},
        "license": {"spdx_id": "MIT"}
    }
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(invalid_manifest)
