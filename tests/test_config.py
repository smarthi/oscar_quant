import json

from pydantic import ValidationError

from granite_oscar_quant.config import OscarKVConfig
from granite_oscar_quant.models import DEFAULT_GRANITE_MODEL_ID
from granite_oscar_quant.schemas import BenchmarkReport, BenchmarkRun


def test_default_config_validates_and_exports_namespace():
    config = OscarKVConfig()

    namespace = config.as_namespace()

    assert namespace.k_bits == 2
    assert namespace.v_bits == 2
    assert namespace.k_groupsize == 32
    assert namespace.k_norm_factoring is True
    assert config.model_dump()["v_groupsize"] == 32


def test_invalid_bits_raise():
    try:
        OscarKVConfig(k_bits=1)
    except ValidationError as exc:
        assert "greater than or equal to 2" in str(exc)
    else:
        raise AssertionError("Expected invalid bit width to raise")


def test_default_model_focuses_granite_40_1b_base():
    assert DEFAULT_GRANITE_MODEL_ID == "ibm-granite/granite-4.0-1b-base"


def test_benchmark_report_serializes_as_json():
    report = BenchmarkReport(
        model_id=DEFAULT_GRANITE_MODEL_ID,
        prompt_tokens=5,
        k_bits=2,
        v_bits=2,
        runs=[
            BenchmarkRun(
                label="baseline",
                elapsed_seconds=1.25,
                new_tokens=10,
                tokens_per_second=8.0,
                text=" Paris.",
            )
        ],
    )

    payload = json.loads(report.model_dump_json())

    assert payload["model_id"] == DEFAULT_GRANITE_MODEL_ID
    assert payload["runs"][0]["label"] == "baseline"
