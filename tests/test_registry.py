import pytest

from transformerlab.attention import AttentionConfig, MaskType, available_attentions, build_attention
from transformerlab.attention.registry import ATTENTION_REGISTRY

EXPECTED = {"mha", "sdpa", "gqa", "mqa", "linear", "local", "flash"}


def test_all_variants_registered():
    assert EXPECTED.issubset(set(available_attentions()))


def test_build_unknown_raises():
    with pytest.raises(KeyError):
        build_attention("does_not_exist", AttentionConfig(dim=32, num_heads=4))


def test_names_match_keys():
    for name, cls in ATTENTION_REGISTRY.items():
        assert cls.name == name


def test_supports_mask_advertised():
    linear = ATTENTION_REGISTRY["linear"]
    assert linear.supports_mask(MaskType.CAUSAL)
    assert not linear.supports_mask(MaskType.FULL)
    assert ATTENTION_REGISTRY["mha"].supports_mask(MaskType.FULL)
