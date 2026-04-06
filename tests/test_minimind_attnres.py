import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import (
    BlockAttnResMiniMindBlock,
    FullAttnResMiniMindBlock,
    MiniMindConfig,
    MiniMindForCausalLM,
    StandardMiniMindBlock,
)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def make_model(residual_mode, hidden_size=768, num_hidden_layers=8, use_final_agg=True):
    config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        residual_mode=residual_mode,
        attnres_block_size=2,
        attnres_use_final_agg=use_final_agg,
    )
    return MiniMindForCausalLM(config)


def test_print_parameter_counts_for_three_modes():
    standard = make_model("standard")
    full = make_model("full_attn_res")
    block = make_model("block_attn_res")

    print(f"standard params: {count_params(standard)}")
    print(f"full_attn_res params: {count_params(full)}")
    print(f"block_attn_res params: {count_params(block)}")


def test_standard_mode_uses_standard_blocks_only():
    model = make_model("standard")

    assert all(isinstance(layer, StandardMiniMindBlock) for layer in model.model.layers)
    assert not any(isinstance(layer, FullAttnResMiniMindBlock) for layer in model.model.layers)
    assert not any(isinstance(layer, BlockAttnResMiniMindBlock) for layer in model.model.layers)
    assert not hasattr(model.model, "out_res_query")
    assert not hasattr(model.model, "out_res_norm")


def test_full_attn_res_mode_uses_full_blocks_and_final_agg_params():
    model = make_model("full_attn_res")

    assert all(isinstance(layer, FullAttnResMiniMindBlock) for layer in model.model.layers)
    assert hasattr(model.model, "out_res_query")
    assert hasattr(model.model, "out_res_norm")
    for layer in model.model.layers:
        assert hasattr(layer, "attn_res_query")
        assert hasattr(layer, "attn_res_norm")
        assert hasattr(layer, "mlp_res_query")
        assert hasattr(layer, "mlp_res_norm")


def test_block_attn_res_mode_uses_block_blocks_and_final_agg_params():
    model = make_model("block_attn_res")

    assert all(isinstance(layer, BlockAttnResMiniMindBlock) for layer in model.model.layers)
    assert hasattr(model.model, "out_res_query")
    assert hasattr(model.model, "out_res_norm")
    for layer in model.model.layers:
        assert hasattr(layer, "attn_res_query")
        assert hasattr(layer, "attn_res_norm")
        assert hasattr(layer, "mlp_res_query")
        assert hasattr(layer, "mlp_res_norm")
        assert layer.block_size == 2


def test_attnres_parameter_delta_matches_expected_with_final_agg():
    hidden_size = 768
    num_hidden_layers = 16

    standard = make_model("standard", hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    full = make_model("full_attn_res", hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    block = make_model("block_attn_res", hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)

    standard_count = count_params(standard)
    full_count = count_params(full)
    block_count = count_params(block)

    # Each AttnRes layer adds:
    # - attn_res_query: hidden_size
    # - attn_res_norm.weight: hidden_size
    # - mlp_res_query: hidden_size
    # - mlp_res_norm.weight: hidden_size
    # Total per layer = 4 * hidden_size
    # Final aggregation adds:
    # - out_res_query: hidden_size
    # - out_res_norm.weight: hidden_size
    # Total final = 2 * hidden_size
    expected_delta = (4 * num_hidden_layers + 2) * hidden_size

    assert full_count - standard_count == expected_delta
    assert block_count - standard_count == expected_delta
    assert full_count == block_count


def test_attnres_parameter_delta_matches_expected_without_final_agg():
    hidden_size = 64
    num_hidden_layers = 3

    standard = make_model("standard", hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_final_agg=False)
    full = make_model("full_attn_res", hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_final_agg=False)
    block = make_model("block_attn_res", hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_final_agg=False)

    standard_count = count_params(standard)
    full_count = count_params(full)
    block_count = count_params(block)
    expected_delta = 4 * num_hidden_layers * hidden_size

    assert not hasattr(full.model, "out_res_query")
    assert not hasattr(block.model, "out_res_query")
    assert full_count - standard_count == expected_delta
    assert block_count - standard_count == expected_delta
    assert full_count == block_count


def test_three_modes_produce_same_logits_shape():
    input_ids = torch.randint(0, 128, (2, 5))

    for mode in ("standard", "full_attn_res", "block_attn_res"):
        model = make_model(mode, hidden_size=64, num_hidden_layers=2)
        output = model(input_ids, labels=input_ids)
        assert output.logits.shape == (2, 5, model.config.vocab_size)
