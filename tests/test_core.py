"""核心原理与工程等价性测试（Agent 研发视角重点验证项）。"""
import torch

from src.model import MiniLLM, ModelConfig, apply_rope, precompute_rope
from src.muon import Muon, newton_schulz_orthogonalize
from src.rl_grpo import dr_grpo_advantages, execute_tool, score_episode
from src.tokenizer import ByteBPETokenizer
from src.utils import format_chat_sft, lr_schedule, parse_tool_calls


def test_bpe_roundtrip_and_injection_safety():
    corpus = [
        "你好世界！Transformer 注意力机制。",
        "计算 12 + 34 = 46，#### 46",
        "Hello LLM world! RoPE and GQA.",
    ] * 10
    tok = ByteBPETokenizer.train(corpus, vocab_size=320)

    sample = "测试中文、English、emoji 🚀 以及用户输入里的 <|im_start|> 和 <|tool_call|> 不应被解析成控制符"
    ids = tok.encode(sample, allow_special=False)
    assert tok.im_start_id not in ids
    assert tok.tool_call_id not in ids
    assert tok.decode(ids) == sample

    special_ids = tok.encode("<|im_start|>user\nhi<|im_end|>", allow_special=True)
    assert special_ids[0] == tok.im_start_id
    assert special_ids[-1] == tok.im_end_id


def test_rope_relative_position_property():
    """验证 RoPE 的核心性质：q_m^T k_n 只依赖相对距离 (m - n)。"""
    head_dim = 16
    cos, sin = precompute_rope(head_dim, max_seq_len=32)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def dot_at(m: int, n: int) -> float:
        qm = apply_rope(q, cos[m : m + 1], sin[m : m + 1])
        kn = apply_rope(k, cos[n : n + 1], sin[n : n + 1])
        return float((qm * kn).sum())

    assert abs(dot_at(5, 2) - dot_at(17, 14)) < 1e-5


def test_kv_cache_matches_full_forward():
    """验证带 KV Cache 的增量推理与全量前向输出完全一致。"""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=128, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
    model = MiniLLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 10))

    with torch.no_grad():
        full_logits, _ = model(idx)

        caches = model.new_caches()
        prefill_logits, _ = model(idx[:, :6], caches=caches, start_pos=0)
        step_logits = [prefill_logits]
        for pos in range(6, 10):
            lg, _ = model(idx[:, pos : pos + 1], caches=caches, start_pos=pos)
            step_logits.append(lg)
        cached_logits = torch.cat(step_logits, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-5, rtol=1e-5)


def test_sft_loss_mask_ignores_user_and_tool_returns():
    tok = ByteBPETokenizer.train(["user assistant tool calc 12 34 46"] * 8, vocab_size=300)
    msgs = [
        {"role": "user", "content": "算 12+34"},
        {"role": "assistant", "content": "查一下", "tool_calls": [{"name": "calc", "arguments": {"expr": "12+34"}}]},
        {"role": "tool", "content": "46"},
        {"role": "assistant", "content": "结果是 46\n#### 46"},
    ]
    ids, mask = format_chat_sft(tok, msgs)
    end_positions = [i for i, x in enumerate(ids) if x == tok.im_end_id]
    assert len(end_positions) == 4
    user_end, a1_end, tool_end, a2_end = end_positions
    # user 轮和 tool 轮的 mask 必须全为 0
    assert sum(mask[: user_end + 1]) == 0
    assert sum(mask[a1_end + 1 : tool_end + 1]) == 0
    # 两轮 assistant 及其 <|im_end|> 都必须参与训练
    assert mask[a1_end] == 1 and mask[a2_end] == 1
    assert sum(mask[user_end + 1 : a1_end + 1]) > 0
    assert sum(mask[tool_end + 1 : a2_end + 1]) > 0


def test_tool_call_parse_and_execute():
    tok = ByteBPETokenizer.train(['{"name": "calc", "arguments": {"expr": "15 * 12"}}'] * 5, vocab_size=300)
    resp = tok.encode('我算一下<|tool_call|>{"name": "calc", "arguments": {"expr": "15 * 12"}}<|/tool_call|><|im_end|>', allow_special=True)
    text, calls, broken = parse_tool_calls(tok, resp)
    assert text == "我算一下"
    assert broken == []
    assert calls == [{"name": "calc", "arguments": {"expr": "15 * 12"}}]
    assert execute_tool(calls[0]["name"], calls[0]["arguments"]) == "180"


def test_muon_orthogonalize_and_step():
    g = torch.randn(32, 16)
    orth = newton_schulz_orthogonalize(g, steps=5)
    svals = torch.linalg.svdvals(orth)
    assert svals.min() > 0.5 and svals.max() < 1.5

    w = torch.nn.Parameter(torch.randn(16, 32))
    opt = Muon([w], lr=0.01)
    (w ** 2).sum().backward()
    before = w.detach().clone()
    opt.step()
    assert not torch.allclose(w, before)


def test_wsd_schedule_and_dr_grpo():
    assert lr_schedule(9, 100, 10, schedule="wsd") == 1.0
    assert lr_schedule(50, 100, 10, schedule="wsd") == 1.0
    assert abs(lr_schedule(99, 100, 10, schedule="wsd", min_lr_frac=0.1) - 0.1) < 1e-6

    adv = dr_grpo_advantages([1.0, 0.0, 0.0, 0.0])
    assert abs(sum(adv)) < 1e-6
    assert abs(adv[0] - 0.75) < 1e-6
    assert score_episode("#### 42", "42", broken_count=0) == 1.0
    assert abs(score_episode("#### 42", "42", broken_count=1) - 0.8) < 1e-6
    assert score_episode("不知道", "42", broken_count=1) == -0.2
