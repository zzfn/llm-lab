"""全面单元测试：覆盖麻雀虽小五脏俱全的全部核心模块与数学等价性。"""
import torch

from src.distill import merge_into, reverse_kl
from src.dpo import dpo_loss
from src.lora import apply_lora, merge_lora
from src.loss import margin_saturation, stable_cross_entropy, token_log_probs
from src.model import MiniLLM, ModelConfig, apply_rope, precompute_rope, sinkhorn
from src.muon import Muon, newton_schulz_orthogonalize
from src.rl_grpo import dr_grpo_advantages, execute_tool, score_choice, score_instruct, score_task
from src.serve import to_internal_messages
from src.tokenizer import ByteBPETokenizer
from src.utils import format_chat_prompt, format_chat_sft, lr_schedule, parse_tool_calls, split_thinking


def test_bpe_roundtrip_and_injection_safety():
    corpus = ["你好世界！Transformer 注意力机制。", "计算 12 + 34 = 46，#### 46"] * 10
    tok = ByteBPETokenizer.train(corpus, vocab_size=320)
    sample = "测试中文、English、emoji 🚀 以及 <|im_start|> 和 <|tool_call|> 防注入"
    ids = tok.encode(sample, allow_special=False)
    assert tok.im_start_id not in ids and tok.tool_call_id not in ids
    assert tok.decode(ids) == sample


def test_stable_cross_entropy_prevents_softmax_collapse():
    # 构造极端高置信度 logits（领先间隔 m = 25 > 16.6），标准 softmax 在 FP32 下已饱和，但稳定写法仍应给出准确梯度
    logits = torch.tensor([[25.0, 0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([0])
    assert margin_saturation(logits.detach(), targets) == 1.0

    loss = stable_cross_entropy(logits, targets)
    loss.backward()
    # 导数约等于 -exp(-25) ≈ -1.39e-11（严格非零，且全词表梯度之和为 0）
    assert logits.grad[0, 0].item() < 0.0
    assert abs(logits.grad.sum().item()) < 1e-12
    assert torch.allclose(token_log_probs(logits.detach(), targets), -loss.detach())


def test_mhc_sinkhorn_doubly_stochastic():
    logits = torch.randn(4, 4)
    mat = sinkhorn(logits, iters=20)
    assert torch.allclose(mat.sum(dim=-1), torch.ones(4), atol=1e-4)
    assert torch.allclose(mat.sum(dim=-2), torch.ones(4), atol=1e-4)
    assert (mat >= 0).all()


def test_rope_relative_position_property():
    cos, sin = precompute_rope(head_dim=16, max_seq_len=32)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def dot_at(m: int, n: int) -> float:
        return float((apply_rope(q, cos[m : m + 1], sin[m : m + 1]) * apply_rope(k, cos[n : n + 1], sin[n : n + 1])).sum())

    assert abs(dot_at(5, 2) - dot_at(17, 14)) < 1e-5


def test_full_architecture_kv_cache_matches_full_forward():
    """同时开启 mHC + Engram + 3:1 滑动窗口(window=4) + MoE，验证带缓存逐词解码与全量前向等价。"""
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=128, dim=64, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=32,
        window=4, long_every=4, n_streams=4,
        engram_orders=(2, 3), engram_heads=4, engram_head_dim=8, engram_rows=64, engram_layers=(1,),
        moe_experts=4, moe_top_k=2, moe_dense_layers=1,
    )
    model = MiniLLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 10))

    with torch.no_grad():
        full_logits, _ = model(idx, return_logits=True)

        state = model.new_state()
        prefill_logits, _ = model(idx[:, :6], state=state, return_logits=True)
        step_logits = [prefill_logits]
        for pos in range(6, 10):
            lg, _ = model(idx[:, pos : pos + 1], state=state, return_logits=True)
            step_logits.append(lg)
        cached_logits = torch.cat(step_logits, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-4, rtol=1e-4)


def test_moe_balance_updates_only_after_step():
    cfg = ModelConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16,
        window=8, moe_experts=4, moe_top_k=2, moe_dense_layers=1, engram_orders=(),
    )
    model = MiniLLM(cfg).train()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    moe = model.blocks[1].ffn
    before = moe.balance_bias.clone()
    _, loss = model(x, y)
    loss.backward()
    # 前向/反向期间偏置保持不变（保证激活重计算一致性）
    assert torch.equal(moe.balance_bias, before)
    model.update_balance()
    assert not torch.equal(moe.balance_bias, before)


def test_sft_loss_mask_and_tool_parsing():
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
    assert sum(mask[: user_end + 1]) == 0
    assert sum(mask[a1_end + 1 : tool_end + 1]) == 0
    assert mask[a1_end] == 1 and mask[a2_end] == 1

    resp = tok.encode('查一下<|tool_call|>{"name": "calc", "arguments": {"expr": "12+34"}}<|/tool_call|><|im_end|>', allow_special=True)
    text, calls, broken = parse_tool_calls(tok, resp)
    assert text == "查一下" and not broken and execute_tool(calls[0]["name"], calls[0]["arguments"]) == "46"


def test_mopd_reverse_kl_and_task_arithmetic():
    logits = torch.tensor([[[2.0, 0.0, -1.0]]])
    assert reverse_kl(logits, logits).item() < 1e-6

    cfg = ModelConfig(vocab_size=32, dim=16, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=16, window=8, engram_orders=())
    s = MiniLLM(cfg)
    e1 = MiniLLM(cfg)
    with torch.no_grad():
        for p in e1.parameters():
            p.copy_(torch.ones_like(p))
        for p in s.parameters():
            p.zero_()
    merge_into(s, [e1])
    assert torch.allclose(next(s.parameters()), torch.ones_like(next(s.parameters())))


def test_muon_wsd_rlvr_and_serve_messages():
    orth = newton_schulz_orthogonalize(torch.randn(32, 16), steps=5)
    svals = torch.linalg.svdvals(orth)
    assert svals.min() > 0.5 and svals.max() < 1.5

    w = torch.nn.Parameter(torch.randn(16, 32))
    opt = Muon([w], lr=0.01)
    (w ** 2).sum().backward()
    before = w.detach().clone()
    opt.step()
    assert not torch.allclose(w, before)

    assert lr_schedule(50, 100, 10, schedule="wsd") == 1.0
    assert abs(lr_schedule(99, 100, 10, schedule="wsd", min_lr_frac=0.1) - 0.1) < 1e-6

    adv = dr_grpo_advantages([1.0, 0.0, 0.0, 0.0])
    assert abs(sum(adv)) < 1e-6 and abs(adv[0] - 0.75) < 1e-6
    assert score_instruct('{"summary": "a", "benefit": "b"}', {"json_keys": ["summary", "benefit"]}) == 1.0
    assert score_choice("答案：B", "B") == 1.0
    assert score_task({"reward": "agent", "answer": "42"}, "#### 42", broken_count=1) == 0.8

    internal = to_internal_messages(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "calc", "parameters": {}}}],
    )
    assert internal[0]["role"] == "system" and "calc" in internal[0]["content"]


def test_lora_zero_init_and_lossless_merge():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16, window=8, engram_orders=())
    model = MiniLLM(cfg).eval()
    x = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        base_out, _ = model(x, return_logits=True)

    trainable, total = apply_lora(model, rank=4, alpha=8.0)
    assert 0 < trainable < total
    # B 全零初始化：刚挂上 LoRA 时输出与原模型严格相等
    with torch.no_grad():
        init_lora_out, _ = model(x, return_logits=True)
    assert torch.allclose(base_out, init_lora_out, atol=1e-6)

    # 模拟训练更新 B 后，验证 merge_lora 前后前向严格等价
    for name, p in model.named_parameters():
        if ".B.weight" in name:
            p.data.normal_(std=0.05)
    with torch.no_grad():
        unmerged_out, _ = model(x, return_logits=True)
    merge_lora(model)
    with torch.no_grad():
        merged_out, _ = model(x, return_logits=True)
    assert torch.allclose(unmerged_out, merged_out, atol=1e-5, rtol=1e-5)


def test_dpo_loss_prefers_higher_chosen_margin():
    ref_c = torch.tensor([-10.0, -10.0])
    ref_r = torch.tensor([-10.0, -10.0])
    good_c = torch.tensor([-8.0, -8.0])
    bad_r = torch.tensor([-12.0, -12.0])
    loss_good, acc_good = dpo_loss(good_c, bad_r, ref_c, ref_r, beta=0.1)
    loss_bad, acc_bad = dpo_loss(bad_r, good_c, ref_c, ref_r, beta=0.1)
    assert loss_good < loss_bad
    assert acc_good == 1.0 and acc_bad == 0.0


def test_yarn_rope_and_thinking_switch():
    cos_base, _ = precompute_rope(16, 64, base=10000.0)
    cos_yarn, _ = precompute_rope(16, 64, base=10000.0, orig_max_len=16, yarn_factor=4.0)
    assert not torch.allclose(cos_base, cos_yarn)

    tok = ByteBPETokenizer.train(["user assistant think calc"] * 5, vocab_size=300)
    on_ids = format_chat_prompt(tok, [{"role": "user", "content": "hi"}], enable_thinking=True)
    off_ids = format_chat_prompt(tok, [{"role": "user", "content": "hi"}], enable_thinking=False)
    assert on_ids[-1] == tok.think_id
    assert off_ids[-2:] == [tok.think_id, tok.think_end_id]
    r, c = split_thinking("<think>先算乘法</think>结果是 42")
    assert r == "先算乘法" and c == "结果是 42"
