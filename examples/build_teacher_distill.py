"""由大模型教师构造的高质量黑盒蒸馏数据集（Black-box Trajectory Distillation + 确定性工具验算过滤）。

用途：
    python -m examples.build_teacher_distill --out data/sft_teacher.jsonl
    python -m examples.run_qwen_mlx --src-data data/sft_teacher.jsonl --adapter-path out/qwen35_teacher_lora --iters 60 --lr 2e-5 --skip-base
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.rl_grpo import execute_tool
from src.utils import SYSTEM_PROMPT

# 教师精心构造的多领域“免工具”高质量问答（严禁与评测集 4 道题重合，用于防过拟合与防乱调 calc）
TEACHER_QA: list[tuple[str, str, str]] = [
    # --- LLM 与推理系统原理 ---
    (
        "Transformer 推理时为什么分 Prefill 和 Decode 两个阶段？",
        "这是推理架构概念题，无需调用 calc，分两阶段对照说明输入并行与自回归瓶颈即可。",
        "Prefill 阶段一次性并行处理全部输入 prompt 并写入缓存，计算密集；Decode 阶段逐个 token 自回归生成，每步受限于显存带宽与历史状态读取，属于访存密集阶段。",
    ),
    (
        "GQA（Grouped-Query Attention）是如何压缩推理显存的？",
        "这是注意力机制架构题，无需调用 calc，直接点出多个 Q 头共享一组 KV 头即可。",
        "GQA 让多个查询头（Query Head）共享同一组键（Key）和值（Value）头，从而把需要缓存的历史键值张量体积按分组倍数成比例缩小，同时保留接近 MHA 的表达能力。",
    ),
    (
        "为什么 Agent 的 system prompt 和工具定义尽量保持固定不要每轮改动？",
        "这是推理优化原理题，无需调用 calc，核心解释 Prefix Cache 命中即可。",
        "固定前缀不变可以持续命中推理引擎的 Prefix Cache（前缀缓存），避免每轮都重新对几百上千个系统词元和工具描述做 Prefill 计算，显著降低首字延迟（TTFT）。",
    ),
    (
        "解释一下 RoPE 旋转位置编码的核心优势。",
        "概念问答无需调用 calc，用通俗准确的语言概括相对位置性质和外推友好性。",
        "RoPE 通过对向量的相邻维度做角度随位置变化的二维旋转，使 Query 与 Key 的点积自然只依赖两者的相对距离，并可通过调大基频支持更长上下文外推。",
    ),
    (
        "为什么自回归模型序列越长，单步生成耗时和显存压力越大？",
        "原理分析题，无需调用 calc，从注意力对历史上下文的读取复杂度切入。",
        "因为每生成一个新 token，注意力层都需要与前面所有历史 token 的缓存状态做交互计算；序列越长，每步搬运的历史张量越多，带宽开销和显存占用随之线性增长。",
    ),
    (
        "什么是 MoE（混合专家模型）里的路由负载均衡？不做会怎样？",
        "架构机制题，无需调用 calc，直击路由坍塌问题。",
        "路由负载均衡用于引导门控网络把不同 token 均匀分配给各专家子网络；若不做均衡约束，少数热门专家会承受绝大部分梯度而其余专家得不到训练，导致模型退化为小模型。",
    ),
    (
        "微调大模型时，为什么通常把 user 提示词和 tool 返回结果的 loss mask 掉？",
        "训练目标设计题，无需调用 calc，直接解释因果监督边界。",
        "因为用户输入和外部工具返回值属于环境观测而非模型应当生成的动作；屏蔽它们的交叉熵损失后，模型才会专注学习如何组织推理、发起调用与总结结论。",
    ),
    (
        "LoRA 微调相比全量微调为什么更不容易遗忘基座原本的通用能力？",
        "微调原理题，无需调用 calc，重点说明冻结主干与低秩增量。",
        "LoRA 冻结预训练主干权重，仅在注意力或 MLP 上旁路训练极低秩的增量矩阵（参数占比常不足 1%），天然限制了权重偏移幅度，从而减轻灾难性遗忘。",
    ),
    (
        "为什么 4-bit 量化后的小模型再接 LoRA 能大幅省显存？",
        "工程优化题，无需调用 calc，说明主干低比特量化存储和增量高精度训练。",
        "QLoRA/4-bit LoRA 把占主体的冻结基座参数以 4-bit 紧凑存放在显存中，仅针对少量可训练的旁路适配器保留高精度参数与优化器状态。",
    ),
    (
        "解释什么是幻觉（Hallucination）以及接外部工具对减少幻觉有什么作用？",
        "概念与 Agent 工程题，无需调用 calc，简要说明原因与工具验证作用。",
        "大模型幻觉指模型以流畅语气输出与事实或数学逻辑不符的虚构内容；引入计算器、检索等外部工具可把精确计算与事实查询交给确定性程序，从而大幅降低编造概率。",
    ),
    (
        "什么叫 Exposure Bias（暴露偏差）？在线蒸馏（On-policy Distillation）怎么缓解它？",
        "蒸馏原理题，无需调用 calc，区分训练态前缀与测试态自生成前缀。",
        "传统离线监督学习只在标准答案前缀上训练，但推理时模型要顺着自己刚生成的 token 续写，一旦写错就容易越偏越远；在线蒸馏让学生在自己生成的轨迹上向老师对齐，能有效弥补这一差距。",
    ),
    (
        "Muon 优化器和常见 AdamW 在更新二维权重矩阵时思路有什么不同？",
        "优化器对比题，无需调用 calc，讲清正交化与逐元素自适应缩放区别。",
        "AdamW 对每个参数独立做一阶/二阶矩自适应缩放，而 Muon 通过 Newton-Schulz 迭代对二维权重更新矩阵做近似正交化，使各个奇异方向的更新步长更均衡。",
    ),
    # --- 自然科学与常识（简明准确，不瞎调用计算器） ---
    (
        "用一句话解释为什么月亮会有阴晴圆缺（月相变化）？",
        "纯天文常识题，无需调用 calc，用一句话精准说明日地月相对位置变化。",
        "月球自身不发光且绕地球公转，地球上观测到的月球受太阳照亮半球的视角随日、地、月三者相对位置周期性改变，从而形成月相盈亏。",
    ),
    (
        "用一句话说明海水潮汐主要是由什么引起的？",
        "自然科学常识题，无需调用 calc，一句话交代核心成因。",
        "海水潮汐主要是由月球（及太阳）对地球不同部位海水的万有引力差——即引潮力——周期性作用引起的。",
    ),
    (
        "高原上煮米饭为什么常常夹生？",
        "物理常识题，无需调用 calc，结合气压与沸点关系回答。",
        "高原海拔高、大气压强低，水的沸点随气压降低而下降（常低于 100℃），水沸腾时温度不够高，米粒中的淀粉难以充分糊化熟透。",
    ),
    (
        "用一句话解释温室效应是如何让地球表面保持温暖的？",
        "地理科学题，无需调用 calc，清晰概括大气对长波辐射的吸收。",
        "太阳短波辐射穿过大气层加热地表后，大气中的二氧化碳、水汽等温室气体会吸收地表向外散发的部分长波热辐射并向地面回传热量，从而使地表维持适宜温度。",
    ),
    (
        "为什么雨后天空有时会出现彩虹？",
        "光学常识题，无需调用 calc，点出折射、色散和反射。",
        "雨后空气中的小水滴像无数微小的棱镜，太阳光进入水滴发生折射、内反射和再折射时，不同波长的色光因折射率不同而色散分离，便形成了彩虹。",
    ),
    (
        "金属勺放在热汤里很快变烫，而木勺几乎不烫手，原因是什么？",
        "热学常识题，无需调用 calc，比较导热系数与自由电子。",
        "金属内部存在大量自由电子，热传导速率极快；而木材属于热的不良导体，分子热运动传递热量很慢，所以木勺握柄升温缓慢。",
    ),
    (
        "为什么飞机在万米高空飞行时，机窗外常常处于平流层？",
        "气象与航空常识题，无需调用 calc，对比对流层和平流层特征。",
        "平流层位于对流层之上，水汽极少、几乎没有雷雨云和强垂直对流，气流以水平运动为主，能见度高且飞行更平稳省油。",
    ),
    (
        "闪电发生时，为什么我们总是先看到闪光后听到雷声？",
        "物理常识题，无需调用 calc，对比光速与声速差异。",
        "闪电和雷声几乎同时产生，但光在空气中的传播速度（约每秒 30 万公里）远大于声音在空气中的传播速度（约每秒 340 米），因此视觉信号总是早于听觉到达。",
    ),
    (
        "植物叶片大多呈绿色，在光学上是什么原因？",
        "生物学与光学常识题，无需调用 calc，解释叶绿素对不同波长的吸收与反射。",
        "植物细胞内的叶绿素强烈吸收太阳光中的红光和蓝紫光用于光合作用，而对绿光吸收较少并将其大量反射出来，因此人眼看到的叶片呈绿色。",
    ),
    (
        "用一句话解释为什么冰块会浮在水面上？",
        "物理化学常识题，无需调用 calc，一句话讲透密度差异。",
        "水结冰时水分子形成疏松的氢键晶体结构，导致体积膨胀、密度小于液态水，因此冰块受到的浮力足以使其漂浮在水面。",
    ),
    # --- 编程、逻辑、格式约束与日常推理 ---
    (
        "请用不超过 25 字解释什么是哈希表，不要出现逗号。",
        "字数与标点受限的文本题，无需调用 calc，直接拟定不带逗号且 25 字以内的解释。",
        "哈希表通过散列函数将键直接映射到数组位置实现快速读写。",
    ),
    (
        "Python 里列表 list 和元组 tuple 最本质的两个区别是什么？",
        "编程基础题，无需调用 calc，简明列出可变性与使用场景即可。",
        "第一，list 可变（支持原地增删改元素），tuple 创建后不可变；第二，因为不可变且支持内容哈希，元素可哈希的 tuple 可作为 dict 的键或 set 的成员，而 list 不能。",
    ),
    (
        "HTTP 状态码 401 和 403 各自代表什么含义？",
        "Web 开发常识题，无需调用 calc，清楚区分未认证与无权限。",
        "401 Unauthorized 表示请求缺少有效身份凭证（未登录或认证失效）；403 Forbidden 表示服务器已识别身份，但该账号没有访问目标资源的权限。",
    ),
    (
        "Git 中 git merge 和 git rebase 的核心区别是什么？",
        "版本控制工具题，无需调用 calc，直白对比提交历史拓扑。",
        "git merge 会保留原分支拓扑并生成一个新的合并提交；git rebase 则会把当前分支的提交按顺序重放到目标基线之后，形成一条线性提交历史。",
    ),
    (
        "如果所有的猫都是哺乳动物，且有些哺乳动物会游泳，能否必然推出“有些猫会游泳”？",
        "形式逻辑推理题，无需调用 calc，直接判断三段论有效性。",
        "不能必然推出。因为“会游泳的哺乳动物”完全可能全部落在猫以外的其他哺乳动物（如海豚、水獭）中，前提没有保证这两个子集存在交集。",
    ),
    (
        "从“今天下雨，地会湿”能反推出“地湿了，所以今天一定下雨”吗？",
        "逻辑谬误辨析题，无需调用 calc，指出充分条件假言推理错误。",
        "不能。这属于肯定后件谬误；地面变湿还可能由洒水车作业、水管破裂或露水等其他充分条件引起。",
    ),
    (
        "用三点总结编写单元测试时为什么要尽量隔离外部网络与数据库依赖？",
        "软件工程题，无需调用 calc，条理分明地列出 3 点。",
        "1. 提升执行速度，毫秒级反馈便于高频运行；2. 保证结果确定可复现，不因网络抖动或脏数据误报；3. 专注验证单个模块的纯业务逻辑。",
    ),
    (
        "只回答“是”或“否”：TCP 协议在建立连接时需要四次挥手。",
        "严格格式约束的是非题，无需调用 calc，建立连接是三次握手而非四次挥手，故输出“否”。",
        "否",
    ),
    (
        "请从以下选项中选出唯一正确的一项，只输出字母：\n以下哪种数据结构遵循后进先出（LIFO）原则？\nA. 队列\nB. 栈\nC. 链表\nD. 堆",
        "单选题且要求只输出字母，无需调用 calc，栈遵循 LIFO，答案是 B。",
        "B",
    ),
    (
        "请从以下选项中选出唯一正确的一项，只输出字母：\n在关系型数据库中，用于确保两张表之间引用完整性的约束是？\nA. 主键\nB. 唯一约束\nC. 外键\nD. 检查约束",
        "数据库单选题且只需输出字母，无需调用 calc，引用完整性对应外键 C。",
        "C",
    ),
]

CN_DIGITS = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}


def int_to_cn(n: int) -> str:
    """将 0~999 整数转为自然中文表达（避开评测题的 38 与 24 组合即可）。"""
    if n < 10:
        return CN_DIGITS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        prefix = "十" if tens == 1 else f"{CN_DIGITS[tens]}十"
        return prefix if ones == 0 else f"{prefix}{CN_DIGITS[ones]}"
    hundreds, rem = divmod(n, 100)
    if rem == 0:
        return f"{CN_DIGITS[hundreds]}百"
    if rem < 10:
        return f"{CN_DIGITS[hundreds]}百零{CN_DIGITS[rem]}"
    tens, ones = divmod(rem, 10)
    tail = f"{CN_DIGITS[tens]}十" + ("" if ones == 0 else CN_DIGITS[ones])
    return f"{CN_DIGITS[hundreds]}百{tail}"


def build_calc_samples(rng: random.Random, count: int = 120) -> list[dict]:
    """构造多样化的数学/工具调用轨迹，并用真实 execute_tool 逐条验算过滤。"""
    rows: list[dict] = []
    seen_q: set[str] = set()

    # 覆盖边界计算与自然中文数词（避开 held-out 的 1 * 0 和 38 * 24）
    templates = [
        ("cn_mul", lambda a, b: (f"帮我算一下{int_to_cn(a)}乘以{int_to_cn(b)}等于多少？", f"{a} * {b}", "cn")),
        ("cn_add", lambda a, b: (f"请问{int_to_cn(a)}加上{int_to_cn(b)}是多少？", f"{a} + {b}", "cn")),
        ("cn_sub", lambda a, b: (f"{int_to_cn(a)}减去{int_to_cn(b)}结果等于几？", f"{a} - {b}", "cn")),
        ("fmt_mul", lambda a, b: (f"请使用计算器计算 {a} * {b}，并按 #### 答案 格式输出。", f"{a} * {b}", "fmt")),
        ("fmt_expr", lambda a, b: (f"请调用工具计算 ({a} + {b}) * 3，并按 #### 答案 格式输出。", f"({a} + {b}) * 3", "fmt")),
        ("word_box", lambda a, b: (f"仓库有 {a} 箱零件，每箱装 {b} 个，一共有多少个零件？请用计算器核对。", f"{a} * {b}", "word")),
        ("word_rem", lambda a, b: (f"一笔预算一共 {a * 10} 元，先花掉 {b * 5} 元，还剩多少元？", f"{a * 10} - {b * 5}", "word")),
        ("div_exact", lambda a, b: (f"请计算 {a * b} 除以 {a} 等于多少，按 #### 答案 格式给出。", f"{a * b} // {a}", "fmt")),
    ]
    zero_cases = [
        ("请使用计算器计算 0 * 25，并按 #### 答案 格式输出。", "0 * 25", "fmt"),
        ("请使用计算器计算 42 * 0，并按 #### 答案 格式输出。", "42 * 0", "fmt"),
        ("请问零乘以八十九等于多少？", "0 * 89", "cn"),
        ("帮我算一算 0 加上 137 等于多少，按 #### 答案 格式输出。", "0 + 137", "fmt"),
    ]

    candidates: list[tuple[str, str, str]] = list(zero_cases)
    while len(candidates) < count + 20:
        kind, fn = rng.choice(templates)
        if kind == "cn_mul":
            a, b = rng.randint(11, 99), rng.randint(11, 99)
            if (a, b) in {(38, 24), (24, 38)}:
                continue
        elif kind == "div_exact":
            a, b = rng.randint(2, 29), rng.randint(2, 49)
        elif kind == "fmt_mul":
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            if {a, b} == {0, 1}:
                continue
        else:
            a, b = rng.randint(12, 180), rng.randint(5, 95)
        q, expr, style = fn(a, b)
        if q not in seen_q:
            seen_q.add(q)
            candidates.append((q, expr, style))

    think_call_templates = [
        lambda e: f"题目需要准确算术结果，提取算式 {e} 调用 calc 计算。",
        lambda e: f"涉及具体数值运算，先用 calc 工具算一下 {e}。",
        lambda e: f"为了避免心算出错，构造表达式 {e} 交给 calc 执行。",
        lambda e: f"属于计算类请求，解析出数学表达式 {e} 后调用 calc。",
    ]
    for idx, (q, expr, style) in enumerate(candidates[:count]):
        res = execute_tool("calc", {"expr": expr})
        if res.startswith("error"):
            continue
        think_call = think_call_templates[idx % len(think_call_templates)](expr)
        if style == "fmt":
            think_ans = f"工具已返回精确计算结果 {res}，用户要求按 #### 答案 格式输出。"
            ans = f"计算结果是 {res}。\n#### {res}"
        elif style == "cn":
            think_ans = f"calc 已返回 {res}，直接简洁回答用户问题即可。"
            ans = f"{expr} 的计算结果是 {res}。"
        else:
            think_ans = f"根据 calc 返回的精确结果 {res}，结合题意清晰写出答案。"
            ans = f"经计算，答案是 {res}。"

        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {
                        "role": "assistant",
                        "thinking": think_call,
                        "content": "",
                        "tool_calls": [{"name": "calc", "arguments": {"expr": expr}}],
                    },
                    {"role": "tool", "content": str(res)},
                    {"role": "assistant", "thinking": think_ans, "content": ans},
                ]
            }
        )
    return rows


def build_dataset(out_path: Path, n_calc: int = 80, seed: int = 42) -> tuple[int, int]:
    rng = random.Random(seed)
    qa_rows = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "thinking": think, "content": ans},
            ]
        }
        for q, think, ans in TEACHER_QA
    ]
    calc_rows = build_calc_samples(rng, count=n_calc)
    all_rows = qa_rows + calc_rows
    rng.shuffle(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + "\n", encoding="utf-8")
    return len(qa_rows), len(calc_rows)


def main():
    p = argparse.ArgumentParser(description="生成教师蒸馏训练集 data/sft_teacher.jsonl")
    p.add_argument("--out", default="data/sft_teacher.jsonl")
    p.add_argument("--n-calc", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    n_qa, n_calc = build_dataset(Path(args.out), n_calc=args.n_calc, seed=args.seed)
    print(f"[Teacher Distill] qa={n_qa} + verified_calc={n_calc} -> {args.out}")


if __name__ == "__main__":
    main()
