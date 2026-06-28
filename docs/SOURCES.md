# KASUMI source notes and search anchors

These notes collect the external sources used to motivate KASUMI's public README and GitHub Pages narrative. They are source cards, not proof that the synthetic model is calibrated to any real ministry.

## Kasumigaseki personnel-policy background

### Cabinet Secretariat / National Personnel Authority young-official proposal

Source: <https://www.cas.go.jp/jp/gaiyou/jimu/jinjikyoku/pdf/wakate_honbun.pdf>

Search anchors:

```text
カラフルな公務を目指して
```

```text
人事異動を、「年功序列」から「手挙げ」へ
```

```text
戦略的な人事のための体制・システムの整備
```

```text
タレントマネジメントシステム
```

```text
業務量に見合った適切な人員の配置
```

```text
残業なし前提の人員配置
```

Why it matters for KASUMI: these anchors motivate the simulation dimensions used here: staffing, transfer design, training, talent management, workload, and guardrails.

### Taro Kono, “危機に直面する霞ヶ関”

Source: <https://www.taro.org/2020/11/%e5%8d%b1%e6%a9%9f%e3%81%ab%e7%9b%b4%e9%9d%a2%e3%81%99%e3%82%8b%e9%9c%9e%e3%83%b6%e9%96%a2.php>

Search anchors:

```text
「今後のキャリア形成に関する人事当局・上司の面談や助言があるか」
```

```text
「適切かつ柔軟な業務分担が職場で行われていない」
```

```text
霞ヶ関をホワイト化して、優秀な人材が今後とも霞ヶ関に来てくれるような努力
```

Why it matters for KASUMI: these anchors motivate modeling personnel advice, workload allocation, and retention-related welfare endpoints.

### Business Insider Japan report on Kasumigaseki work-style reform

Source: <https://www.businessinsider.jp/article/225315/>

Search anchors:

```text
霞が関が危機的な状況にあると思っている
```

```text
過労死レベルとされる100時間を超えていた官僚が全体の約4割
```

```text
現状では政策をやろうにも、現場の状況を知り、分析しながらというプロセスができない
```

```text
霞が関のデジタル化の遅れ
```

Why it matters for KASUMI: these anchors motivate the connection between work strain, digital support, analytical capacity, and public-sector performance.

### Nikkei article supplied by the project author

Source: <https://www.nikkei.com/article/DGXZQOUC208FV0Q6A520C2000000/?n_cid=SNSTW005&n_tw=1779599309>

Search anchor from public social preview:

```text
富士通、AIが人事異動案を作成 工数を約98%削減
```

Why it matters for KASUMI: this is used only as adjacent industry context that AI-assisted personnel-transfer planning is becoming technically plausible. KASUMI does not deploy or recommend personnel decisions.

## Automated science and LLM-agent simulation background

### The AI Scientist v1

Source: <https://sakana.ai/ai-scientist/>

Search anchors:

```text
The AI Scientist automates the entire research lifecycle
```

```text
idea generation, literature search, experiment planning, experiment iterations, figure generation, manuscript writing, and reviewing
```

```text
automated peer review process
```

Why it matters for KASUMI: KASUMI adapts this loop to a synthetic public-administration task.

### Shachi

Source: <https://github.com/sakanaai/shachi>

Search anchors:

```text
Shachi: A Modular, Controllable Framework for LLM-Based Agent-Based Modeling
```

```text
LLM, Tools, Memory, and Configuration
```

```text
reproducible experiments across a variety of social, economic, and cognitive simulation tasks
```

Why it matters for KASUMI: KASUMI exposes a compact public-service environment surface inspired by modular LLM-based ABM.

### EconAgent

Source: <https://arxiv.org/abs/2310.10436>

Search anchors:

```text
Agent-based modeling (ABM) emerging as a prominent bottom-up simulation paradigm
```

```text
large language model-empowered agent with human-like characteristics for macroeconomic simulation
```

```text
memory module
```

Why it matters for KASUMI: EconAgent is an adjacent example of using LLM-empowered heterogeneous agents to study macro-level social/economic behavior.
