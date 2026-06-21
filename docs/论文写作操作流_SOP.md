# 论文写作操作流 SOP（设计 → 画图 → 编写 → 验收）

> 面向 RL/CV 方向研究，以 **nature-skills 为主干**，按需补 paperspine 的"审计"环节。
> 设计原则：可核查、可复现、零编造（学术不端红线）。

---

## 一、两套工具的重复度与定位

**结论**：两套在"文献/引用/写作/LaTeX/翻译"重叠；但各有对方完全没有的能力。
- nature-skills 独有且命中硬环节：**画图、润色、审稿人视角、回复审稿、数据声明**。
- paperspine 独有且有价值：**流程纪律 + 完整性/防造假审计（audit）**。
- ⚠️ **禁用 `paper-spine-humanize`**：它是"降低 AIGC 检测率"的反检测工具，与学术伦理红线冲突。

| 流程环节 | nature-skills | paper-spine | 关系 |
|---|---|---|---|
| 选题·结构 | `nature-writing` | `research`+`intake`+`build` | 重叠 |
| 文献检索·核查 | `nature-academic-search` | `research`+`citation` | 重叠（nature 多源+arXiv，可核查更强） |
| 引用插入 | `nature-citation` | `citation` | 重叠 |
| **画图** | `nature-figure` | ❌ 无（latex 仅摆放） | **nature 独有** |
| 写作起草 | `nature-writing` | `build`/`rewrite` | 重叠 |
| **润色** | `nature-polishing` | ❌ 无 | **nature 独有** |
| LaTeX 装配 | `nature-polishing`(布局) | `latex` | 重叠 |
| 中英对照 | `nature-reader` | `translate` | 重叠 |
| **验收·评审视角** | `nature-reviewer` | ❌ 无 | **nature 独有** |
| **验收·防造假审计** | ❌ 无 | `audit` | **paperspine 独有** |
| 投稿后·回复 | `nature-response` | ❌ 无 | **nature 独有** |
| 数据声明 | `nature-data` | ❌ 无 | **nature 独有** |
| AI 反检测 | ❌（刻意不做） | `humanize` ⚠️ | **禁用** |

---

## 二、操作流（10 步，每步：用什么 → 产出 → 验证点）

| 阶段 | 用什么 | 产出 | 验证点 |
|---|---|---|---|
| ① 立项·动机 | `nature-writing` | 一句话 motivation、贡献清单、章节蓝图 | motivation 窄而具体，不灌水多 claim |
| ② 文献 | `nature-academic-search` | 可核查文献库（作者-年份-题目-期刊/会议+PDF） | 逐条能点开原文，零编造 |
| ③ 画图 | `nature-figure`（Python，必要时 R） | 投稿级多面板图 + 可复现脚本（含种子） | 重跑脚本图一致；矢量 PDF/TIFF 达标 |
| ④ 起草 | `nature-writing` | 各章初稿 | 每段有依据，无悬空 claim |
| ⑤ 润色 | `nature-polishing` | Nature 级英文 + LaTeX 布局修复 | 术语一致、无浮夸、编译无 overfull |
| ⑥ 引用落地 | `nature-citation` | 文中引用 + RIS/BibTeX 导出 | 引用-论点对应表，零编造 |
| ⑦ 验收·防造假 | `paper-spine-audit`（唯一需新装） | `integrity_audit.md` | 无 BLOCKED 项才放行 |
| ⑧ 验收·过审 | `nature-reviewer` | 3 份模拟 referee + 综述 | 按报告补漏洞/前提/对照实验 |
| ⑨ 投稿后 | `nature-response` | 逐点回复信 | 每条对应 reviewer + 修改位置 |
| ⑩ 数据声明 | `nature-data` | Data/Code Availability + FAIR 清单 | 仓库/accession 真实可达 |

> **关键提醒**：画图（③）paperspine 全套都救不了——它只摆放现成图片，不画图。`nature-figure` 不可替代。

---

## 三、安装清单（决定后再执行）

nature-skills 已全部就绪，无需安装。paperspine 按形态二选一：

**形态 A（推荐）— nature 主干 + 只补 audit：**
```bash
npx -y skills add wubing2023/paperspine@paper-spine-audit
```

**形态 B — paperspine 当端到端总驱动（/paperspine 一键）+ nature-figure 画图：**
```bash
for s in paper-spine paper-spine-intake paper-spine-research \
         paper-spine-build paper-spine-latex paper-spine-audit; do
  npx -y skills add "wubing2023/paperspine@$s"
done
# 注意：跳过 paper-spine-humanize（反检测，禁用）
```

> 安全提示：`npx skills add` 全局安装会被 Claude Code 安全分类器拦截（未受信第三方代码常驻）。
> 需自行在终端执行，或显式开放 Bash 权限规则后由助手执行。
> paperspine 配置缺失时会尝试运行 `launch_paperspine_ui.ps1` 并请求提权（面向 Windows/Codex；macOS 触发不到）。

---

## 四、来源与信任备注

- paperspine 仓库：`wubing2023/paperspine`（installs 约 250–400，单作者，**不在 vercel-labs/anthropics/microsoft 可信源清单**）。
- find-skills 机制：Vercel Skills CLI（https://skills.sh/ ），`npx skills find/add/check/update`。
- 评估日期：2026-06-15。
