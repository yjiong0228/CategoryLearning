"""将一个会话目录内的录音转写为 iSession,iTrial,text CSV。

从仓库根目录运行 python -m src.audio_to_word --help。配置与示例见 README。
API 尚未配置；预览不需要 SDK 或凭据。仅支持纯数字试次文件名（如 12.wav）。
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile
import time
from typing import Callable

# 等申请新 API 后配置；不要将密钥写入源码。
API_KEY = ""
API_BASE_URL = ""
MODEL_NAME = ""
COLUMNS = ("iSession", "iTrial", "text")
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a"}
DEFAULT_PROMPT = (
    "这是类别学习实验中被试的口头报告。请忠实转写实际说出的内容，保留否定、"
    "犹豫、自我修正、比较关系和数字，并添加适当标点。不要推断类别规则，"
    "不要补全未提及的特征，不要根据实验背景改写回答。只输出转写文本。"
)


EXPERIMENT_CONTEXTS = {
    "exp123": (
        "刺激涉及身体部位的长度，包括头、脖子、腿、尾巴、四肢、躯干等。"
        "口头报告可能使用这些部位的其他称呼，或比较部位之间的长短、大小关系。"
        "请保留被试实际使用的称呼，不要统一替换部位名称。"
    ),
    "exp4": (
        "刺激涉及带颜色器官的长度，器官颜色包括绿色、黄色、粉色、蓝色。"
        "口头报告可能用颜色指代器官，并描述或比较其长短。"
        "请准确区分颜色名称并保留长度及比较关系，不要将颜色改写成身体部位。"
    ),
    "exp5": (
        "刺激涉及格子的颜色，颜色为黑色或白色。"
        "请准确保留被试提及的格子、黑白颜色，以及实际说出的数量、位置或关系。"
        "不要将黑白颜色描述改写成长度或身体部位描述。"
    ),
}


def resolve_experiment(folder: Path, experiment: str = "auto") -> str:
    """优先使用显式选择，否则从 data/<实验>/ 路径识别；未知任务使用通用背景。"""
    if experiment != "auto":
        if experiment not in (*EXPERIMENT_CONTEXTS, "generic"):
            raise ValueError(f"未知实验：{experiment}")
        return experiment
    parts = folder.absolute().parts
    matches = {parts[i + 1] for i, part in enumerate(parts[:-1])
               if part == "data" and parts[i + 1] in EXPERIMENT_CONTEXTS}
    if len(matches) > 1:
        raise ValueError("路径包含多个实验标识，请用 --experiment 明确指定")
    return next(iter(matches), "generic")


def build_prompt(context: str = "", *, experiment: str = "generic") -> str:
    """实验词汇与补充背景仅辅助辨词，不作为被试报告或正确答案。"""
    if experiment not in (*EXPERIMENT_CONTEXTS, "generic"):
        raise ValueError(f"未知实验：{experiment}")
    backgrounds = [EXPERIMENT_CONTEXTS.get(experiment, ""), context.strip()]
    background = "\n".join(item for item in backgrounds if item)
    return DEFAULT_PROMPT + ("\n实验背景（仅辅助辨词）：\n" + background if background else "")


def discover_audio(folder: Path) -> list[tuple[int, Path]]:
    if not folder.is_dir():
        raise ValueError(f"录音目录不存在：{folder}")
    trials = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        if not path.stem.isascii() or not path.stem.isdecimal() or int(path.stem) < 1:
            raise ValueError(f"无法确定试次编号：{path.name}；需要 1.wav 这样的文件名")
        trial = int(path.stem)
        if trial in trials:
            raise ValueError(f"重复试次 {trial}：{trials[trial].name}, {path.name}")
        trials[trial] = path
    if not trials:
        raise ValueError(f"目录内没有录音：{folder}")
    return sorted(trials.items())


def make_transcriber(prompt: str) -> Callable[[Path], str]:
    """集中封装 DashScope ASR；若新 API 协议不同，只需替换此入口。

    参考 https://www.alibabacloud.com/help/zh/model-studio/qwen-asr-api-reference
    """
    key = os.environ.get("DASHSCOPE_API_KEY") or API_KEY
    url = os.environ.get("DASHSCOPE_BASE_URL") or API_BASE_URL
    model = os.environ.get("DASHSCOPE_MODEL") or MODEL_NAME
    if not all((key, url, model)):
        raise ValueError("API 配置尚未填写：请设置 DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL、DASHSCOPE_MODEL")
    try:
        import dashscope
    except ImportError as exc:
        raise ValueError("转写需要可选依赖：python -m pip install dashscope") from exc
    dashscope.base_http_api_url = url

    def transcribe(path: Path) -> str:
        response = dashscope.MultiModalConversation.call(
            api_key=key, model=model, result_format="message",
            messages=[
                {"role": "system", "content": [{"text": prompt}]},
                {"role": "user", "content": [{"audio": str(path.resolve())}]},
            ],
            asr_options={"language": "zh", "enable_itn": True},
        )
        if response.get("status_code") != 200:
            # 不打印完整响应，避免日志包含请求凭据或录音内容。
            raise RuntimeError(f"ASR 请求失败，HTTP {response.get('status_code')}")
        content = response["output"]["choices"][0]["message"]["content"]
        return "".join(item.get("text", "") for item in content).strip()

    return transcribe


def read_existing(output: Path) -> dict[tuple[int, int], str]:
    rows = {}
    if output.exists():
        with output.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(COLUMNS):
                raise ValueError(f"输出文件列名必须为 {COLUMNS}")
            for row in reader:
                key = (int(row["iSession"]), int(row["iTrial"]))
                if min(key) < 1 or key in rows or not row["text"].strip():
                    raise ValueError(f"已有输出含非法、重复或空白记录：{key}")
                rows[key] = row["text"]
    return rows


def save_rows(output: Path, rows: dict[tuple[int, int], str]) -> None:
    """原子写入，保留严格三列；失败记录不会冒充转录文本。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                         dir=output.parent, delete=False) as stream:
            tmp = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            for (session, trial), text in sorted(rows.items()):
                writer.writerow((session, trial, text))
        tmp.replace(output)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def process_folder(folder: Path, output: Path, session: int,
                   transcribe: Callable[[Path], str], *, resume: bool = False,
                   attempts: int = 3, retry_delay: float = 5) -> None:
    if session < 1 or attempts < 1 or retry_delay < 0:
        raise ValueError("session 和 attempts 必须为正整数，retry_delay 不能为负")
    audio = discover_audio(folder)
    if output.exists() and not resume:
        raise ValueError(f"输出已存在：{output}；核对来源后使用 --resume，或选择新文件")
    rows = read_existing(output)
    for trial, path in audio:
        if (session, trial) in rows:
            continue
        for attempt in range(attempts):
            try:
                text = transcribe(path).strip()
                if not text:
                    raise RuntimeError("ASR 返回空文本，需人工检查录音")
                break
            except Exception:
                if attempt + 1 == attempts:
                    raise RuntimeError(f"转写失败：{path.name}；已成功的记录已保存，可用 --resume 重试") from None
                time.sleep(retry_delay)
        rows[(session, trial)] = text
        save_rows(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="单个被试的单个会话录音目录")
    parser.add_argument("--session", type=int, required=True, help="输出 iSession，明确指定，不从目录名猜测")
    parser.add_argument("--output", type=Path, required=True, help="目标 *_rec.csv；建议先写入 results 下审核")
    parser.add_argument("--experiment", choices=("auto", *EXPERIMENT_CONTEXTS, "generic"), default="auto",
                        help="默认从 data/<实验>/ 路径识别；外部目录可手动指定，generic 使用通用背景")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 补充背景/词汇文件，追加到所选实验背景后")
    parser.add_argument("--resume", action="store_true", help="保留已有会话/试次，仅转写缺失项；须确认同一被试及配置")
    parser.add_argument("--dry-run", action="store_true", help="检查文件与 prompt，不调用 API、不写文件")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5)
    args = parser.parse_args()
    try:
        if args.session < 1 or args.attempts < 1 or args.retry_delay < 0:
            raise ValueError("session、attempts 必须为正，retry-delay 不能为负")
        audio = discover_audio(args.folder)
        experiment = resolve_experiment(args.folder, args.experiment)
        prompt = build_prompt(args.prompt_file.read_text(encoding="utf-8-sig") if args.prompt_file else "",
                              experiment=experiment)
        print(f"实验 prompt：{experiment}")
        if args.output.exists() and not args.resume:
            raise ValueError("输出文件已存在；请选择新文件或明确使用 --resume")
        read_existing(args.output)
        print(f"会话 {args.session}：{len(audio)} 条录音，试次范围 {audio[0][0]}–{audio[-1][0]}")
        if args.dry_run:
            print(f"输出：{args.output}\n列：{','.join(COLUMNS)}\nPrompt：{prompt}")
            return
        process_folder(args.folder, args.output, args.session, make_transcriber(prompt),
                       resume=args.resume, attempts=args.attempts, retry_delay=args.retry_delay)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
