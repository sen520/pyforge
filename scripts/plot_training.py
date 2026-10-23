"""读取 training_log.jsonl，保存训练指标散点图；不加载模型或占用 GPU。"""

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import sys


METRICS = ('loss', 'eval_loss', 'grad_norm', 'learning_rate', 'epoch')


def read_metrics(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines(keepends=True)
    series = {key: {} for key in METRICS}
    last_step = -1
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if number == len(lines) and not line.endswith(('\n', '\r')):
                print('提示：忽略尚未写完的最后一行，可稍后重新画图。', file=sys.stderr)
                break
            raise ValueError(f'日志第 {number} 行不是合法 JSON') from exc
        if not isinstance(row, dict):
            raise ValueError(f'日志第 {number} 行必须为对象')
        step = row.get('step')
        if type(step) is not int or step < 0:
            raise ValueError(f'日志第 {number} 行缺少非负整数 step')
        if step < last_step:
            # 从早期检查点恢复时，旧分支后续点不应混进当前曲线。
            series = {key: {s: v for s, v in points.items() if s <= step} for key, points in series.items()}
        last_step = step
        for key in METRICS:
            if key not in row:
                continue
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'日志第 {number} 行 {key} 不是有限数值')
            series[key][step] = value
    result = {key: sorted(points.items()) for key, points in series.items()}
    if not any(result.values()):
        raise ValueError('日志中没有可绘制的指标')
    return result


def rolling_mean(points, window):
    values = deque()
    total = 0.0
    result = []
    for step, value in points:
        values.append(value)
        total += value
        if len(values) > window:
            total -= values.popleft()
        if len(values) == window:
            result.append((step, total / window))
    return result


def loss_trend_figure(plt, metrics, name, window):
    fig, axis = plt.subplots(figsize=(12, 6), layout='constrained')
    points = metrics['loss']
    if points:
        axis.plot(*zip(*points), color='#94a3b8', alpha=0.22, linewidth=0.7, label='Raw train loss')
        blocks = [points[i:i + window] for i in range(0, len(points), window)]
        complete = [block for block in blocks if len(block) == window]
        means = [(sum(s for s, _ in b) / len(b), sum(v for _, v in b) / len(b)) for b in complete]
        if means:
            axis.plot(*zip(*means), color='#1d4ed8', linewidth=3, marker='o', markersize=5,
                label=f'Train mean / {window} logged points')
            first, last = means[0][1], means[-1][1]
            subtitle = f'Complete-block train mean: {first:.4f} -> {last:.4f}'
        else:
            subtitle = f'Waiting for {window} train points; currently {len(points)}'
        if blocks and len(blocks[-1]) < window:
            tail = blocks[-1]
            axis.scatter(sum(s for s, _ in tail) / len(tail), sum(v for _, v in tail) / len(tail),
                s=70, marker='D', facecolors='none', edgecolors='#1d4ed8',
                label=f'Partial block ({len(tail)}/{window} points)', zorder=4)
    else:
        subtitle = 'No train loss logged'
    validation = metrics['eval_loss']
    if validation:
        axis.plot(*zip(*validation), color='#dc2626', linewidth=2.2, marker='s', markersize=6,
            label='Validation loss (unsmoothed)', zorder=5)
        best = min(validation, key=lambda p: p[1])
        subtitle += f' | Best validation: {best[1]:.4f} at step {best[0]}'
    axis.set(title=subtitle, xlabel='Optimizer step', ylabel='Loss', ylim=(0, None))
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False, loc='best')
    fig.suptitle('Loss trend | ' + name, fontsize=16, fontweight='bold')
    return fig


def plot(log, output=None, window=0, dpi=160, style='scatter'):
    if window < 0 or dpi <= 0:
        raise ValueError('window 必须非负，dpi 必须为正')
    log = Path(log)
    if log.is_dir():
        log = log / 'training_log.jsonl'
    if style not in ('scatter', 'trend'):
        raise ValueError('未知图表样式')
    if style == 'trend' and window == 0:
        window = 20
    filename = 'loss_trend.png' if style == 'trend' else 'training_scatter.png'
    output = Path(output) if output else log.parent / 'plots' / filename
    if output.suffix.lower() not in ('.png', '.svg', '.pdf'):
        raise ValueError('输出文件必须为 .png、.svg 或 .pdf')
    metrics = read_metrics(log)
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('请先安装绘图依赖：python -m pip install -e ".[plot]"') from exc

    if style == 'trend':
        with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False}):
            fig = loss_trend_figure(plt, metrics, log.parent.name, window)
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                fig.savefig(output, dpi=dpi)
            finally:
                plt.close(fig)
        return {'output': str(output.resolve()), 'style': style, 'window': window,
            'points': {k: len(v) for k, v in metrics.items()}}

    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        for axis, key, title, color in zip(axes.flat,
                ('loss', 'grad_norm', 'learning_rate', 'epoch'),
                ('Loss', 'Gradient norm (before clipping)', 'Learning rate', 'Epoch'),
                ('#2563eb', '#7c3aed', '#059669', '#d97706')):
            points = metrics[key]
            if points:
                axis.scatter(*zip(*points), s=12, alpha=0.55, color=color, linewidths=0,
                    label='Train loss' if key == 'loss' else key, rasterized=True)
                if window > 1 and key in ('loss', 'grad_norm'):
                    trend = rolling_mean(points, window)
                    if trend:
                        axis.plot(*zip(*trend), color='#111827', linewidth=1.5,
                            label=f'{window}-point moving mean')
            if key == 'loss' and metrics['eval_loss']:
                axis.scatter(*zip(*metrics['eval_loss']), s=50, marker='x', color='#dc2626',
                    linewidths=1.6, label='Validation loss', zorder=4)
            if not points and not (key == 'loss' and metrics['eval_loss']):
                axis.text(0.5, 0.5, 'No logged data', ha='center', va='center', transform=axis.transAxes)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(loc='best', frameon=False, fontsize=8)
            axis.set(title=title, xlabel='Optimizer step', ylabel='Loss' if key == 'loss' else key)
            axis.grid(alpha=0.18)
            axis.set_axisbelow(True)
            if key == 'learning_rate':
                axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        fig.suptitle('Training metrics | ' + log.parent.name, fontsize=15, fontweight='bold')
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(output, dpi=dpi)
        finally:
            plt.close(fig)
    return {'output': str(output.resolve()), 'points': {k: len(v) for k, v in metrics.items()}, 'window': window}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', help='训练输出目录或 training_log.jsonl 文件')
    parser.add_argument('--output', help='输出 .png/.svg/.pdf；默认保存到运行目录 plots/training_scatter.png')
    parser.add_argument('--window', type=int, default=0, help='可选移动平均窗口；默认 0，只画散点')
    parser.add_argument('--dpi', type=int, default=160)
    parser.add_argument('--style', choices=['scatter', 'trend'], default='scatter', help='trend 突出 loss 分段平均趋势，默认每 20 点一段')
    args = parser.parse_args()
    try:
        print(json.dumps(plot(args.log, args.output, args.window, args.dpi, args.style), ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'错误：{exc}\n')


if __name__ == '__main__':
    main()
