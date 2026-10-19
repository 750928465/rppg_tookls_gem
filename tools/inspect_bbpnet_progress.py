"""Read BBPNet stdout logs and plot completed epochs without touching training.

No Torch, checkpoints or datasets are loaded. The plateau rule is a diagnostic
heuristic, not proof of convergence, accuracy or a request to stop a process.
"""
import argparse
import ast
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo


ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
SUMMARY = re.compile(r'STAGE2 epoch=(\d+) train=(\{[^\r\n]*?\}) valid=(\{[^\r\n]*?\})')
DEFAULT_LOG_DIR = Path('/home/zhoubin/out_rppg/Phys_shifted/logs')


def parse_log(text):
    text = ANSI.sub('', text)
    rows, issues = [], []
    previous = -1
    for match in SUMMARY.finditer(text):
        epoch = int(match[1])
        if epoch <= previous:
            raise ValueError('日志中轮次重复或倒退，可能混入多个实验；请用 --log 指定单次运行的日志。')
        previous = epoch
        try:
            train, valid = ast.literal_eval(match[2]), ast.literal_eval(match[3])
            values = (train['loss'], valid['loss'], valid['sbp_mae'], valid['dbp_mae'])
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('invalid metric')
            row = dict(epoch=epoch + 1, checkpoint_index=epoch,
                       train_loss=float(values[0]), valid_loss=float(values[1]),
                       sbp_mae=float(values[2]), dbp_mae=float(values[3]),
                       score=(float(values[2]) + float(values[3])) / 2,
                       n_states=valid.get('n_states'))
        except (ValueError, SyntaxError, TypeError, KeyError):
            issues.append(f'第 {epoch + 1} 轮指标缺失、非有限或格式异常。')
            continue
        rows.append(row)
    if rows and [r['epoch'] for r in rows] != list(range(1, rows[-1]['epoch'] + 1)):
        issues.append('日志未包含从第 1 轮开始的连续完整记录，不能据此判断早停。')
    if rows:
        states = [r['n_states'] for r in rows]
        if any(type(n) is not int or n <= 0 for n in states) or len(set(states)) != 1:
            issues.append('验证集 n_states 缺失或变化，需核对状态级评价口径。')
    progress = re.findall(r'STAGE2 (?:train|valid|test):[^\r\n]*', text)
    headers = re.findall(r'STAGE2 epoch (\d+)/(\d+); lr=([^\r\n]+)', text)
    return rows, issues, {
        'latest_progress': progress[-1].strip() if progress else None,
        'latest_epoch_header': '/'.join(headers[-1][:2]) if headers else None,
        'latest_lr': headers[-1][2].strip() if headers else None,
    }


def assess(rows, issues, min_epochs=20, patience=10, min_delta=0.1):
    if min_epochs < 1 or patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError('min_epochs/patience 必须为正整数，min_delta 必须为有限非负数。')
    if not rows:
        raise ValueError('尚无完整训练及验证轮次；等待第一轮验证结束后再运行。')
    anchor = rows[0]['score']
    anchor_epoch = rows[0]['epoch']
    wait = 0
    waits = []
    for row in rows:
        improvement = anchor - row['score']
        if improvement > 0 and improvement >= min_delta - 1e-12:
            anchor, anchor_epoch, wait = row['score'], row['epoch'], 0
        elif row is not rows[0]:
            wait += 1
        waits.append(wait)
    best = min(rows, key=lambda r: r['score'])
    if issues:
        status = 'UNRELIABLE_HISTORY'
    elif len(rows) < min_epochs:
        status = 'INSUFFICIENT_HISTORY'
    elif wait >= patience:
        status = 'PLATEAU_CANDIDATE'
    else:
        status = 'KEEP_OBSERVING'
    return dict(status=status, completed_epochs=len(rows), last=rows[-1], best=best,
                min_epochs=min_epochs, patience=patience, min_delta_mmHg=min_delta,
                last_significant_improvement_epoch=anchor_epoch,
                epochs_without_significant_improvement=wait, wait_history=waits,
                issues=issues, rule='Cumulative decrease from last significant improvement; validation only.',
                interpretation='A plateau candidate is not proof of convergence or acceptable BP accuracy.')


def format_report(report, source, progress, tail):
    best, last = report['best'], report['last']
    lines = [f"日志：{source['path']}",
             f"日志修改时间：{source['modified_at']}；读取快照：{source['read_at']}",
             '以下是日志快照，不代表已确认训练进程仍在运行。',
             f"最近轮次提示：{progress['latest_epoch_header'] or '无'}；lr={progress['latest_lr'] or '未知'}",
             f"最近进度：{progress['latest_progress'] or '无'}",
             f"可解析完整轮数：{report['completed_epochs']}；最后完成第 {last['epoch']} 轮",
             '', ' Epoch  Train loss  Valid loss   SBP MAE   DBP MAE    Score (mmHg)']
    for row in tail:
        lines.append(f"{row['epoch']:6d} {row['train_loss']:11.5f} {row['valid_loss']:11.5f}"
                     f" {row['sbp_mae']:9.3f} {row['dbp_mae']:9.3f} {row['score']:10.3f}")
    lines += ['', f"日志内最佳：第 {best['epoch']} 轮（权重编号 {best['checkpoint_index']}），"
              f"score={best['score']:.4f}，SBP MAE={best['sbp_mae']:.4f}，DBP MAE={best['dbp_mae']:.4f} mmHg",
              f"候选规则：至少 {report['min_epochs']} 轮；连续 {report['patience']} 轮无有效改善；"
              f"有效改善为 score 累计下降至少 {report['min_delta_mmHg']:g} mmHg（且必须下降）。",
              f"最近有效改善：第 {report['last_significant_improvement_epoch']} 轮；"
              f"已连续 {report['epochs_without_significant_improvement']} 轮未达到有效改善。"]
    messages = {
        'UNRELIABLE_HISTORY': '记录不完整或指标异常，暂不能判断是否满足早停条件。',
        'INSUFFICIENT_HISTORY': '完整轮数不足，继续观察。',
        'KEEP_OBSERVING': '当前未满足候选早停规则。',
        'PLATEAU_CANDIDATE': '满足候选早停规则，可结合两项 MAE 曲线评估是否结束训练。',
    }
    lines += [f"判断：{report['status']} — {messages[report['status']]}"]
    lines += [f'注意：{issue}' for issue in report['issues']]
    lines += ['score=(验证 SBP MAE + 验证 DBP MAE)/2；按受试者 before/after 状态聚合。',
              '平台期不等于准确度达标或已找到最优解。脚本不会停止训练，不使用测试集。',
              '当前 Trainer 中断后不会自动导出最终 history/图表或运行测试；已保存权重可单独测试。']
    return '\n'.join(lines) + '\n'


def plot_progress(rows, report, path, source_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [r['epoch'] for r in rows]
    scores = [r['score'] for r in rows]
    best_so_far, running = [], float('inf')
    for score in scores:
        running = min(running, score)
        best_so_far.append(running)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, [r['train_loss'] for r in rows], '.-', label='Train')
    axes[0, 0].plot(epochs, [r['valid_loss'] for r in rows], '.-', label='Validation')
    axes[0, 0].set(title='Train / validation loss', ylabel='Loss')
    for key, label in [('sbp_mae', 'SBP'), ('dbp_mae', 'DBP')]:
        axes[0, 1].plot(epochs, [r[key] for r in rows], '.-', label=label)
    axes[0, 1].set(title='Validation: state-level BP error', ylabel='MAE (mmHg)')
    axes[1, 0].plot(epochs, scores, '.-', label='Mean SBP/DBP MAE')
    axes[1, 0].plot(epochs, best_so_far, '--', label='Best so far')
    best = report['best']
    axes[1, 0].scatter(best['epoch'], best['score'], marker='*', s=140,
                       color='darkorange', zorder=5, label=f"Best: epoch {best['epoch']}")
    axes[1, 0].set(title='Validation checkpoint-selection score', ylabel='MAE (mmHg)')
    axes[1, 1].step(epochs, report['wait_history'], where='mid', label='No significant improvement')
    axes[1, 1].axhline(report['patience'], color='darkorange', linestyle='--', label='Patience')
    axes[1, 1].set(title=f"Improvement threshold: {report['min_delta_mmHg']:g} mmHg", ylabel='Consecutive epochs')
    for ax in axes.flat:
        ax.set_xlabel('Completed epoch (1-based)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"BBPNet progress | {report['status']}\n{source_name}", fontsize=11)
    fig.text(.5, .012, 'Validation only. A plateau is not proof of convergence or acceptable accuracy. No process is stopped.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .95))
    for suffix in ('.png', '.svg'):
        fig.savefig(path.with_suffix(suffix), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--log', type=Path, help='Single-run stdout log, or tmux capture with joined lines')
    source.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR,
                        help='Select latest modified bbpnet_*.log; printed for verification')
    parser.add_argument('--output', type=Path, help='Dedicated report directory (not training weights)')
    parser.add_argument('--min-epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min-delta', type=float, default=.1, help='Absolute validation-score change in mmHg')
    parser.add_argument('--last', type=int, default=10, help='Number of completed epochs to print')
    args = parser.parse_args()
    if args.last < 1:
        parser.error('--last must be positive')
    try:
        log = args.log
        if log is None:
            logs = list(args.log_dir.expanduser().glob('bbpnet_*.log'))
            if not logs:
                parser.error('没有找到 bbpnet_*.log；请用 --log 指定训练日志或 tmux 导出文本。')
            log = max(logs, key=lambda p: p.stat().st_mtime)
        log = log.expanduser().resolve()
        # Bound the read to a snapshot even while tee continues appending.
        with log.open('rb') as f:
            stat = os.fstat(f.fileno())
            text = f.read(stat.st_size).decode('utf-8', errors='replace')
        rows, issues, progress = parse_log(text)
        report = assess(rows, issues, args.min_epochs, args.patience, args.min_delta)
        tz = ZoneInfo('Asia/Shanghai')
        source_info = dict(path=str(log), size_bytes=stat.st_size,
                           modified_at=datetime.fromtimestamp(stat.st_mtime, tz).isoformat(timespec='seconds'),
                           read_at=datetime.now(tz).isoformat(timespec='seconds'))
        summary = format_report(report, source_info, progress, rows[-args.last:])
        print(summary, flush=True)
        output = (args.output or log.parent / 'progress' / log.stem).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        prefix = output / 'BBPNet_STAGE2_progress'
        # These names are independent of the Trainer's history/checkpoint outputs.
        targets = [prefix.with_suffix(s) for s in ('.txt', '.json', '.png', '.svg')]
        if any(p.resolve() == log for p in targets):
            raise ValueError('报告路径与输入日志冲突，请更换 --output。')
        prefix.with_suffix('.txt').write_text(summary, encoding='utf-8')
        prefix.with_suffix('.json').write_text(json.dumps(
            dict(source=source_info, progress=progress, assessment=report, epochs=rows),
            ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        plot_progress(rows, report, prefix, log.name)
        for path in targets:
            print(f'已保存：{path.resolve()}')
    except (OSError, ValueError) as exc:
        parser.exit(1, f'无法完成日志分析：{exc}\n')


if __name__ == '__main__':
    main()
