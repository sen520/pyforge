"""归档轻量训练/预测记录到 reports，不复制权重、优化器或环境文件。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


RUN_FILES = ('run_manifest.json', 'run_status.json', 'metrics.json', 'training_log.jsonl')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(run, output, predictions=None, evaluation=None):
    run, output = Path(run), Path(output)
    if output.exists():
        raise ValueError('归档目录已存在，拒绝覆盖')
    if evaluation and not predictions:
        raise ValueError('提供评估报告时必须同时提供原始预测')
    files = {name: run / name for name in RUN_FILES}
    if any(not path.is_file() for path in files.values()):
        raise ValueError('运行缺少完整的训练清单、状态、汇总或日志')
    manifest = json.loads(files['run_manifest.json'].read_text(encoding='utf-8'))
    status = json.loads(files['run_status.json'].read_text(encoding='utf-8'))
    if status.get('status') not in ('completed', 'stopped'):
        raise ValueError('请在训练完成或正常暂停后归档')
    prediction_count = None
    if predictions:
        predictions = Path(predictions)
        sidecar = predictions.with_suffix('.manifest.json')
        provenance = json.loads(sidecar.read_text(encoding='utf-8'))
        if provenance.get('status') != 'completed' or provenance.get('predictions_sha256') != sha(predictions):
            raise ValueError('预测未完成或文件哈希不匹配')
        if provenance.get('run_manifest_sha256') != sha(files['run_manifest.json']):
            raise ValueError('预测不属于指定训练运行')
        rows = [json.loads(line) for line in predictions.read_text(encoding='utf-8').splitlines()]
        if not rows or len({row['id'] for row in rows}) != len(rows) or len(rows) != provenance['count']:
            raise ValueError('预测 ID 重复、为空或数量不符')
        prediction_count = len(rows)
        files.update({'predictions.jsonl': predictions, 'predictions.manifest.json': sidecar})
    if evaluation:
        evaluation = Path(evaluation)
        report = json.loads(evaluation.read_text(encoding='utf-8'))
        if report['predictions_sha256'] != sha(predictions) or report['suite_sha256'] != provenance['suite_sha256']:
            raise ValueError('执行报告与预测或题库哈希不匹配')
        files['evaluation.json'] = evaluation
    archive = {'schema_version': 1, 'run': run.name, 'model_revision': manifest['model_revision'],
        'data_sha256': manifest['data_sha256'], 'prediction_count': prediction_count,
        'evaluation_status': 'completed' if evaluation else 'not_archived',
        'files': {name: {'source': path.as_posix(), 'sha256': sha(path)} for name, path in files.items()},
        'exporter_sha256': sha(__file__),
        'weights_included': False, 'note': '用于审计和绘图，不包含完整恢复训练所需的检查点或模型权重'}
    output.mkdir(parents=True)
    for name, path in files.items():
        shutil.copyfile(path, output / name)
        if sha(output / name) != archive['files'][name]['sha256']:
            raise RuntimeError('归档过程中源文件发生变化：' + name)
    (output / 'archive_manifest.json').write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--predictions')
    parser.add_argument('--evaluation')
    args = parser.parse_args()
    result = export(args.run, args.output, args.predictions, args.evaluation)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'files': len(result['files']),
        'prediction_count': result['prediction_count'], 'evaluation_status': result['evaluation_status']}, ensure_ascii=False, indent=2))
