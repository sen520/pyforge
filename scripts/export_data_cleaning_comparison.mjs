// 使用 Codex 附带 artifact-tool 导出工作簿；不修改原始预测。
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const args = Object.fromEntries(Array.from({ length: (process.argv.length - 2) / 2 }, (_, i) =>
  [process.argv[2 + i * 2].replace(/^--/, ''), process.argv[3 + i * 2]]));
if (!args.input || !args.output || !args.runtime) throw new Error('需要 --input --output --runtime');
const require = createRequire(path.join(path.resolve(args.runtime), 'loader.cjs'));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const payload = JSON.parse(await fs.readFile(args.input, 'utf8'));
const workbook = Workbook.create();
const overview = workbook.worksheets.add('效果汇总');
const sheets = {
  validation: workbook.worksheets.add('验证集对比'),
  test: workbook.worksheets.add('测试集对比'),
};
const colors = { ink: '#202C43', muted: '#657084', line: '#C7D0DD', header: '#283954' };
const headers = ['序号', '题号', '场景', '中文需求', '基础严格', '微调严格', '基础诊断', '微调诊断',
  '诊断变化', '参考JSON', '基础完整回答', '微调完整回答', '基础问题', '微调问题',
  '操作组合', '基础token', '微调token', '基础截断', '微调截断', '严格变化'];
const widths = [54, 300, 210, 420, 100, 100, 100, 100, 100, 520, 520, 520, 330, 330, 280, 90, 90, 90, 90, 100];
const safe = value => typeof value === 'string' && value.startsWith('=') ? "'" + value : value;
const column = index => {
  let out = '';
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) out = String.fromCharCode(65 + (n - 1) % 26) + out;
  return out;
};
function header(sheet, range) {
  sheet.getRange(range).format = { fill: colors.header,
    font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
    horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true };
}
function baseStyle(sheet, range) {
  sheet.showGridLines = false;
  sheet.getRange(range).format = { font: { name: 'Arial', size: 10, color: colors.ink },
    verticalAlignment: 'center', rowHeightPx: 26 };
}
function lineCount(text, width) {
  return String(text ?? '').split('\n').reduce((count, line) => {
    const pixels = [...line].reduce((n, c) => n + (c.charCodeAt(0) > 255 ? 13 : 7), 0);
    return count + Math.max(1, Math.ceil(pixels / (width - 16)));
  }, 0);
}

for (const [split, sheet] of Object.entries(sheets)) {
  const rows = payload.results[split].rows;
  const end = rows.length + 5;
  baseStyle(sheet, `A1:T${end}`);
  sheet.getRange('A2').values = [[split === 'test' ? '测试集：基础模型与微调模型逐题对比' : '验证集：基础模型与微调模型逐题对比']];
  sheet.getRange('A2').format.font = { name: 'Arial', size: 15, bold: true, color: colors.ink };
  const source = payload.provenance.splits[split];
  sheet.getRange('A3').values = [[`来源：${source.reference_path}；基础预测：${source.before_predictions}；微调预测：${source.after_predictions}`]];
  sheet.getRange('A3').format.font = { name: 'Arial', size: 10, italic: true, color: colors.muted };
  sheet.getRange('A5:T5').values = [headers];
  const matrix = rows.map((r, i) => [i + 1, r.id, r.group, r.prompt,
    r.strict_before.status, r.strict_after.status, r.diagnostic_before.status, r.diagnostic_after.status,
    r.diagnostic_change, r.reference, r.before_response, r.after_response,
    r.diagnostic_before.error || (r.diagnostic_before.correct ? '' : '配置与参考答案不同'),
    r.diagnostic_after.error || (r.diagnostic_after.correct ? '' : '配置与参考答案不同'),
    r.operations, r.before_tokens ?? null, r.after_tokens ?? null,
    r.before_truncated == null ? '未记录' : r.before_truncated ? '是' : '否',
    r.after_truncated == null ? '未记录' : r.after_truncated ? '是' : '否', r.strict_change].map(safe));
  for (const row of matrix) if (row.some(value => typeof value === 'string' && value.length > 32767)) {
    throw new Error('回答超过 Excel 单元格上限，拒绝静默截断');
  }
  sheet.getRange(`A6:T${end}`).values = matrix;
  sheet.tables.add(`A5:T${end}`, true, split === 'test' ? 'TestComparison' : 'ValidationComparison');
  header(sheet, 'A5:T5');
  sheet.getRange('A5:T5').format.rowHeightPx = 40;
  widths.forEach((width, i) => { sheet.getRange(`${column(i)}1:${column(i)}${end}`).format.columnWidthPx = width; });
  sheet.getRange(`D6:D${end}`).format.wrapText = true;
  sheet.getRange(`J6:O${end}`).format.wrapText = true;
  sheet.getRange(`D6:D${end}`).format.verticalAlignment = 'top';
  sheet.getRange(`J6:O${end}`).format.verticalAlignment = 'top';
  for (let i = 0; i < matrix.length; i++) {
    const maxLines = Math.max(...[3, 9, 10, 11, 12, 13, 14].map(c => lineCount(matrix[i][c], widths[c])));
    sheet.getRange(`A${i + 6}:T${i + 6}`).format.rowHeightPx = Math.min(500, Math.max(62, maxLines * 15 + 12));
  }
  sheet.getRange(`A6:A${end}`).setNumberFormat('0');
  sheet.getRange(`P6:Q${end}`).setNumberFormat('0');
  sheet.getRange(`E6:I${end}`).format.horizontalAlignment = 'center';
  sheet.getRange(`R6:T${end}`).format.horizontalAlignment = 'center';
  sheet.getRange(`I6:I${end}`).conditionalFormats.add('containsText', {
    text: '改善', format: { fill: '#E7F3EF', font: { color: '#0C6548' } } });
  sheet.getRange(`I6:I${end}`).conditionalFormats.add('containsText', {
    text: '退步', format: { fill: '#FBE9E8', font: { bold: true, color: '#A12929' } } });
  sheet.getRange(`E6:H${end}`).conditionalFormats.add('containsText', {
    text: '错误', format: { fill: '#FFF1E9', font: { color: '#9C4520' } } });
  sheet.freezePanes.freezeRows(5);
  sheet.freezePanes.freezeColumns(2);
}

baseStyle(overview, 'A1:H32');
overview.tabColor = '#283954';
overview.getRange('A2').values = [['数据清洗模型效果对比']];
overview.getRange('A2').format.font = { name: 'Arial', size: 16, bold: true, color: colors.ink };
overview.getRange('A3').values = [['之前：同一1.5B基础模型（不加载LoRA）；现在：本次训练后的LoRA模型。']];
overview.getRange('A3').format.font = { name: 'Arial', size: 10, italic: true, color: colors.muted };
overview.getRange('A3:H3').format.borders = { bottom: { style: 'thin', color: colors.line } };
overview.getRange('A5:H5').values = [['集合', '评分口径', '题数', '基础正确', '微调正确', '基础正确率', '微调正确率', '提升（百分点）']];
header(overview, 'A5:H5');
overview.getRange('A5:H5').format.rowHeightPx = 38;
const summaryCases = [['validation', '严格'], ['validation', '去外壳诊断'], ['test', '严格'], ['test', '去外壳诊断']];
for (let i = 0; i < summaryCases.length; i++) {
  const [split, method] = summaryCases[i];
  const name = sheets[split].name;
  const row = i + 6;
  const end = payload.results[split].rows.length + 5;
  const before = method === '严格' ? 'E' : 'G';
  const after = method === '严格' ? 'F' : 'H';
  overview.getRange(`A${row}:B${row}`).values = [[split === 'test' ? '测试集' : '验证集', method]];
  overview.getRange(`C${row}:H${row}`).formulas = [[
    `=COUNTA('${name}'!$B$6:$B$${end})`,
    `=COUNTIFS('${name}'!$${before}$6:$${before}$${end},"正确")`,
    `=COUNTIFS('${name}'!$${after}$6:$${after}$${end},"正确")`,
    `=D${row}/C${row}`, `=E${row}/C${row}`, `=(G${row}-F${row})*100`,
  ]];
}
overview.getRange('C6:E9').setNumberFormat('#,##0');
overview.getRange('F6:G9').setNumberFormat('0.0%');
overview.getRange('H6:H9').setNumberFormat('0.0');
overview.getRange('A13:G13').values = [['集合', '基础格式合格率', '微调格式合格率', '诊断改善题数', '诊断退步题数', '诊断均正确', '诊断均错误']];
header(overview, 'A13:G13');
overview.getRange('A13:G13').format.rowHeightPx = 38;
for (const [i, split] of ['validation', 'test'].entries()) {
  const row = i + 14;
  const name = sheets[split].name;
  const end = payload.results[split].rows.length + 5;
  overview.getRange(`A${row}`).values = [[split === 'test' ? '测试集' : '验证集']];
  overview.getRange(`B${row}:G${row}`).formulas = [[
    `=(COUNTIFS('${name}'!E6:E${end},"正确")+COUNTIFS('${name}'!E6:E${end},"配置不符"))/COUNTA('${name}'!B6:B${end})`,
    `=(COUNTIFS('${name}'!F6:F${end},"正确")+COUNTIFS('${name}'!F6:F${end},"配置不符"))/COUNTA('${name}'!B6:B${end})`,
    ...['改善', '退步', '均正确', '均错误'].map(label => `=COUNTIFS('${name}'!I6:I${end},"${label}")`),
  ]];
}
overview.getRange('B14:C15').setNumberFormat('0.0%');
overview.getRange('D14:G15').setNumberFormat('0');
overview.getRange('A18').values = [['评分说明']];
overview.getRange('A18').format.font.bold = true;
const notes = [
  '严格评分：原始回答必须直接是合法JSON，并与参考配置匹配；Markdown代码块不算通过。',
  '去外壳诊断：仅去掉完整包裹回答的一个JSON代码块，再用相同规则比较；不改写字段或答案。',
  '严格结果与诊断结果分开显示，诊断不替换原始预测或正式评分。',
  '配置匹配忽略对象键顺序和空白，保留数组顺序和值类型；不是执行结果正确率。',
  '逐题表可筛选“诊断变化”和错误；完整参考答案、原始回答保留在J—L列。',
];
notes.forEach((note, i) => { overview.getRange(`A${i + 19}`).values = [[note]]; });
overview.getRange('A26').values = [[`运行：${payload.provenance.run}`]];
overview.getRange('A27').values = [[`来源：${path.dirname(payload.provenance.splits.test.reference_path)}（validation.jsonl、test.jsonl）`]];
overview.getRange('A28').values = [[`原始预测：${path.dirname(payload.provenance.splits.test.before_predictions)}（两种模型各两份JSONL，完整哈希见comparison.json）`]];
overview.getRange('A26:A28').format.font = { name: 'Arial', size: 10, color: colors.muted };
[95, 150, 145, 125, 125, 140, 140, 150].forEach((width, i) => {
  overview.getRange(`${column(i)}1:${column(i)}32`).format.columnWidthPx = width;
});

workbook.recalculate();
const metrics = overview.getRange('C6:H9').values;
summaryCases.forEach(([split, method], i) => {
  const summary = payload.results[split].summary;
  const prefix = method === '严格' ? 'strict' : 'diagnostic';
  const expected = [summary.total, summary[`${prefix}_before`].correct, summary[`${prefix}_after`].correct,
    summary[`${prefix}_before`].accuracy, summary[`${prefix}_after`].accuracy,
    (summary[`${prefix}_after`].accuracy - summary[`${prefix}_before`].accuracy) * 100];
  expected.forEach((value, j) => {
    if (typeof metrics[i][j] !== 'number' || Math.abs(metrics[i][j] - value) > 1e-9) {
      throw new Error(`汇总公式核对失败：${i},${j}，实际 ${metrics[i][j]}，预期 ${value}`);
    }
  });
});
console.log((await workbook.inspect({ kind: 'table', range: '效果汇总!A5:H9', include: 'values,formulas',
  tableMaxRows: 5, tableMaxCols: 8, maxChars: 2200 })).ndjson);
console.log((await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options: { useRegex: true, maxResults: 20 }, summary: 'formula error scan' })).ndjson);
const previewDir = path.join(path.dirname(args.output), '.previews');
await fs.mkdir(previewDir, { recursive: true });
for (const [name, range, file] of [['效果汇总', 'A1:H24', 'summary.png'],
  ['验证集对比', 'A1:I8', 'validation.png'], ['测试集对比', 'A1:I8', 'test.png'],
  ['测试集对比', 'J5:L7', 'answers.png']]) {
  const image = await workbook.render({ sheetName: name, range, scale: 1, format: 'png' });
  await fs.writeFile(path.join(previewDir, file), new Uint8Array(await image.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(args.output);
console.log(`Excel exported: ${args.output}`);
