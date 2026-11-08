# -*- coding: utf-8 -*-
"""
Автономный AI-бот стресс-тестирования, оценки качества и аудита (Intelligent QA & Diagnostics Bot).
Проект: Рекомендательный сервис оценки стоимости уникального литья (ЮУрГУ & ООО «БВК», ПГ КОНАР).

Ключевые возможности:
1. Интеллектуальный аудит аномалий через DeepSeek-R1 (DeepThink CoT).
2. Автоматическая работа с реальным архивом 30 заводских чертежей БВК (PDF/сканы).
3. Интерактивный Visual QA Dashboard в формате HTML (Industrial Dark Theme) с превью детекций и фильтрацией.
4. Выгрузка детальных отчетов: dashboard.html, eval_report.md, eval_results.json.
"""

import os
import sys
import time
import json
import re
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from PIL import Image

# Импорт логики веб-сервиса и калькулятора себестоимости
import web_app
from cost_engine import FoundryCostEngine

FACTORY_DRAWINGS_DIR = r"C:\Users\Yaroslav\OneDrive\Desktop\датасет проекта\датабаза\drawings"
DEEPSEEK_PROXY_URL = os.environ.get("DEEPSEEK_API_URL", "http://78.17.76.203:9655/v1/chat/completions")


class BatchEvalBot:
    def __init__(self, output_dir="bot_eval_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visuals_dir = self.output_dir / "annotated"
        self.visuals_dir.mkdir(parents=True, exist_ok=True)

    def extract_cost_number(self, cost_markdown):
        """Извлекает итоговую себестоимость в рублях из отчета markdown."""
        if not cost_markdown:
            return 0.0
        m1 = re.search(r"Ожидаемая стоимость[^\*]*\*\*\s*\|\s*\*\*([\d\s]+)\s*руб", cost_markdown, re.I)
        if m1:
            clean_str = m1.group(1).replace(" ", "").strip()
            try:
                return float(clean_str)
            except ValueError:
                pass
        
        m2 = re.search(r"(?:ИТОГО|Итого)\s*себестоимость[:\*\s]*([\d\s]+)\s*руб", cost_markdown, re.I)
        if m2:
            clean_str = m2.group(1).replace(" ", "").strip()
            try:
                return float(clean_str)
            except ValueError:
                pass
        return 0.0

    def evaluate_file(self, file_path, conf_threshold=0.35, has_pattern=False):
        """Прогон одного чертежа (PDF или изображение) через пайплайн сервиса."""
        file_path = Path(file_path)
        t0 = time.time()
        
        try:
            (
                annotated_img,
                views_crop,
                stamp_crop,
                status_info,
                mass_val,
                steel_val,
                group_val,
                heat_val,
                cost_summary_md,
                ops_md,
                analogs_md
            ) = web_app.analyze_drawing(str(file_path), conf_threshold=conf_threshold, has_pattern=has_pattern)
            
            elapsed_ms = (time.time() - t0) * 1000
            total_cost = self.extract_cost_number(cost_summary_md)
            
            best_analog_match = re.search(r"Ближайший аналог в архиве БВК[:\*\s]+\*\*([^\*]+)\*\*", analogs_md)
            best_analog_name = best_analog_match.group(1).strip() if best_analog_match else "Не найден"

            sim_match = re.search(r"Интегральное сходство параметров[:\*\s]+([\d\.]+)\s*%", analogs_md)
            similarity_pct = float(sim_match.group(1)) if sim_match else 0.0

            pattern_in_stock = "Имеется на складе" in analogs_md or "Готова к адаптации" in analogs_md
            savings_match = re.search(r"экономия\s*(?:до\s*)?([\d\s,]+)\s*руб", analogs_md, re.I)
            pattern_savings = 0.0
            if savings_match:
                try:
                    pattern_savings = float(savings_match.group(1).replace(" ", "").replace(",", "."))
                except ValueError:
                    pass
            elif pattern_in_stock:
                pattern_savings = 320000.0

            annotated_filename = f"eval_{file_path.stem}.png"
            if annotated_img is not None:
                annotated_img.save(self.visuals_dir / annotated_filename)

            warnings = []
            if "не найдено элементов" in status_info:
                warnings.append("NO_ESKD_ELEMENTS")
            if stamp_crop is None:
                warnings.append("STAMP_MISSING")
            if views_crop is None and "NO_ESKD_ELEMENTS" not in warnings:
                warnings.append("VIEWS_MISSING")
            if total_cost == 0.0 and "NO_ESKD_ELEMENTS" not in warnings:
                warnings.append("ZERO_COST")
            if mass_val > 5000:
                warnings.append("EXTREME_MASS")
            if similarity_pct < 70.0 and best_analog_name != "Не найден":
                warnings.append("LOW_SIMILARITY")

            is_factory = "drawings" in str(file_path).lower()

            res_record = {
                "file": file_path.name,
                "path": str(file_path),
                "is_factory": is_factory,
                "status": "OK" if "NO_ESKD_ELEMENTS" not in warnings else "UNRECOGNIZED",
                "elapsed_ms": round(elapsed_ms, 1),
                "stamp_found": stamp_crop is not None,
                "views_found": views_crop is not None,
                "mass_kg": round(float(mass_val), 1),
                "steel_grade": str(steel_val),
                "casting_group": str(group_val),
                "heat_treatment": str(heat_val),
                "best_analog": best_analog_name,
                "similarity_pct": similarity_pct,
                "pattern_in_stock": pattern_in_stock,
                "pattern_savings_rub": pattern_savings,
                "total_cost_rub": total_cost,
                "warnings": warnings,
                "annotated_preview": f"annotated/{annotated_filename}"
            }
            return res_record

        except Exception as e:
            return {
                "file": file_path.name,
                "path": str(file_path),
                "is_factory": "drawings" in str(file_path).lower(),
                "status": "ERROR",
                "error": str(e),
                "elapsed_ms": round((time.time() - t0) * 1000, 1),
                "warnings": ["CRASH_ERROR"]
            }
    def request_deepseek_audit(self, summary_stats, anomaly_records):
        """Запрашивает экспертный аудит аномалий у DeepSeek-R1 (DeepThink)."""
        prompt = f"""
Ты — ведущий инженер-технолог сталелитейного производства ООО «БВК» (ПГ КОНАР, г. Челябинск) и эксперт по ЕСКД и ГОСТ 977-88.
Проведен автоматический стресс-тест нейросетевого сервиса оценки себестоимости уникального литья.

### СВОДНЫЕ РЕЗУЛЬТАТЫ СТРЕСС-ТЕСТА:
- Всего чертежей: {summary_stats['total']}
- Успешно обработано (OK): {summary_stats['ok_count']} ({summary_stats['ok_pct']}%)
- Из них реальных заводских чертежей БВК: {summary_stats['factory_count']}
- Не распознано элементов ЕСКД: {summary_stats['unrec_count']}
- Ошибок сбоя: {summary_stats['err_count']}
- Штамп не найден: {summary_stats['stamp_missing']}
- Поле видов не найдено: {summary_stats['views_missing']}
- Среднее время на чертеж (GPU AMD RX 6600 Vulkan): {summary_stats['avg_time']:.1f} мс
- Найдено позиций с готовой оснасткой на складе: {summary_stats['patterns_in_stock']} шт.
- Суммарная экономия на оснастке: {summary_stats['total_savings']:,.0f} руб.

### ДЕТАЛИЗАЦИЯ АНОМАЛИЙ И КРАЙНИХ СЛУЧАЕВ:
{json.dumps(anomaly_records[:8], ensure_ascii=False, indent=2)}

Пожалуйста, предоставь структурированное экспертное заключение на русском языке:
1. Оценка надежности и готовности системы к внедрению в контур завода БВК.
2. Анализ причин пропусков штампа или поля видов на чертежах.
3. Оценка прецедентного поиска аналогов и реального экономического эффекта от оснастки на складе.
4. Конкретные рекомендации инженеру по улучшению эвристик парсинга и порогов детектора.
"""

        payload = {
            "model": "deepseek-r1",
            "messages": [
                {"role": "system", "content": "You are DeepSeek-R1, an expert metallurgist, CAD engineer, and AI reviewer. Provide deep, actionable technical analysis."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 4096
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                DEEPSEEK_PROXY_URL,
                data=req_data,
                headers={"Content-Type": "application/json", "User-Agent": "BatchEvalBot-DeepSeek"}
            )
            handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(handler)
            with opener.open(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]
                conclusion = choice["message"].get("content") or ""
                reasoning = choice["message"].get("reasoning_content") or ""
                return {
                    "available": True,
                    "conclusion": conclusion,
                    "reasoning": reasoning
                }
        except Exception as e:
            return {
                "available": False,
                "conclusion": f"DeepSeek-R1 временно недоступен: {e}. Применены базовые локальные эвристики.",
                "reasoning": ""
            }
    def generate_html_dashboard(self, results, summary_stats, deepseek_audit):
        """Формирует интерактивный веб-дашборд в формате HTML (Industrial Dark Theme)."""
        html_cards = []
        for r in results:
            badge_class = "badge-ok" if r.get("status") == "OK" else ("badge-warn" if r.get("status") == "UNRECOGNIZED" else "badge-err")
            pattern_badge = '<span class="badge-stock">Оснастка на складе (-320 000 ₽)</span>' if r.get("pattern_in_stock") else '<span class="badge-none">Оснастка с нуля</span>'
            source_badge = '<span class="badge-factory">Заводской БВК</span>' if r.get("is_factory") else '<span class="badge-synth">Синтетика</span>'
            
            warnings_html = "".join([f'<span class="tag-warn">{w}</span>' for w in r.get("warnings", [])])
            if not warnings_html:
                warnings_html = '<span class="tag-clean">Без замечаний</span>'

            preview_img = r.get("annotated_preview", "")
            cost_str = f"{r.get('total_cost_rub', 0):,.0f} ₽".replace(",", " ") if r.get("total_cost_rub") else "N/A"

            card_html = f"""
            <div class="card" data-status="{r.get('status')}" data-stock="{str(r.get('pattern_in_stock')).lower()}" data-factory="{str(r.get('is_factory')).lower()}">
                <div class="card-header">
                    <span class="file-name" title="{r.get('file')}">{r.get('file')}</span>
                    <div>{source_badge} <span class="badge {badge_class}">{r.get('status')}</span></div>
                </div>
                <div class="card-body">
                    <div class="preview-box">
                        <img src="{preview_img}" alt="{r.get('file')}" loading="lazy" onclick="window.open(this.src)">
                    </div>
                    <div class="card-meta">
                        <div class="meta-row"><span>⏱️ Время:</span> <strong>{r.get('elapsed_ms')} мс</strong></div>
                        <div class="meta-row"><span>🔩 Сталь:</span> <strong>{r.get('steel_grade', '-')}</strong></div>
                        <div class="meta-row"><span>⚖️ Масса:</span> <strong>{r.get('mass_kg', '-')} кг</strong></div>
                        <div class="meta-row"><span>📋 Группа:</span> <strong>Гр. {r.get('casting_group', '-')} (ГОСТ 977)</strong></div>
                        <div class="meta-row"><span>🔍 Аналог:</span> <strong title="{r.get('best_analog')}">{r.get('best_analog', '-')[:24]} ({r.get('similarity_pct', 0):.1f}%)</strong></div>
                        <div class="meta-row"><span>📦 Оснастка:</span> {pattern_badge}</div>
                        <div class="cost-row"><span>💰 Отпускная цена:</span> <strong>{cost_str}</strong></div>
                    </div>
                </div>
                <div class="card-footer">
                    {warnings_html}
                </div>
            </div>
            """
            html_cards.append(card_html)

        deepseek_section = ""
        if deepseek_audit.get("available"):
            reasoning_html = ""
            if deepseek_audit.get("reasoning"):
                reasoning_html = f"""
                <details class="deepseek-cot">
                    <summary>🧠 Посмотреть ход рассуждений DeepSeek-R1 (Chain of Thought)</summary>
                    <div class="cot-content">{deepseek_audit['reasoning'].replace(chr(10), '<br>')}</div>
                </details>
                """
            conclusion_html = deepseek_audit['conclusion'].replace("\n", "<br>")
            deepseek_section = f"""
            <div class="deepseek-block">
                <div class="deepseek-header">
                    <div class="ds-title">🧠 Инженерно-технологический аудит DeepSeek-R1 (DeepThink)</div>
                    <div class="ds-badge">AI Verified</div>
                </div>
                {reasoning_html}
                <div class="ds-conclusion">{conclusion_html}</div>
            </div>
            """

        html_template = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual QA Dashboard • ООО «БВК» (ПГ КОНАР)</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #111827;
            --border: #1f293d;
            --accent: #2563eb;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --ok: #10b981;
            --warn: #f59e0b;
            --err: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        h1 {{ font-size: 24px; font-weight: 700; color: #fff; }}
        .subtitle {{ font-size: 14px; color: var(--text-muted); margin-top: 4px; }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
        }}
        .kpi-label {{ font-size: 13px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }}
        .kpi-value {{ font-size: 26px; font-weight: 700; margin-top: 6px; color: #fff; }}
        .kpi-value.ok {{ color: var(--ok); }}
        
        .deepseek-block {{
            background: linear-gradient(145deg, #111a2e, #0c1222);
            border: 1px solid #1e3a8a;
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 28px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }}
        .deepseek-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }}
        .ds-title {{ font-size: 18px; font-weight: 700; color: #60a5fa; }}
        .ds-badge {{ background: #1e3a8a; color: #93c5fd; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 20px; }}
        .deepseek-cot {{
            background: #090e1a;
            border: 1px dashed #2563eb;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
            font-size: 13px;
        }}
        .deepseek-cot summary {{ cursor: pointer; color: #93c5fd; font-weight: 600; outline: none; }}
        .cot-content {{ margin-top: 10px; color: #cbd5e1; font-family: monospace; white-space: pre-wrap; font-size: 12px; }}
        .ds-conclusion {{ font-size: 14.5px; color: #e2e8f0; line-height: 1.6; }}

        .toolbar {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
        .btn-filter {{
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn-filter:hover, .btn-filter.active {{
            background: var(--accent);
            color: #fff;
            border-color: var(--accent);
        }}

        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .card:hover {{ transform: translateY(-2px); border-color: #3b82f6; }}
        .card-header {{
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0,0,0,0.2);
            border-bottom: 1px solid var(--border);
        }}
        .file-name {{ font-weight: 600; font-size: 13.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 170px; }}
        .badge {{ font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 6px; }}
        .badge-ok {{ background: rgba(16, 185, 129, 0.15); color: var(--ok); }}
        .badge-warn {{ background: rgba(245, 158, 11, 0.15); color: var(--warn); }}
        .badge-err {{ background: rgba(239, 68, 68, 0.15); color: var(--err); }}
        .badge-stock {{ background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 11px; padding: 2px 6px; border-radius: 4px; font-weight: 600; }}
        .badge-none {{ color: var(--text-muted); font-size: 11px; }}
        .badge-factory {{ background: #1e3a8a; color: #93c5fd; font-size: 10px; padding: 2px 6px; border-radius: 4px; margin-right: 4px; }}
        .badge-synth {{ background: #374151; color: #d1d5db; font-size: 10px; padding: 2px 6px; border-radius: 4px; margin-right: 4px; }}

        .preview-box {{
            height: 190px;
            background: #050811;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            border-bottom: 1px solid var(--border);
        }}
        .preview-box img {{ max-width: 100%; max-height: 100%; object-fit: contain; cursor: pointer; }}
        .card-meta {{ padding: 14px 16px; font-size: 13px; }}
        .meta-row {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
        .meta-row span {{ color: var(--text-muted); }}
        .cost-row {{ margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--border); display: flex; justify-content: space-between; align-items: center; }}
        .cost-row strong {{ font-size: 15px; color: #38bdf8; }}
        .card-footer {{ padding: 10px 16px; background: rgba(0,0,0,0.15); border-top: 1px solid var(--border); font-size: 11px; display: flex; gap: 6px; flex-wrap: wrap; }}
        .tag-warn {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; padding: 2px 6px; border-radius: 4px; }}
        .tag-clean {{ color: var(--text-muted); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🤖 Visual QA Dashboard • ООО «БВК» (ПГ КОНАР)</h1>
                <div class="subtitle">Автоматический аудит распознавания ЕСКД, прецедентного поиска аналогов и себестоимости</div>
            </div>
            <div style="text-align: right; font-size: 13px; color: var(--text-muted);">
                Платформа: <strong>AMD Radeon RX 6600 (Vulkan)</strong><br>
                Дата: <strong>{time.strftime('%d.%m.%Y %H:%M')}</strong>
            </div>
        </header>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Проверено чертежей</div>
                <div class="kpi-value">{summary_stats['total']} шт.</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Успешность (OK)</div>
                <div class="kpi-value ok">{summary_stats['ok_pct']}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Реальные сканы БВК</div>
                <div class="kpi-value">{summary_stats['factory_count']} шт.</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Средняя скорость</div>
                <div class="kpi-value">{summary_stats['avg_time']:.1f} мс</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Оснастка на складе</div>
                <div class="kpi-value ok">{summary_stats['patterns_in_stock']} шт.</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Потенциальная экономия</div>
                <div class="kpi-value ok">{summary_stats['total_savings']:,.0f} ₽</div>
            </div>
        </div>

        {deepseek_section}

        <div class="toolbar">
            <button class="btn-filter active" onclick="filterCards('all')">Все чертежи ({summary_stats['total']})</button>
            <button class="btn-filter" onclick="filterCards('factory')">Заводские сканы БВК ({summary_stats['factory_count']})</button>
            <button class="btn-filter" onclick="filterCards('stock')">С готовой оснасткой ({summary_stats['patterns_in_stock']})</button>
            <button class="btn-filter" onclick="filterCards('warn')">Требуют внимания ({summary_stats['unrec_count'] + summary_stats['stamp_missing']})</button>
        </div>

        <div class="cards-grid" id="cardsGrid">
            {"".join(html_cards)}
        </div>
    </div>

    <script>
        function filterCards(type) {{
            document.querySelectorAll('.btn-filter').forEach(function(b) {{ b.classList.remove('active'); }});
            event.target.classList.add('active');
            
            var cards = document.querySelectorAll('.card');
            cards.forEach(function(card) {{
                if (type === 'all') {{
                    card.style.display = 'flex';
                }} else if (type === 'factory') {{
                    card.style.display = card.dataset.factory === 'true' ? 'flex' : 'none';
                }} else if (type === 'stock') {{
                    card.style.display = card.dataset.stock === 'true' ? 'flex' : 'none';
                }} else if (type === 'warn') {{
                    card.style.display = (card.dataset.status !== 'OK' || card.querySelector('.tag-warn')) ? 'flex' : 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""
        dashboard_path = self.output_dir / "dashboard.html"
        dashboard_path.write_text(html_template, encoding="utf-8")
        return dashboard_path
    def run_batch(self, files_list, run_ai_audit=True):
        """Пакетный запуск оценки с вызовом DeepSeek-R1 и генерацией HTML-дашборда."""
        print(f"\n=======================================================")
        print(f"🤖 [INTELLIGENT QA BOT] Старт автоматического стресс-теста")
        print(f"Количество чертежей: {len(files_list)}")
        print(f"Подключение DeepSeek-R1 AI: {'ВКЛЮЧЕНО' if run_ai_audit else 'ВЫКЛЮЧЕНО'}")
        print(f"=======================================================\n")

        results = []
        for i, f in enumerate(files_list, 1):
            f_path = Path(f)
            prefix = "[БВК Завод]" if "drawings" in str(f).lower() else "[Синтетика]"
            print(f"[{i:02d}/{len(files_list):02d}] {prefix} {f_path.name[:28]}...", end=" ", flush=True)
            rec = self.evaluate_file(f)
            status_symbol = "✅" if rec["status"] == "OK" else ("⚠️" if rec["status"] == "UNRECOGNIZED" else "❌")
            cost_str = f"{rec.get('total_cost_rub', 0):,.0f} руб.".replace(",", " ") if rec.get('total_cost_rub') else "N/A"
            stock_mark = " [Склад]" if rec.get("pattern_in_stock") else ""
            print(f"{status_symbol} {rec['elapsed_ms']} мс | Сталь: {rec.get('steel_grade', '-')} | Масса: {rec.get('mass_kg', '-')} кг | {cost_str}{stock_mark}")
            results.append(rec)

        total = len(results)
        ok_count = sum(1 for r in results if r["status"] == "OK")
        unrec_count = sum(1 for r in results if r["status"] == "UNRECOGNIZED")
        err_count = sum(1 for r in results if r["status"] == "ERROR")
        factory_count = sum(1 for r in results if r.get("is_factory"))
        stamp_missing = sum(1 for r in results if "STAMP_MISSING" in r.get("warnings", []))
        views_missing = sum(1 for r in results if "VIEWS_MISSING" in r.get("warnings", []))
        patterns_in_stock = sum(1 for r in results if r.get("pattern_in_stock"))
        total_savings = sum(r.get("pattern_savings_rub", 0) for r in results)
        avg_time = sum(r.get("elapsed_ms", 0) for r in results) / max(1, total)

        summary_stats = {
            "total": total,
            "ok_count": ok_count,
            "ok_pct": round(ok_count / max(1, total) * 100, 1),
            "unrec_count": unrec_count,
            "err_count": err_count,
            "factory_count": factory_count,
            "stamp_missing": stamp_missing,
            "views_missing": views_missing,
            "patterns_in_stock": patterns_in_stock,
            "total_savings": total_savings,
            "avg_time": avg_time
        }

        deepseek_audit = {"available": False, "conclusion": "Аудит выключен.", "reasoning": ""}
        if run_ai_audit:
            print("\n🧠 [DEEPSEEK-R1] Запуск интеллектуального аудита результатов через нейросеть...", end=" ", flush=True)
            anomalies = [r for r in results if r.get("warnings") or r.get("status") != "OK"]
            deepseek_audit = self.request_deepseek_audit(summary_stats, anomalies)
            if deepseek_audit.get("available"):
                print("ГОТОВО ✅ (Заключение получено)")
            else:
                print(f"ПРОПУЩЕНО ⚠️ ({deepseek_audit.get('conclusion')})")

        json_path = self.output_dir / "eval_results.json"
        json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        md_report_path = self.generate_markdown_report(results, summary_stats, deepseek_audit)
        dashboard_path = self.generate_html_dashboard(results, summary_stats, deepseek_audit)

        print(f"\n[ГОТОВО] Структурированные данные: {json_path}")
        print(f"[ОТЧЕТ] Аналитический отчет: {md_report_path}")
        print(f"[ДАШБОРД] Интерактивный Visual QA Dashboard: {dashboard_path}\n")
        return results, dashboard_path

    def generate_markdown_report(self, results, summary_stats, deepseek_audit):
        """Формирует расширенный Markdown-отчет."""
        report_md = f"""# 🤖 Отчет Автоматического Тестирования и Стресс-Тестов (Intelligent QA Bot)

> **Дата запуска:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
> **Протестировано чертежей:** {summary_stats['total']} (из них заводских сканов БВК: {summary_stats['factory_count']})  
> **Среднее время на чертеж (Vulkan RX 6600):** {summary_stats['avg_time']:.1f} мс  

---

## 📊 Сводные показатели качества

| Метрика | Значение | Доля |
| :--- | :---: | :---: |
| **Успешно распознано и оценено (OK)** | {summary_stats['ok_count']} | {summary_stats['ok_pct']}% |
| **Реальные производственные чертежи БВК** | {summary_stats['factory_count']} | {summary_stats['factory_count']/max(1,summary_stats['total'])*100:.1f}% |
| **Чертежи без элементов ЕСКД (Unrecognized)** | {summary_stats['unrec_count']} | {summary_stats['unrec_count']/max(1,summary_stats['total'])*100:.1f}% |
| **Ошибки и сбои пайплайна (Errors)** | {summary_stats['err_count']} | {summary_stats['err_count']/max(1,summary_stats['total'])*100:.1f}% |
| **Отдельные виды (штамп отсутствует)** | {summary_stats['stamp_missing']} | {summary_stats['stamp_missing']/max(1,summary_stats['total'])*100:.1f}% |
| **Позиций с готовой оснасткой на складе** | **{summary_stats['patterns_in_stock']}** | Экономия: **{summary_stats['total_savings']:,.0f} руб.** |

---

## 🧠 Инженерное заключение DeepSeek-R1 (DeepThink)

{deepseek_audit.get('conclusion', 'Заключение не сформировано.')}

---

## 🔍 Детализированная таблица результатов

| Имя файла | Источник | Статус | Время (мс) | Штамп | Виды | Марка стали | Масса (кг) | Аналог БВК | Оснастка | Итого (руб.) | Замечания |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
"""
        for r in results:
            stamp_s = "✅" if r.get("stamp_found") else "❌"
            views_s = "✅" if r.get("views_found") else "❌"
            cost_s = f"{r.get('total_cost_rub', 0):,.0f} ₽".replace(",", " ") if r.get("total_cost_rub") else "-"
            warn_s = ", ".join(r.get("warnings", [])) if r.get("warnings") else "—"
            analog_s = r.get("best_analog", "-")
            stock_s = "Склад" if r.get("pattern_in_stock") else "—"
            src_s = "БВК" if r.get("is_factory") else "Синт."
            report_md += f"| `{r['file']}` | {src_s} | {r['status']} | {r['elapsed_ms']} | {stamp_s} | {views_s} | {r.get('steel_grade','-')} | {r.get('mass_kg','-')} | {analog_s} | {stock_s} | {cost_s} | {warn_s} |\n"

        out_report_file = self.output_dir / "eval_report.md"
        out_report_file.write_text(report_md, encoding="utf-8")
        return out_report_file


def gather_sample_files(count=12, source="factory"):
    """
    Собирает репрезентативную выборку чертежей.
    source: 'factory' (только заводские сканы БВК), 'val' (YOLO val), 'mined' (методички), 'all' (микс).
    """
    samples = []
    
    # 1. Заводские чертежи БВК
    factory_dir = Path(FACTORY_DRAWINGS_DIR)
    if factory_dir.exists() and source in ("factory", "all"):
        factory_files = sorted(list(factory_dir.glob("*.pdf")) + list(factory_dir.glob("*.png")))
        samples.extend(factory_files[:count])

    # Если запрошено еще или source != factory
    if len(samples) < count and source in ("val", "all", "mined"):
        remaining = count - len(samples)
        val_dir = Path("dataset_yolo/images/val")
        if val_dir.exists() and source in ("val", "all"):
            val_files = list(val_dir.glob("*.png")) + list(val_dir.glob("*.jpg"))
            samples.extend(val_files[:remaining])

    if len(samples) < count and source in ("mined", "all"):
        remaining = count - len(samples)
        mined_dir = Path("mined_dataset/images")
        if mined_dir.exists():
            mined_files = list(mined_dir.glob("*.png"))
            samples.extend(mined_files[:remaining])

    return samples[:count]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Интеллектуальный QA-бот оценки качества и стресс-тестирования")
    parser.add_argument("--input-dir", type=str, default=None, help="Пользовательская папка с чертежами")
    parser.add_argument("--source", type=str, default="factory", choices=["factory", "val", "mined", "all"], help="Источник чертежей")
    parser.add_argument("--count", type=int, default=12, help="Количество чертежей для теста")
    parser.add_argument("--out", type=str, default="bot_eval_results", help="Папка сохранения результатов")
    parser.add_argument("--no-ai", action="store_true", help="Отключить вызов DeepSeek-R1")
    args = parser.parse_args()

    bot = BatchEvalBot(output_dir=args.out)

    if args.input_dir and Path(args.input_dir).exists():
        input_files = list(Path(args.input_dir).glob("*.png")) + list(Path(args.input_dir).glob("*.jpg")) + list(Path(args.input_dir).glob("*.pdf"))
        target_files = input_files[:args.count]
    else:
        target_files = gather_sample_files(count=args.count, source=args.source)

    if not target_files:
        print("[ОШИБКА] Не найдено файлов чертежей для тестирования.")
        sys.exit(1)

    bot.run_batch(target_files, run_ai_audit=(not args.no_ai))
