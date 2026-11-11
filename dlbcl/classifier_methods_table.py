"""Great Tables rendering for the supplementary molecular classifier methods.

Uses the typography and horizontal-rule conventions of table1.ipynb. Only the
public methods JSON is read; patient data and classifier inference are not used.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path


COLUMNS = ["Classifier", "Inputs", "Assignment", "Code"]
COLUMN_WIDTHS = dict(zip(COLUMNS, [150, 210, 470, 170]))
TABLE_WIDTH = sum(COLUMN_WIDTHS.values())


def code_url(data, ref):
    if "url" in ref:
        return ref["url"]
    value = f"{data['repository']}/blob/{data['code_revision']}/{ref['path']}"
    return value + (f"#L{ref['line']}" if "line" in ref else "")


def render_classifier_methods_table(data, section_index):
    """Return one displayable GT object, with the same 11px/14px sizing as Table 1."""
    import pandas as pd
    from great_tables import GT, html, loc, style
    from dlbcl.validation_clinical_table import TABLE_FONT_FAMILY

    group = data["sections"][section_index]
    identifier = f"classifier_methods_{section_index}"
    rows = []
    for row in group["rows"]:
        method = (
            f"<a class='method' href='{escape(row['publication'], quote=True)}'>"
            f"<strong><em>{escape(row['method'])}</em></strong></a>"
            f"<br>{escape(row['status'])}"
        )
        links = "<br>".join(
            f"<a href='{escape(code_url(data, ref), quote=True)}'>{escape(ref['label'])}</a>"
            for ref in row["code"]
        )
        rows.append(dict(Classifier=method, Inputs=row["inputs"], Assignment=row["implementation"], Code=links))
    table = (
        GT(pd.DataFrame(rows), id=identifier)
        .tab_header(title=data["subtitle"], subtitle=group["title"])
        .fmt_markdown(columns=["Classifier", "Code"])
        .cols_label(Classifier="Classifier and implementation", Inputs="Inputs",
                    Assignment="Assignment and unavailable features", Code="Analysis code")
        .cols_align(align="left", columns=COLUMNS)
        .cols_width(**{k: f"{v}px" for k, v in COLUMN_WIDTHS.items()})
        .tab_style(style=style.text(weight="bold"), locations=loc.column_labels())
        .tab_style(style=style.text(v_align="top", whitespace="normal"), locations=loc.body())
        .tab_options(
            table_width=f"{TABLE_WIDTH}px", table_layout="fixed",
            table_font_names=[TABLE_FONT_FAMILY, "sans-serif"],
            table_font_size="11px", heading_title_font_size="14px",
            heading_subtitle_font_size="11px", heading_subtitle_font_weight="bold",
            table_background_color="white", heading_background_color="white",
            column_labels_background_color="white", source_notes_background_color="white",
            table_body_hlines_color="#D3D3D3", table_body_hlines_width="1px",
            table_body_vlines_style="none", column_labels_vlines_style="none",
            data_row_padding_horizontal="10px", column_labels_padding_horizontal="10px",
            source_notes_padding_horizontal="10px", source_notes_font_size="10px",
            row_striping_include_table_body=False, container_padding_y="0px",
            table_additional_css=[
                f"#{identifier} .gt_subtitle {{ font-style: italic; }}",
                f"#{identifier} .gt_row, #{identifier} .gt_col_heading {{ white-space: normal; line-height: 1.35; }}",
                f"#{identifier} a {{ color: inherit; text-decoration: underline; text-underline-offset: 2px; }}",
                f"#{identifier} a.method {{ text-decoration: none; }}",
                f"#{identifier} .gt_sourcenote {{ line-height: 1.35; }}",
            ],
        )
    )
    table = table.tab_source_note(html(f"<strong>Inputs and preprocessing.</strong> {escape(group['lead'])}"))
    for note in group["notes"]:
        table = table.tab_source_note(note)
    resources = [dict(label=group["notebook_label"], path=group["notebook"])]
    resources += data["resources"][:1] if section_index == 0 else data["resources"][1:]
    links = " · ".join(
        f"<a href='{escape(code_url(data, ref), quote=True)}'>{escape(ref['label'])}</a>"
        for ref in resources
    )
    table = table.tab_source_note(html(
        f"Code revision <a href='{escape(data['repository'])}/tree/{data['code_revision']}'>"
        f"{data['code_revision'][:7]}</a>. {links}"
    ))
    return table


@lru_cache(maxsize=1)
def _embedded_fonts():
    """Embed the manuscript font so exported HTML works without local font paths."""
    from matplotlib.font_manager import FontProperties, findfont
    from dlbcl.validation_clinical_table import TABLE_FONT_FAMILY

    css = []
    for weight, style in [("normal", "normal"), ("bold", "normal"), ("bold", "italic")]:
        path = Path(findfont(FontProperties(family=TABLE_FONT_FAMILY, weight=weight, style=style)))
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        css.append(
            "@font-face {"
            f"font-family:'{TABLE_FONT_FAMILY}';font-weight:{weight};font-style:{style};"
            f"src:url(data:font/ttf;base64,{encoded}) format('truetype');}}"
        )
    return "\n".join(css)


def tables_html(tables, *, landscape=False):
    """Portable HTML, with one table per printed A4 landscape page."""
    content = "\n".join(
        f"<section class='methods-page'>{table.as_raw_html(make_page=False)}</section>"
        for table in tables
    )
    page_css = "@page {size:A4 landscape; margin:0.4in;}" if landscape else ""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Molecular classifier implementations and feature handling</title><style>"
        + _embedded_fonts()
        + page_css
        + "body {margin:8px; background:white;} .methods-page {margin:0 auto 24px;} "
        "@media print { body {margin:0;} .methods-page {break-after:page; margin:0;} "
        ".methods-page:last-child {break-after:auto;} tr {break-inside:avoid;} }"
        "</style></head><body>" + content + "</body></html>"
    )


def export_classifier_methods_tables(data, output_dir, *, tables=None):
    """Export GT HTML, vector PDFs and PNG previews; fail explicitly on export errors."""
    import nokap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = tables if tables is not None else [render_classifier_methods_table(data, i) for i in range(2)]
    if len(tables) != 2:
        raise ValueError("Expected genomic and transcriptomic tables")
    paths = {}
    for label, table in zip(["genomic", "transcriptomic"], tables):
        markup = tables_html([table])
        stem = f"classifier_methods_{label}"
        path = output_dir / f"{stem}.html"
        path.write_text(markup, encoding="utf-8")
        paths[f"{label}_html"] = path
        for ext in ["pdf", "png"]:
            path = output_dir / f"{stem}.{ext}"
            nokap.from_html(markup, path, selector="table", expand=8, zoom=2.0,
                            delay=.5, vwidth=TABLE_WIDTH+40, vheight=1400)
            paths[f"{label}_{ext}"] = path
    markup = tables_html(tables, landscape=True)
    path = output_dir / "Supplementary_Table_classifier_methods.html"
    path.write_text(markup, encoding="utf-8")
    paths["combined_html"] = path
    path = output_dir / "Supplementary_Table_classifier_methods.pdf"
    # CSS declares landscape explicitly; the CDP landscape flag would double-
    # rotate dimensions in nokap versions that also swap the paper dimensions.
    nokap.from_html(markup, path, selector="html", page_size="a4", landscape=False,
                    margins=.4, zoom=1.0, print_background=True, delay=.5,
                    vwidth=TABLE_WIDTH+40, vheight=1400)
    paths["combined_pdf"] = path
    return paths
