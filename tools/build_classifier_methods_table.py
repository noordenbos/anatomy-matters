"""Build the public supplementary classifier table without loading patient data.

The Great Tables figure export uses the standard figure environment.
Requires python-docx only for --docx. Examples from the repository root:
    python tools/build_classifier_methods_table.py --figures
    python tools/build_classifier_methods_table.py --docx

Edit docs/supplementary/classifier_methods.json and rebuild both formats.
Code links are pinned to the source revision declared in that JSON file.
--verify-repo checks every code path and line locator against that Git revision.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def code_url(data, ref):
    if "url" in ref:
        return ref["url"]
    url = f"{data['repository']}/blob/{data['code_revision']}/{ref['path']}"
    return url + (f"#L{ref['line']}" if "line" in ref else "")


def all_code_refs(data):
    for section in data["sections"]:
        yield {"path": section["notebook"]}
        for row in section["rows"]:
            yield from row["code"]
    yield from data["resources"]


def validate(data, repo=None):
    assert re.fullmatch(r"(?:[0-9a-f]{40}|v[0-9]+\.[0-9]+\.[0-9]+)", data["code_revision"])
    rows = [r for s in data["sections"] for r in s["rows"]]
    assert len(rows) == 9 and len({r["method"] for r in rows}) == 9
    for row in rows:
        assert all(row[k] for k in ("method", "status", "publication", "inputs", "implementation", "code"))
    # Public artifacts must not expose private machine locations or review instructions.
    serialized = json.dumps(data)
    for forbidden in ("/Users/", "file://", "Manuscript change", "Implementation requiring disclosure"):
        assert forbidden not in serialized, forbidden
    if repo:
        checked = set()
        for ref in all_code_refs(data):
            if "path" not in ref:
                continue
            key = (ref["path"], ref.get("line"))
            if key in checked:
                continue
            source = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{data['code_revision']}:{ref['path']}"], text=True
            )
            if "line" in ref:
                assert 1 <= ref["line"] <= len(source.splitlines()), key
            checked.add(key)
        print(f"Verified {len(checked)} versioned code targets")


def markdown(data):
    out = [f"# {data['subtitle']}", "", data["introduction"], "",
           f"Code revision: [`{data['code_revision'][:7]}`]({data['repository']}/tree/{data['code_revision']}).", ""]
    for section in data["sections"]:
        out += [f"## {section['title']}", "", section["lead"], "",
                f"[{section['notebook_label']}]({code_url(data, {'path': section['notebook']})})", "",
                "| Classifier and implementation | Inputs | Assignment and unavailable features | Analysis code |",
                "|---|---|---|---|"]
        for row in section["rows"]:
            refs = "<br>".join(f"[{r['label']}]({code_url(data, r)})" for r in row["code"])
            vals = [f"[{row['method']}]({row['publication']})<br>{row['status']}", row["inputs"], row["implementation"], refs]
            out.append("| " + " | ".join(v.replace("|", "\\|") for v in vals) + " |")
        out += [""]
        for note in section["notes"]:
            out += [note, ""]
    out += ["## Reproducibility resources", ""]
    out += [f"- [{r['label']}]({code_url(data, r)})" for r in data["resources"]]
    out += ["", "The Great Tables version is displayed in `notebooks/supplementary_classifier_methods.ipynb`, "
            "using the typography and horizontal rules of `table1.ipynb`. "
            "Regenerate HTML, PDF and PNG with `python tools/build_classifier_methods_table.py --figures`; "
            "use `--docx` for the editable Word companion. "
            "The JSON source is `docs/supplementary/classifier_methods.json`; no patient data or classifier execution is required.", ""]
    return "\n".join(out)


def word(data, dest):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = Inches(11.69), Inches(8.27)
    section.top_margin = section.bottom_margin = Inches(.48)
    section.left_margin = section.right_margin = Inches(.60)
    normal = doc.styles["Normal"]
    normal.font.name = "DejaVu Sans"
    normal.font.size = Pt(8.25)
    normal.font.color.rgb = RGBColor.from_string("333333")
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.03
    for name, size in [("Title", 10.5), ("Subtitle", 8.25), ("Heading 1", 8.25)]:
        style = doc.styles[name]
        style.font.name = "DejaVu Sans"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(7 if name == "Heading 1" else 0)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style.font.bold = name == "Heading 1"
        style.font.italic = name == "Heading 1"
        for border in style.element.xpath("./w:pPr/w:pBdr"):
            border.getparent().remove(border)
        if name == "Subtitle":
            style.font.italic = False
        for spacing in style.element.xpath("./w:rPr/w:spacing"):
            spacing.getparent().remove(spacing)
    doc.core_properties.title = data["subtitle"]
    doc.core_properties.subject = "Supplementary methods for molecular classifier implementation"
    doc.core_properties.author = ""

    def hyperlink(p, label, url, bold=False, color="333333", size=None, italic=False):
        h = OxmlElement("w:hyperlink")
        h.set(qn("r:id"), p.part.relate_to(url, RT.HYPERLINK, is_external=True))
        r = OxmlElement("w:r")
        props = OxmlElement("w:rPr")
        shade = OxmlElement("w:color"); shade.set(qn("w:val"), color); props.append(shade)
        if size is not None:
            sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(round(size * 2))); props.append(sz)
        if bold:
            props.append(OxmlElement("w:b"))
        if italic:
            props.append(OxmlElement("w:i"))
        if not bold:
            underline = OxmlElement("w:u"); underline.set(qn("w:val"), "single"); props.append(underline)
        r.append(props)
        text = OxmlElement("w:t"); text.text = label; r.append(text)
        h.append(r); p._p.append(h)

    def para(text, small=False):
        p = doc.add_paragraph(text)
        if small:
            for r in p.runs:
                r.font.size = Pt(7.5)
        return p

    widths = [1.57, 2.20, 4.93, 1.79]
    for page, group in enumerate(data["sections"]):
        title = doc.add_paragraph(data["subtitle"], "Title")
        title.paragraph_format.page_break_before = bool(page)
        doc.add_paragraph(group["title"], "Heading 1")
        table = doc.add_table(rows=1, cols=4)
        table.autofit = False
        for col, width in zip(table.columns, widths):
            col.width = Inches(width)
        headers = ["Classifier and implementation", "Inputs", "Assignment and unavailable features", "Analysis code"]
        for cell, title in zip(table.rows[0].cells, headers):
            cell.text = title
        table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for row in group["rows"]:
            cells = table.add_row().cells
            hyperlink(cells[0].paragraphs[0], row["method"], row["publication"], bold=True, italic=True)
            p = cells[0].add_paragraph(row["status"])
            cells[1].text = row["inputs"]
            cells[2].text = row["implementation"]
            for i, ref in enumerate(row["code"]):
                p = cells[3].paragraphs[0] if i == 0 else cells[3].add_paragraph()
                hyperlink(p, ref["label"], code_url(data, ref))
        for ri, row in enumerate(table.rows):
            row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
                tc = cell._tc.get_or_add_tcPr()
                shade = OxmlElement("w:shd"); shade.set(qn("w:fill"), "FFFFFF"); tc.append(shade)
                borders = OxmlElement("w:tcBorders")
                for edge in ("top", "left", "bottom", "right"):
                    visible = edge == "bottom" or (edge == "top" and ri == 0)
                    strong = ri == 0 or ri == len(table.rows) - 1
                    el = OxmlElement("w:" + edge)
                    el.set(qn("w:val"), "single" if visible else "nil")
                    el.set(qn("w:sz"), "12" if strong else "6")
                    el.set(qn("w:color"), "A8A8A8" if ri == len(table.rows) - 1 else "D3D3D3")
                    borders.append(el)
                tc.append(borders)
                margins = OxmlElement("w:tcMar")
                for edge, val in (("top", 120), ("bottom", 120), ("left", 150), ("right", 150)):
                    el = OxmlElement("w:" + edge); el.set(qn("w:w"), str(val)); el.set(qn("w:type"), "dxa"); margins.append(el)
                tc.append(margins)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(3)
                    p.paragraph_format.line_spacing = 1.15
                    for run in p.runs:
                        run.font.size = Pt(8.25)
                        if ri == 0:
                            run.bold = True
        for i, note in enumerate(["Inputs and preprocessing. " + group["lead"], *group["notes"]]):
            p = para(note, small=True)
            if i == 0:
                p.paragraph_format.space_before = Pt(7)
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.add_run("Code revision ").font.size = Pt(7.5)
        hyperlink(p, data["code_revision"][:7], f"{data['repository']}/tree/{data['code_revision']}", size=7.5)
        p.add_run(". ")
        refs = [dict(label=group["notebook_label"], path=group["notebook"])]
        refs += data["resources"][:1] if page == 0 else data["resources"][1:]
        for i, ref in enumerate(refs):
            if i:
                p.add_run(" · ")
            hyperlink(p, ref["label"], code_url(data, ref), size=7.5)
    doc.save(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs/supplementary/classifier_methods.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/supplementary")
    parser.add_argument("--docx", action="store_true")
    parser.add_argument("--figures", action="store_true", help="Export Great Tables HTML, PDF and PNG")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures/supplementary_classifier_methods")
    parser.add_argument("--verify-repo", type=Path)
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    validate(data, args.verify_repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    md = args.output_dir / "classifier_methods.md"
    md.write_text(markdown(data))
    print(md)
    if args.docx:
        path = args.output_dir / "Supplementary_Table_classifier_methods.docx"
        word(data, path)
        print(path)
    if args.figures:
        sys.path.insert(0, str(ROOT))
        from dlbcl.classifier_methods_table import export_classifier_methods_tables
        for path in export_classifier_methods_tables(data, args.figure_dir).values():
            print(path)


if __name__ == "__main__":
    main()
