"""Portable Excel writer for the public extraction notebook (standard dependencies)."""
from pathlib import Path
import math
import re


def write_workbook(tables, output):
    """Format already extracted AnnData tables; this function never reads CSV inputs."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo
    book=Workbook();book.remove(book.active)
    for idx,item in enumerate(tables):
        ws=book.create_sheet(item['name']);ws.sheet_view.showGridLines=False
        n=len(item['headers']);end=get_column_letter(n);title_end=get_column_letter(min(n,6))
        ws.merge_cells(f'A1:{title_end}1');ws['A1']=item['title'];ws['A1'].font=Font(name='Arial',size=14,bold=True)
        ws.merge_cells(f'A2:{title_end}2');ws['A2']=item['source']
        ws.merge_cells(f'A3:{title_end}3');ws['A3']=item.get('note','')
        ws.row_dimensions[1].height=36;ws.row_dimensions[2].height=30;ws.row_dimensions[3].height=60;ws.row_dimensions[4].height=10
        for r in [1,2,3]:ws.cell(r,1).alignment=Alignment(wrap_text=True,vertical='center')
        for j,header in enumerate(item['headers'],1):ws.cell(5,j,header)
        widths=[]
        for j,header in enumerate(item['headers']):
            sample=[row[j] for row in item['rows'][:100] if row[j] is not None]
            numeric=bool(sample) and all(isinstance(x,(int,float,bool)) for x in sample)
            width=max(14,min(22,len(header)/2+4)) if numeric else max(20,min(60,max([len(header)]+[len(str(x)) for x in sample])+2))
            if item['name']=='Read me':width=30 if j==0 else 120
            if item['name']=='S5 Classifier methods':width=[26,40,66,100,60,96][j]
            widths.append(width);ws.column_dimensions[get_column_letter(j+1)].width=width
        ws.row_dimensions[5].height=60
        for row in ws.iter_rows(min_row=5,max_row=5):
            for cell in row:
                cell.font=Font(name='Arial',size=10,bold=True);cell.fill=PatternFill('solid',fgColor='F2F2F2')
                cell.alignment=Alignment(wrap_text=True,vertical='center')
                cell.border=Border(top=Side(style='thin',color='555555'),bottom=Side(style='thin',color='888888'))
        for i,row in enumerate(item['rows'],6):
            lines=1
            for j,value in enumerate(row,1):
                cell=ws.cell(i,j,value)
                if isinstance(value,str):cell.data_type='s'
                cell.font=Font(name='Arial',size=10);cell.alignment=Alignment(wrap_text=True,vertical='center')
                cell.border=Border(bottom=Side(style='hair',color='E5E5E5'))
                if isinstance(value,(int,float)) and not isinstance(value,bool):
                    significance=re.search(r'(^|_)(p|q|pval|qval|pvalue|qvalue|padj|fdr)(_|$)',item['headers'][j-1],re.I) or item['name'].endswith('gsea_fdr')
                    cell.number_format='0.000E+00' if significance else '0' if float(value).is_integer() else '0.0000'
                lines=max(lines,sum(max(1,math.ceil(len(t)/(widths[j-1]-2))) for t in str(value or '').splitlines()))
            ws.row_dimensions[i].height=min(400,max(20,13*lines+7))
        table=Table(displayName=f'Supplement_{idx+1}',ref=f'A5:{end}{len(item["rows"])+5}')
        table.tableStyleInfo=TableStyleInfo(name='TableStyleLight1',showRowStripes=False)
        ws.add_table(table);ws.freeze_panes='B6'
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);book.save(output)
    return output
