"""Generate the original workflow figure; optional dependency: reportlab."""
from pathlib import Path
from reportlab.graphics.shapes import Drawing, Rect, Line, Polygon, String
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor, white

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
d=Drawing(1060,650)
ink=HexColor('#172B4D');muted=HexColor('#526477');blue=HexColor('#2563EB')
green=HexColor('#087E6E');light=HexColor('#EFF5FD');border=HexColor('#CCD7E5')

def text(x,y,value,size=12,color=ink,bold=False,anchor='start'):
    d.add(String(x,y,value,fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,fillColor=color,textAnchor=anchor))
def box(x,y,w,h,title,lines,fill=light):
    d.add(Rect(x,y,w,h,rx=9,ry=9,fillColor=fill,strokeColor=border,strokeWidth=1))
    text(x+14,y+h-25,title,13,bold=True)
    for i,line in enumerate(lines): text(x+14,y+h-48-i*18,line,11,color=muted)
def arrow(x1,y1,x2,y2,color=blue):
    d.add(Line(x1,y1,x2,y2,strokeColor=color,strokeWidth=2))
    if x2>x1: p=[x2,y2,x2-8,y2+4,x2-8,y2-4]
    elif x2<x1: p=[x2,y2,x2+8,y2+4,x2+8,y2-4]
    elif y2>y1: p=[x2,y2,x2-4,y2-8,x2+4,y2-8]
    else: p=[x2,y2,x2-4,y2+8,x2+4,y2+8]
    d.add(Polygon(p,fillColor=color,strokeColor=color))

d.add(Rect(0,0,1060,650,fillColor=white,strokeColor=white))
text(28,614,'Quantum-Chemistry Assistant: Local Adaptation Workflow',22,bold=True)
text(28,590,'Public references + original exercises | Apple Silicon | Fixed diagnostic prompts',12,color=muted)

xs=[28,286,544,802]
box(xs[0],430,230,125,'1  Curate sources',['Textbook concepts and papers','PySCF API and public code','Record version + provenance'])
box(xs[1],430,230,125,'2  Build exercises',['Original bilingual Q&A','88 train / 12 validation','8 fixed diagnostic prompts'])
box(xs[2],430,230,125,'3  Validate data',['Check formulas and code','Check exact prompt overlap','Check chat token lengths'])
box(xs[3],430,230,125,'4  Establish baseline',['Pinned MLX model revision','Greedy decoding, seed 42','Save responses + stop reasons'])
for i in range(3):arrow(xs[i]+232,492,xs[i+1]-5,492)
arrow(918,425,918,390)

box(544,215,488,172,'5  Train low-rank adapters',['Frozen 4-bit base weights; group size 64','Last 4 blocks: query + value projections','Rank 8; MLX scale 16; Adam; learning rate 0.0001','Batch 1; 176 steps; answer-token loss only','Validation every 44 steps; gradient checkpointing'])
box(286,215,230,172,'6  Compare responses',['Same model and tokenizer','Same prompts and decoding','Review correctness and code','Report repetition + truncation','Use the fixed final checkpoint'])
box(28,215,230,172,'7  Package the study',['Adapter + reproducible scripts','Data and public references','English report and manuscript','Publish only reviewed files'])
arrow(538,300,522,300);arrow(280,300,264,300)

text(28,182,'Inside each adapted projection',14,bold=True)
box(28,70,190,85,'Frozen branch',['x -> quantized linear layer'],fill=HexColor('#F1F4F8'))
box(255,70,290,85,'Trainable branch',['x -> A -> B -> multiply by scale'],fill=HexColor('#E9F7F3'))
text(235,110,'+',22,bold=True)
arrow(551,110,592,110)
box(600,70,180,85,'Output',['Sum both branches'])
arrow(786,110,824,110)
box(832,70,200,85,'Supervision',['Predict answer tokens'])
text(28,36,'y = Q(W0)x + sBAx     |     Only A and B are updated; source documents are not copied into the training set.',11,color=muted)
renderSVG.drawToFile(d,str(OUT/'workflow.svg'))
renderPDF.drawToFile(d,str(OUT/'workflow.pdf'))
print(OUT/'workflow.svg')
