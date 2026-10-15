"""Editable vector figures for the TweetQA report (ReportLab; no model inference)."""
from pathlib import Path
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, PolyLine
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor, white

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'pdf'
OUT.mkdir(parents=True, exist_ok=True)
INK = '#263342'
GREY = '#657382'
BLUE = '#E9F1F9'
TEAL = '#E7F3EF'
GOLD = '#FBF1D8'
PURPLE = '#F0EAF7'
LIGHT = '#F5F7F9'

def txt(d, x, y, s, size=12, bold=False, anchor='middle', color=INK):
    d.add(String(x, y, s, fontName='Helvetica-Bold' if bold else 'Helvetica',
                 fontSize=size, textAnchor=anchor, fillColor=HexColor(color)))

def box(d, x, y, w, h, lines, fill=BLUE, size=12, bold=False, stroke=INK):
    d.add(Rect(x, y, w, h, rx=4, ry=4, strokeColor=HexColor(stroke),
               strokeWidth=.85, fillColor=HexColor(fill)))
    if isinstance(lines, str): lines = [lines]
    step = size * 1.23
    first = y + h/2 + (len(lines)-1)*step/2 - size*.34
    for i, s in enumerate(lines): txt(d, x+w/2, first-i*step, s, size, bold)

def arrow(d, pts, dashed=False, color=INK):
    d.add(PolyLine(sum(([x,y] for x,y in pts), []), strokeColor=HexColor(color),
                   strokeWidth=1, fillColor=None, strokeDashArray=[4,3] if dashed else None))
    x,y=pts[-1]; px,py=pts[-2]
    import math
    a=math.atan2(y-py,x-px); length=6; half=2.5
    bx=x-length*math.cos(a); by=y-length*math.sin(a)
    d.add(Polygon([x,y,bx+half*math.sin(a),by-half*math.cos(a),
                   bx-half*math.sin(a),by+half*math.cos(a)],
                  fillColor=HexColor(color), strokeColor=None))

def down(d,x,y1,y2): arrow(d,[(x,y1),(x,y2)])

def save(d, name):
    # The physical width is 7 inches, suitable for a two-column figure*.
    d.scale(.7,.7); d.width*=.7; d.height*=.7
    renderPDF.drawToFile(d,str(OUT/(name+'.pdf')))
    renderSVG.drawToFile(d,str(OUT/(name+'.svg')))

def architecture():
    d=Drawing(720,470)
    txt(d,130,449,'(a) TweetBiDAF span predictor',13,True)
    txt(d,492,449,'(b) Final inference pipeline',13,True)
    d.add(Line(270,38,270,433,strokeColor=HexColor('#D5DDE5'),strokeWidth=.8))

    box(d,13,396,111,31,'Tweet tokens',LIGHT)
    box(d,139,396,111,31,'Question tokens',LIGHT)
    arrow(d,[(69,396),(69,381),(130,381),(130,368)])
    d.add(PolyLine([195,396,195,381,130,381],strokeColor=HexColor(INK),strokeWidth=1,fillColor=None))
    box(d,13,321,237,47,['Word + character CNN + flags','100 + 150 + 8 dimensions'],BLUE)
    down(d,130,321,306)
    box(d,13,277,237,29,'Two-layer highway',BLUE)
    down(d,130,277,262)
    box(d,13,218,237,44,['Shared BiLSTM encoder','128 hidden units per direction'],BLUE)
    down(d,130,218,203)
    box(d,13,164,237,39,['BiDAF: C2Q + Q2C','Project to 256 dimensions'],TEAL)
    down(d,130,164,149)
    box(d,13,116,237,33,'Gated self-attention (4 heads)',TEAL)
    down(d,130,116,101)
    box(d,13,57,237,44,['Modelling BiLSTMs + span heads','Start distribution / end distribution'],BLUE)
    txt(d,130,38,'Same encoder weights for tweet and question',10,color=GREY)

    box(d,391,397,202,30,'Tweet + question',LIGHT)
    for x in (295,440,585):
        cx=x+65
        arrow(d,[(492,397),(492,382),(cx,382),(cx,367)])
        box(d,x,320,130,47,['TweetBiDAF',f'Seed {(x-295)//145+1}'],BLUE,12,True)
    for cx in (360,505,650):
        arrow(d,[(cx,320),(cx,294),(492,294),(492,280)])
    box(d,355,235,274,45,['Average probabilities across 3 seeds','Mean start / mean end distributions'],TEAL,11.5)
    down(d,492,235,220)
    box(d,355,178,274,42,['Top-10 valid answer spans','Maximum span length: 15 tokens'],GOLD,12)
    down(d,492,178,150)
    box(d,355,119,274,31,'30 numerical features per candidate',GOLD)
    down(d,492,119,104)
    box(d,355,65,274,39,['Fixed OOF-trained LambdaMART','One score per candidate'],PURPLE,12)
    down(d,492,65,50)
    box(d,355,20,274,30,'Best span -> original-text answer',LIGHT,12,True)
    save(d,'tweetqa_final_architecture')

def oof():
    d=Drawing(720,435)
    txt(d,360,414,'OOF candidate generation and reranker training',14,True)
    txt(d,360,393,'Split the training partition by tweet; keep outer validation and test separate.',11,color=GREY)
    for x,s in [(88,'Base-model data'),(263,'Fold-specific model'),(438,'Unseen fold'),(622,'Reranker training records')]:
        txt(d,x,368,s,11,True)
    for y,train,held,num in [(288,'B + C','A',1),(204,'A + C','B',2),(120,'A + B','C',3)]:
        box(d,13,y,150,57,[train,'Fit / internal validation'],BLUE,11.5)
        arrow(d,[(163,y+29),(188,y+29)])
        box(d,188,y,150,57,['TweetBiDAF',f'Fold model {num}'],BLUE,12)
        arrow(d,[(338,y+29),(363,y+29)])
        box(d,363,y,150,57,[f'Predict on {held}','Top-10 spans / question'],GOLD,11.5)
        arrow(d,[(513,y+29),(542,y+29)])
        box(d,542,y,160,57,['Candidate features',f'+ labels from gold in {held}'],PURPLE,11.5)
    # Each row contributes to the pooled reranker training set.
    arrow(d,[(702,317),(713,317),(713,88),(360,88),(360,72)])
    arrow(d,[(702,233),(713,233)])
    arrow(d,[(702,149),(713,149)])
    box(d,188,28,344,44,['Pool all OOF records -> train ONE LambdaMART','9,570 questions / 95,700 candidate rows'],PURPLE,12,True)
    txt(d,360,10,'Gold supplies training labels only. Fold models are not used for final prediction.',10,color=GREY)
    txt(d,88,274,'Hold out A entirely',10,color=GREY)
    txt(d,88,190,'Hold out B entirely',10,color=GREY)
    txt(d,88,106,'Hold out C entirely',10,color=GREY)
    save(d,'tweetqa_oof_training')

if __name__=='__main__':
    architecture(); oof()
    print(OUT)
