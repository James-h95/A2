"""Model-oriented figure, v2. Keeps the first figures and report unchanged."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from draw_paper_figures import txt, box, arrow, INK, BLUE, TEAL, GOLD, PURPLE, LIGHT
from reportlab.graphics.shapes import Drawing, Rect, Line, Circle, Polygon, PolyLine
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor

OUT=Path(__file__).resolve().parents[1]/'output'/'pdf'/'v2'
OUT.mkdir(parents=True,exist_ok=True)
d=Drawing(1000,690)

def line(x1,y1,x2,y2,color='#7B8795',width=.8):
    d.add(Line(x1,y1,x2,y2,strokeColor=HexColor(color),strokeWidth=width))

def vector(x,y,n=6,w=16,h=66,color='#7DAED4'):
    for k in range(n):
        d.add(Rect(x,y+k*h/n,w,h/n-1.3,strokeColor=None,
                   fillColor=HexColor(color if k%3 else '#BAD2E6')))

def matrix(x,y,cols,rows,cell=12):
    palette=['#EDF5F3','#D7EAE5','#A7D0C3','#6CB39E','#338C75']
    for r in range(rows):
        for c in range(cols):
            v=(r*3+c*2+(3 if r==c else 0))%5
            d.add(Rect(x+c*cell,y+r*cell,cell-1,cell-1,strokeColor=None,fillColor=HexColor(palette[v])))

def circ(x,y,label):
    d.add(Circle(x,y,12,strokeColor=HexColor(INK),fillColor=HexColor('#FFFFFF'),strokeWidth=1))
    txt(d,x,y-4,label,14,True)

txt(d,18,664,'TweetBiDAF with OOF-trained answer reranking',21,True,anchor='start')
txt(d,18,642,'(a) Inside one span predictor',15,True,anchor='start')
txt(d,982,642,'Illustrative tokens and weights',11,anchor='end',color='#657382')

# Two input streams, kept distinct through the shared encoder.
txt(d,87,607,'TWEET',11,True)
txt(d,87,437,'QUESTION',11,True)
for x,s in [(20,'Tom'),(68,'visited'),(130,'Perth')]:
    box(d,x,566,44 if s!='visited' else 57,28,s,LIGHT,11)
for x,s in [(20,'Where'),(82,'did'),(126,'Tom')]:
    box(d,x,396,56 if s=='Where' else 40,28,s,LIGHT,11)
txt(d,88,544,'... on Monday.',11,color='#657382')
txt(d,88,374,'... visit?',11,color='#657382')

txt(d,252,607,'TOKEN REPRESENTATIONS',11,True)
txt(d,252,437,'SAME EMBEDDING LAYERS',11,True)
for base in (490,320):
    for k in range(3):
        x=205+k*34
        vector(x,base+36,3,21,30,'#7DAED4')
        vector(x,base+10,3,21,24,'#8EBFAE')
        vector(x,base,2,21,8,'#DDBD6A')
    arrow(d,[(174,base+70),(197,base+70)])
    box(d,193,base-33,119,23,'Highway',BLUE,11)
    arrow(d,[(250,base),(250,base-10)])
txt(d,252,588,'word / characters / flags',9,color='#657382')

txt(d,409,607,'SHARED BiLSTM',11,True)
for base in (492,322):
    for k in range(3):
        x=344+k*43
        box(d,x,base+39,35,25,'LSTM',BLUE,8.5)
        box(d,x,base+4,35,25,'LSTM',PURPLE,8.5)
        if k<2:
            arrow(d,[(x+35,base+52),(x+43,base+52)])
            arrow(d,[(x+43,base+16),(x+35,base+16)])
        line(x+17,base+29,x+17,base+39)
    arrow(d,[(312,base-21),(328,base-21),(328,base+33),(344,base+33)])
    for k in range(3):vector(349+k*43,base-28,4,25,25,'#9CACCC')
txt(d,409,452,'forward + backward states',10,color='#657382')
txt(d,409,278,'H: tweet states     U: question states',10,color='#657382')

# Cross attention matrix receiving both streams.
txt(d,582,607,'BiDAF ATTENTION',11,True)
matrix(533,449,6,5,15)
txt(d,578,535,'S = similarity(H, U)',12)
txt(d,578,432,'tweet x question',10,color='#657382')
arrow(d,[(475,524),(509,524),(509,493),(529,493)])
arrow(d,[(475,354),(509,354),(509,466),(529,466)])
box(d,525,381,107,30,'C2Q: a U',TEAL,12)
box(d,525,334,107,30,'Q2C: b H',TEAL,12)
arrow(d,[(578,449),(578,411)])
arrow(d,[(623,481),(643,481),(643,349),(632,349)])
box(d,515,275,128,34,['Concatenate interactions'],TEAL,10)
arrow(d,[(550,381),(550,309)])
arrow(d,[(606,334),(606,309)])
txt(d,579,258,'[H; C2Q; H*C2Q; H*Q2C]',9.5,color='#657382')

# Projection and self-attention, with explicit QKV and residual.
txt(d,744,607,'GATED SELF-ATTENTION',11,True)
box(d,687,278,116,28,'Projection',TEAL,11)
arrow(d,[(643,292),(687,292)])
for k in range(4):vector(695+k*26,328,5,18,47,'#8EBFAE')
arrow(d,[(745,306),(745,325)])
for x,s in [(700,'Q'),(745,'K'),(790,'V')]:
    box(d,x-15,398,30,23,s,TEAL,11,True)
    arrow(d,[(745,375),(745,386),(x,386),(x,398)])
matrix(710,438,5,5,14)
for x in (700,745,790):arrow(d,[(x,421),(x,434),(745,434),(745,440)])
txt(d,782,514,'4 heads',10,anchor='start')
arrow(d,[(745,508),(745,526)])
circ(745,540,'x')
box(d,793,528,42,24,'gate',PURPLE,10)
arrow(d,[(803,292),(840,292),(840,520),(814,520),(814,528)],dashed=True,color='#7B8795')
arrow(d,[(780,488),(826,488),(826,520)],dashed=True,color='#7B8795')
arrow(d,[(793,540),(757,540)])
arrow(d,[(745,552),(745,567)])
circ(745,579,'+')
arrow(d,[(687,292),(673,292),(673,579),(733,579)])
txt(d,670,552,'residual',9,anchor='end',color='#657382')
arrow(d,[(757,579),(854,579)])
txt(d,812,588,'LayerNorm',10,color='#657382')

# Modelling and boundary heads.
txt(d,917,607,'SPAN OUTPUT',11,True)
box(d,854,530,130,62,['Modelling','BiLSTMs'],BLUE,13)
box(d,852,460,58,28,'Start',GOLD,11)
box(d,925,460,58,28,'End',GOLD,11)
arrow(d,[(883,530),(883,488)])
arrow(d,[(954,530),(954,488)])
for x,vals in [(851,[.1,.15,.8,.2,.12]),(924,[.08,.12,.86,.18,.1])]:
    for k,v in enumerate(vals):
        d.add(Rect(x+k*12,396,8,46*v,strokeColor=None,fillColor=HexColor('#C6A04F')))
    line(x,394,x+58,394)
arrow(d,[(881,460),(881,446)])
arrow(d,[(954,460),(954,446)])
txt(d,881,377,'p(start)',10)
txt(d,954,377,'p(end)',10)
box(d,853,314,130,35,'Perth',GOLD,14,True)
arrow(d,[(881,369),(881,357),(917,357),(917,349)])
arrow(d,[(954,369),(954,357),(917,357)])
txt(d,918,292,'example Top-1 span',10,color='#657382')
txt(d,918,272,'Output heads shown schematically',8.5,color='#657382')

# Compact actual inference system; highlight candidates and selection rather than boxes.
line(18,235,982,235,'#D3DCE5')
txt(d,18,211,'(b) Final prediction: three seeds + one fixed reranker',15,True,anchor='start')
for k in range(3):
    box(d,22+k*6,104+k*11,140,42,'TweetBiDAF x 3' if k==2 else '',BLUE,11)
txt(d,93,83,'seeds 1, 2, 3; same data',10,color='#657382')
arrow(d,[(174,137),(205,137)])
circ(220,137,'+')
txt(d,220,109,'mean',11)
txt(d,220,92,'start / end',10,color='#657382')
arrow(d,[(232,137),(275,137)])
for k,(s,c) in enumerate([('1  Perth on',LIGHT),('2  Perth',GOLD),('3  on Monday',LIGHT)]):
    box(d,277,150-k*25,230,22,s,c,10.5)
txt(d,392,82,'Top-10 spans (illustrative ranking)',10,color='#657382')
arrow(d,[(507,137),(546,137)])
matrix(550,110,6,3,12)
txt(d,587,91,'30 features / span',10,color='#657382')
arrow(d,[(623,137),(660,137)])
box(d,661,113,145,48,['OOF-trained','LambdaMART'],PURPLE,12,True)
arrow(d,[(806,137),(842,137)])
box(d,844,111,138,51,'Perth',GOLD,14,True)
txt(d,915,91,'selected answer',10,color='#657382')
txt(d,18,40,'Blue: sequence representations     Green: attention     Gold: answer candidates     Purple: learned selection',11,anchor='start',color='#657382')
txt(d,18,20,'The detailed predictor is shared across seeds. Gold answers are used for training/evaluation only, never for prediction.',10.5,anchor='start',color='#657382')

# Export true vector outputs, 7 inches wide.
d.scale(.504,.504); d.width*=.504; d.height*=.504
renderPDF.drawToFile(d,str(OUT/'tweetqa_model_v2.pdf'))
renderSVG.drawToFile(d,str(OUT/'tweetqa_model_v2.svg'))
print(OUT)
