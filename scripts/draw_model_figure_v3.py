from pathlib import Path
from draw_paper_figures import txt,box,arrow,INK,BLUE,TEAL,GOLD,PURPLE,LIGHT
from reportlab.graphics.shapes import Drawing,Rect,Line,Circle
from reportlab.graphics import renderPDF,renderSVG
from reportlab.lib.colors import HexColor
OUT=Path(__file__).resolve().parents[1]/'output/pdf/v3'; OUT.mkdir(parents=True,exist_ok=True)
d=Drawing(760,1370)
def t(x,y,s,z=11,b=False,anchor='middle',color=INK):txt(d,x,y,s,z,b,anchor,color)
def a(*p):arrow(d,list(p))
def panel(y,h,title):
 d.add(Rect(16,y,728,h,rx=8,ry=8,fillColor=None,strokeColor=HexColor('#CBD5DF'),strokeWidth=.8));t(30,y+h-23,title,14,True,'start')
def heat(x,y,n,m,c=13):
 pal=['#E4F1EC','#B6D8CB','#82BAA6','#418F78']
 for r in range(m):
  for k in range(n):d.add(Rect(x+k*c,y+r*c,c-1,c-1,fillColor=HexColor(pal[(r*3+k*2+(r==k))%4]),strokeColor=None))
def cells(x,y):
 for r in range(2):
  for k in range(4):
   box(d,x+k*57,y-r*29,48,22,'LSTM',BLUE if r==0 else PURPLE,9)
   if k<3:
    p=[(x+k*57+48,y-r*29+11),(x+(k+1)*57,y-r*29+11)]
    a(*(p if r==0 else p[::-1]))
 # Dimensions are stated below both streams.
t(380,1341,'TweetBiDAF + answer reranker',23,True)
t(380,1320,'Model architecture | one input question, one selected answer',11)
panel(1006,297,'1  Token representation and contextual encoding')
box(d,43,1225,310,35,['Tweet: Tom visited Perth on Monday .'],LIGHT,11)
box(d,406,1225,310,35,'Question: Where did Tom visit ?',LIGHT,11)
t(380,1207,'Same embedding layers and encoder weights for both sequences',10)
for x,w,label,col in [(43,225,['Word embedding: 100d','FastText initialization; trainable'],BLUE),(282,235,['Raw-token Character CNN: 150d','32d chars; widths 3/4/5; 50 each'],TEAL),(531,185,['Token flags: 8d','Handcrafted indicators'],GOLD)]:
 box(d,x,1146,w,45,label,col,10)
 a((x+w/2,1146),(x+w/2,1132),(380,1132),(380,1120))
box(d,241,1085,278,35,'Concatenate (258d) -> 2-layer Highway',BLUE,11)
t(380,120,'',10)
t(380,1137,'',10)
for x in (80,443):
 a((380,1085),(380,1080),(x+109,1080),(x+109,1077));cells(x,1055)
t(380,1015,'Shared 1-layer BiLSTM: H and U; 128 per direction -> 256d',10,True)
# Compact note outside the feature boxes.
t(380,1268,'FastText is trained only on the training partition; no external pretrained vectors.',9)
a((380,1006),(380,985))
panel(773,212,'2  Question-aware tweet representations')
t(145,932,'BiDAF cross-attention',12,True);heat(69,839,7,5,17)
t(128,823,'S: tweet x question',10)
box(d,230,911,208,30,'C2Q = softmax(S) U',TEAL,11)
box(d,230,867,208,30,'Q2C = softmax(max S) H',TEAL,11)
a((188,889),(211,889),(211,926),(230,926));a((211,889),(211,882),(230,882))
box(d,480,867,231,74,['[H; C2Q; H*C2Q; H*Q2C]','1024d interactions','Linear + ReLU -> 256d'],TEAL,10)
a((438,926),(480,926));a((438,882),(480,882))
t(380,796,'C2Q: attend to question tokens. Q2C: summarize question-relevant tweet tokens.',10)
a((595,867),(595,812),(724,812),(724,756),(380,756),(380,741))
panel(551,190,'3  Gated self-attention over tweet tokens')
box(d,44,639,105,40,['Projected','tweet states Z'],BLUE,11)
for x,s in [(195,'Q'),(242,'K'),(289,'V')]:
 box(d,x,644,33,30,s,TEAL,11,True);a((149,659),(174,659),(174,688),(x+16,688),(x+16,674))
heat(356,624,5,5,13);a((322,659),(356,659))
t(388,611,'4 heads',10)
box(d,456,636,106,46,['Gate g','sigmoid([Z; A])'],PURPLE,10)
a((421,659),(456,659))
box(d,601,631,112,55,['Gate + residual','LayerNorm','X = LN(Z + g*A)'],TEAL,9.5)
a((562,659),(601,659));a((96,639),(96,594),(655,594),(655,631))
t(380,573,'A = multi-head attention(Z, Z, Z). Dropout is omitted from this schematic.',10)
a((380,551),(380,528))
panel(248,280,'4  Additional BiLSTMs and explicit span predictions')
box(d,42,435,100,40,['X: question-aware','tweet states'],TEAL,10)
box(d,183,435,215,40,['M1: 2-layer modelling BiLSTM','New layers, after attention'],BLUE,11)
box(d,446,435,261,40,['M2: 1 additional modelling BiLSTM','Input: M1; output: M2'],BLUE,11)
a((142,455),(183,455));a((398,455),(446,455))
box(d,183,380,215,30,'Linear([X; M1]) + softmax',GOLD,11)
box(d,446,380,261,30,'Linear([X; M2]) + softmax',GOLD,11)
a((290,435),(290,410));a((576,435),(576,410))
a((92,435),(92,421),(170,421),(170,395),(183,395))
a((92,421),(92,368),(722,368),(722,395),(707,395))
# Exact illustrative normalized distributions, all six tokens.
t(60,346,'Token',10,True,'start');t(60,325,'p(start)',10,True,'start');t(60,304,'p(end)',10,True,'start')
for k,(s,p,q) in enumerate(zip(['0: Tom','1: visited','2: Perth','3: on','4: Monday','5: .'],[.03,.02,.70,.15,.08,.02],[.02,.02,.38,.48,.08,.02])):
 x=225+k*88;t(x,346,s,10);t(x,325,f'{p:.2f}',11);t(x,304,f'{q:.2f}',11)
t(380,275,'Enumerate valid (i, j): i <= j, length <= 15. Score = p(start=i) x p(end=j).',10)
a((380,248),(380,225))
panel(16,209,'5  Top-10 candidates -> LambdaMART reranker -> answer')
# Candidate table.
t(52,175,'Candidate (i, j)',10,True,'start');t(238,175,'Base',10,True)
for y,s,v in [(153,'Perth on (2, 3)','0.336'),(131,'Perth (2, 2)','0.266'),(109,'on (3, 3)','0.072')]:
 if y==131:d.add(Rect(38,y-5,230,19,fillColor=HexColor(GOLD),strokeColor=None))
 t(48,y,s,10,False,'start');t(238,y,v,10)
t(151,87,'... 7 more candidates',10)
a((273,136),(296,136));heat(300,109,5,3,9)
t(321,167,'10 x 30',10,True);t(321,95,'features',10)
# miniature boosted trees
for x in (384,434,484):
 d.add(Circle(x,149,4,fillColor=HexColor('#A48ABB'),strokeColor=None))
 for dx in (-14,14):
  d.add(Line(x,145,x+dx,128,strokeColor=HexColor(INK)))
  d.add(Rect(x+dx-4,120,8,8,fillColor=HexColor(PURPLE),strokeColor=HexColor(INK),strokeWidth=.7))
t(434,177,'LambdaMART',11,True);t(434,103,'Sum of learned trees',9)
a((346,136),(366,136));a((506,136),(532,136))
box(d,536,103,74,61,['Scores','0.4','1.7','-0.2'],PURPLE,9)
a((610,136),(632,136));box(d,636,115,88,43,['argmax','Perth'],GOLD,12,True)
t(380,65,'Features: base score/rank, boundaries/length, overlap, question type, tweet region, flags.',9.5)
t(380,44,'One scalar score per candidate, not a class label. Reranker fitted on OOF candidates.',9.5)
t(380,28,'Illustrative probabilities, attention weights and reranker scores; gold is not an inference input.',9)
d.scale(.72,.72);d.width*=.72;d.height*=.72
renderPDF.drawToFile(d,str(OUT/'tweetqa_architecture_vertical.pdf'))
renderSVG.drawToFile(d,str(OUT/'tweetqa_architecture_vertical.svg'))
print(OUT)
