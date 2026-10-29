"""Maintain input illustrations for Testing docs without importing or solving SCNSim.

Run with a documentation environment providing Plotly, Kaleido and local Chrome:
    python tests/docs_figures.py
The SVG is the Askr ImageViewer source; each HTML uses sibling plotly.min.js.
"""
from pathlib import Path
import argparse,ast,asyncio,hashlib,html,importlib.metadata,json,math,os,shutil,sys,tempfile

CAP=8_000_000
INK='#13243B';MUTED='#53647C';BLUE='#235AC7';TEAL='#087F8C';ORANGE='#AF5819';LINE='#CBD5E1';PAPER='#FFFFFF'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def function(tree,name):
    nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    if len(nodes)!=1:raise ValueError('expected one source function '+name)
    return nodes[0]
def kws(call):return {v.arg:v.value for v in call.keywords}
def call_named(node,name):
    calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id==name) or (isinstance(n.func,ast.Attribute) and n.func.attr==name))]
    if len(calls)!=1:raise ValueError('expected one fixture call '+name)
    return calls[0]
def quantity(node):
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):return quantity(node.left)
    if not isinstance(node,ast.BinOp) or not isinstance(node.op,ast.Mult):raise ValueError('fixture quantity syntax changed')
    return ast.literal_eval(node.left)
def matrix(node):
    lists=[n for n in ast.walk(node) if isinstance(n,ast.List)]
    if not lists:raise ValueError('matrix fixture missing')
    return ast.literal_eval(lists[0])
def component_values(fn):
    values={}
    for n in ast.walk(fn):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('capacitor','inductor','resistor'):
            k=kws(n);kind=n.func.attr;field={'capacitor':'capacitance','inductor':'inductance','resistor':'resistance'}[kind]
            values[ast.literal_eval(k['id'])]={'kind':kind,'value':quantity(k[field])}
    return values

def source_facts(root):
    files={};trees={};sources=[]
    required=['test_rlgc.py','test_direct_analysis.py','test_affine_parameters.py','test_solve_diagnostics.py']
    for name in required:
        raw=(root/name).read_bytes();trees[name]=ast.parse(raw,filename=name)
        files[name]=sha(raw)
    def cite(name,fn):
        n=function(trees[name],fn);entry={'path':'tests/'+name,'function':fn,'lines':[n.lineno,n.end_lineno],'sha256':files[name]};sources.append(entry);return n,entry
    f,r_cite=cite('test_rlgc.py','_coupled_rlgc');k=kws(call_named(f,'RLGC'))
    rlgc={'conductors':ast.literal_eval(k['conductors']),'reference':ast.literal_eval(k['reference_conductor']),
          'R':matrix(k['resistance_per_length']),'L':matrix(k['inductance_per_length']),
          'G':matrix(k['conductance_per_length']),'C':matrix(k['capacitance_per_length']),
          'GHz':quantity(k['extraction_frequency'])}
    q2d,q_cite=cite('test_rlgc.py','_q2d_bytes')
    block_assignments=[n.value for n in q2d.body if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='blocks']
    if len(block_assignments)!=1:raise ValueError('expected one literal Q2D blocks assignment')
    blocks=ast.literal_eval(block_assignments[0])
    titles={'Resistance Matrix':'R','Inductance Matrix':'L','Conductance Matrix':'G','Capacitance Matrix':'C'}
    if not isinstance(blocks,tuple) or len(blocks)!=4:raise ValueError('Q2D matrix block structure changed')
    seen=set()
    for block in blocks:
        if not isinstance(block,tuple) or len(block)!=2:raise ValueError('Q2D block structure changed')
        title,values=block
        if title not in titles or title in seen:raise ValueError('Q2D matrix titles changed or repeated')
        seen.add(title)
        if not isinstance(values,tuple) or len(values)!=2 or any(not isinstance(row,tuple) or len(row)!=2 for row in values):
            raise ValueError('Q2D matrix shape changed')
        if [list(row) for row in values]!=rlgc[titles[title]]:
            raise ValueError('manual and synthetic Q2D matrix differ: '+title)
    advanced,a_cite=cite('test_direct_analysis.py','_advanced_direct_plan');lc=component_values(advanced)
    matched,m_cite=cite('test_direct_analysis.py','_matched_zero_plan');matched_values=component_values(matched)
    port=kws(call_named(matched,'add_port'));matched_values['port']={'id':ast.literal_eval(port['id']),'role':ast.literal_eval(port['role']),'Z0':quantity(port['reference_impedance'])}
    affine,af_cite=cite('test_affine_parameters.py','mapped_resonator')
    maps=[kws(n) for n in ast.walk(affine) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='AffineMap']
    af={'maps':[{'slope':quantity(k['slope']),'intercept':quantity(k['intercept']),'support':[quantity(x) for x in k['support'].elts]} for k in maps],
        'baseline':quantity(kws(call_named(affine,'parameter'))['baseline'])}
    opt,opt_cite=cite('test_affine_parameters.py','test_cma_records_rejected_and_authorized_affine_candidates')
    af['bounds']=[quantity(x) for x in kws(call_named(opt,'OptimizationVariable'))['bounds'].elts]
    outside,out_cite=cite('test_affine_parameters.py','test_ordinary_unauthorized_extrapolation_is_an_authoring_error')
    point=call_named(outside,'ParameterSet');af['outside']=quantity(point.args[0].values[0])
    diag,d_cite=cite('test_solve_diagnostics.py','test_approximate_identity_solution_distinguishes_the_two_metrics')
    assignments={n.targets[0].id:n.value for n in ast.walk(diag) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    eye=assignments['a'];order=ast.literal_eval(eye.args[0]);b=ast.literal_eval(assignments['b'].args[0]);x=ast.literal_eval(assignments['x'].args[0])
    if order!=2:raise ValueError('diagnostic illustration requires the assigned identity fixture')
    error=[abs(xi-bi) for xi,bi in zip(x,b,strict=True)];scale=[abs(xi)+abs(bi) for xi,bi in zip(x,b,strict=True)]
    diag_values={'A':[[1,0],[0,1]],'b':b,'x':x,'error':error,'scale':scale,'component_ratios':[e/s for e,s in zip(error,scale,strict=True)],'aggregate':max(error)/max(scale)}
    return {'rlgc':rlgc,'lc':lc,'matched':matched_values,'affine':af,'diagnostics':diag_values}, {'rlgc':[r_cite,q_cite],'lc':[a_cite,m_cite],'affine':[af_cite,opt_cite,out_cite],'diagnostics':[d_cite]}, None

def base(title,subtitle,*,height=820,xrange=(0,14),yrange=(0,9)):
    fig=go.Figure();fig.update_layout(width=1400,height=height,paper_bgcolor=PAPER,plot_bgcolor=PAPER,
        font=dict(family='Noto Sans CJK TC, Noto Sans CJK, Arial, sans-serif',size=17,color=INK),
        margin=dict(l=28,r=28,t=25,b=25),showlegend=False,hovermode='closest',dragmode='pan',
        xaxis=dict(range=xrange,visible=False,fixedrange=False),yaxis=dict(range=yrange,visible=False,fixedrange=False))
    text(fig,xrange[0]+min(.12,(xrange[1]-xrange[0])*.009),yrange[1]-.28,title,26,anchor='left')
    text(fig,xrange[0]+min(.12,(xrange[1]-xrange[0])*.009),yrange[1]-.78,subtitle,15,color=MUTED,anchor='left')
    return fig

def text(fig,x,y,value,size=17,color=INK,anchor='center'):
    fig.add_annotation(x=x,y=y,text=value,showarrow=False,font=dict(size=size,color=color),xanchor=anchor,align='left' if anchor=='left' else 'center')
def line(fig,xs,ys,color=INK,width=3,dash=None,hover=None):
    fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines',line=dict(color=color,width=width,dash=dash),
        text=hover,hoverinfo='text' if hover else 'skip',showlegend=False))
def node(fig,x,y,color=BLUE,label=None):
    fig.add_trace(go.Scatter(x=[x],y=[y],mode='markers',marker=dict(size=10,color=color),text=[label or 'declared node'],hovertemplate='%{text}<extra></extra>'))
def rect(fig,x0,y0,x1,y1,fill='#F5F8FC',color=LINE):
    fig.add_shape(type='rect',x0=x0,y0=y0,x1=x1,y1=y1,line=dict(color=color,width=1),fillcolor=fill,layer='below')
def ground(fig,x,y):
    line(fig,[x,x],[y,y-.16],MUTED,2)
    for offset,width in [(0,.35),(.10,.24),(.20,.12)]:line(fig,[x-width,x+width],[y-.16-offset]*2,MUTED,2)
def capacitor_v(fig,x,top,bottom,label):
    mid=(top+bottom)/2;line(fig,[x,x],[top,mid+.1]);line(fig,[x-.28,x+.28],[mid+.1]*2,BLUE)
    line(fig,[x-.28,x+.28],[mid-.1]*2,BLUE);line(fig,[x,x],[mid-.1,bottom]);text(fig,x,bottom-.28,label,16)
def inductor_v(fig,x,top,bottom,label):
    hi=top-.22;lo=bottom+.22;line(fig,[x,x],[top,hi]);line(fig,[x,x],[lo,bottom])
    t=[i*8*math.pi/100 for i in range(101)]
    line(fig,[x+.2*math.sin(v) for v in t],[hi-(hi-lo)*i/100 for i in range(101)],TEAL);text(fig,x,bottom-.28,label,16)
def capacitor_h(fig,left,right,y,label):
    mid=(left+right)/2;line(fig,[left,mid-.1],[y,y]);line(fig,[mid+.1,right],[y,y]);line(fig,[mid-.1]*2,[y-.26,y+.26],BLUE);line(fig,[mid+.1]*2,[y-.26,y+.26],BLUE);text(fig,mid,y+.45,label,16)
def inductor_h(fig,left,right,y,label):
    mid=(left+right)/2;a=mid-.55;b=mid+.55;line(fig,[left,a],[y,y]);line(fig,[b,right],[y,y])
    t=[i*8*math.pi/100 for i in range(101)];line(fig,[a+(b-a)*i/100 for i in range(101)],[y+.18*math.sin(v) for v in t],TEAL);text(fig,mid,y-.45,label,16)

def rlgc_figure(data):
    fig=base('兩個訊號導體，共用一個參考導體','合成 Q2D CSV 的測試輸入；線條僅表示拓撲，不代表截面或實體尺寸。',height=850)
    for y,name,csv_name,color in [(6.7,'readout','trace_a',BLUE),(5.65,'filter','trace_b',TEAL)]:
        line(fig,[1.2,12.6],[y,y],color,5,hover=f'{csv_name} → {name}')
        node(fig,1.2,y,color,name);node(fig,12.6,y,color,name)
        text(fig,1.2,y+.35,f'{csv_name} → {name}',18,color,anchor='left')
    line(fig,[1.2,12.6],[4.65,4.65],MUTED,3);ground(fig,7,4.65)
    text(fig,1.2,4.2,'reference_conductor = ground',17,MUTED,anchor='left')
    text(fig,12.6,4.15,f"extraction_frequency = {data['GHz']:g} GHz",16,MUTED,anchor='right')
    units={'R':'Ω/m','L':'nH/m','G':'S/m','C':'pF/m'}
    for index,name in enumerate(('R','L','G','C')):
        x0=.1+index*3.5;rect(fig,x0,.65,x0+3.25,3.65)
        text(fig,x0+.2,3.3,f"{name}  [{units[name]}]",21,anchor='left')
        for column,label in enumerate(data['conductors']):text(fig,x0+1.35+column,2.75,label,13,MUTED)
        for row,label in enumerate(data['conductors']):
            y=2.15-row*.75;text(fig,x0+.15,y,label,13,MUTED,anchor='left')
            for column in range(2):
                value=data[name][row][column];text(fig,x0+1.35+column,y,f'{value:g}',21,BLUE if row!=column and value else INK)
    text(fig,.2,.25,'矩陣列／欄順序皆為 readout、filter。CSV 不含線寬、間隙、基板或線長。',14,MUTED,anchor='left')
    return fig

def lc_figure(lc,matched):
    fig=base('兩種獨立的小型測試網路','上：耦合 LC 的節點 a、b；下：帶 Port 的匹配 RLC。數字皆為測試宣告的輸入值。',height=1000,yrange=(0,11))
    rect(fig,.15,4.1,13.85,9.55);text(fig,.4,9.25,'_advanced_direct_plan()  ·  沒有宣告 Port',17,MUTED,anchor='left')
    for x,label in [(3,'a'),(11,'b')]:
        line(fig,[x,x],[7.8,8.75]);node(fig,x,7.8,label=label);text(fig,x-.4,7.8,label,22,BLUE)
        line(fig,[x,x],[7.8,6.9]);line(fig,[x-1,x+1],[6.9,6.9])
    capacitor_h(fig,3,11,8.75,f"{lc['coupling_capacitor']['value']:g} fF")
    inductor_h(fig,3,11,7.8,f"{lc['coupling_inductor']['value']:g} nH")
    for x,c,l in [(3,'capacitor_a','inductor_a'),(11,'capacitor_b','inductor_b')]:
        capacitor_v(fig,x-1,6.9,5.2,f"{lc[c]['value']:g} fF")
        inductor_v(fig,x+1,6.9,5.2,f"{lc[l]['value']:g} nH")
        line(fig,[x-1,x-1],[5.2,4.65]);line(fig,[x+1,x+1],[5.2,4.65])
    line(fig,[2,12],[4.65,4.65]);ground(fig,7,4.65)
    rect(fig,.15,.2,13.85,3.65);text(fig,.4,3.32,'_matched_zero_plan()  ·  獨立的 Spp 零點 fixture',17,MUTED,anchor='left')
    fig.add_shape(type='circle',x0=1,y0=1.95,x1=1.7,y1=2.65,line=dict(color=BLUE,width=3),fillcolor='#EDF3FF')
    text(fig,1.35,2.3,'p',19,BLUE);line(fig,[1.7,11.5],[2.3,2.3]);node(fig,4,2.3,label='p')
    text(fig,1.05,1.2,f"Port p · {matched['port']['role']}<br>Z₀ = {matched['port']['Z0']:g} Ω",15,MUTED)
    x=5;line(fig,[x,x],[2.3,1.85]);fig.add_shape(type='rect',x0=x-.2,y0=1.0,x1=x+.2,y1=1.85,line=dict(color=ORANGE,width=2),fillcolor=PAPER);line(fig,[x,x],[1,.75]);text(fig,x,.43,f"R = {matched['resistor']['value']:g} Ω",16)
    capacitor_v(fig,8,2.3,.75,f"C = {matched['capacitor']['value']:g} fF")
    inductor_v(fig,11,2.3,.75,f"L = {matched['inductor']['value']:g} nH")
    line(fig,[5,12.5],[.75,.75]);ground(fig,12.5,.75)
    text(fig,8,1.1,'',12)
    return fig

def affine_figure(data):
    fig=base('support 與最佳化 bounds 是不同的區間','同一個 length 參數同時供應兩個電容；圖上的點是指定輸入，不是最佳化軌跡。',height=780,xrange=(.885,1.12),yrange=(0,8))
    lo,hi=data['bounds'];support=data['maps'][0]['support']
    line(fig,[lo,hi],[5.55]*2,BLUE,9,hover=f'OptimizationVariable bounds = [{lo}, {hi}] m')
    for point in (lo,hi):line(fig,[point,point],[5.4,5.7],BLUE,3)
    text(fig,.89,6.17,f'最佳化 bounds：[{lo:g}, {hi:g}] m',20,BLUE,anchor='left')
    line(fig,support,[4.15]*2,TEAL,14,hover=f'Both AffineMap supports = {support} m')
    for point in support:line(fig,[point,point],[3.95,4.35],TEAL,3)
    text(fig,.89,4.8,f'兩個 AffineMap 的 support：[{support[0]:g}, {support[1]:g}] m',20,TEAL,anchor='left')
    baseline=data['baseline'];outside=data['outside']
    line(fig,[baseline,baseline],[3.25,5.85],MUTED,1,'dot');line(fig,[outside,outside],[3.25,5.85],ORANGE,1,'dot')
    node(fig,baseline,4.15,BLUE,'baseline: 1 m, inside support');node(fig,outside,4.15,ORANGE,'explicit point: 1.05 m, outside support')
    text(fig,baseline,3.3,f'baseline<br>{baseline:g} m',18,BLUE)
    text(fig,outside,3.3,f'指定輸入 {outside:g} m<br>需明確 allow_extrapolation',18,ORANGE)
    rect(fig,.89,.4,1.115,2.55)
    text(fig,.895,2.1,'兩條實際的輸入映射',20,anchor='left')
    for row,item in enumerate(data['maps']):
        text(fig,.895,1.5-row*.6,f"C{row+1}(length) = ({item['slope']:g} fF/m) × length + {item['intercept']:g} fF",19,anchor='left')
    text(fig,1.108,.9,'兩個電容與 8 nH 並聯<br>Port 的 Z₀ = 50 Ω',17,MUTED,anchor='right')
    # Coordinate ticks describe the authored parameter, not any CAD dimensions.
    fig.update_xaxes(visible=True,range=(.885,1.12),tickvals=[lo,support[0],baseline,support[1],outside,hi],
        ticktext=[f'{v:g}' for v in [lo,support[0],baseline,support[1],outside,hi]],showgrid=False,zeroline=False,
        showline=False,title='length 參數（m）',tickfont=dict(size=15,color=MUTED))
    return fig

def diagnostics_figure(data):
    fig=base('同一份殘差，兩種量測方式','一個 RHS、兩個分量：A = I₂，b = [1, 10⁻²⁰]ᵀ，指定近似解 x̃ = [1, 0]ᵀ。',height=800)
    rect(fig,.3,3.2,13.7,7.35)
    columns=[(.7,'分量'),(3.0,'b'),(5.5,'x̃'),(8.0,'|Ax̃ − b|'),(10.4,'|A| |x̃| + |b|'),(12.7,'逐分量比值')]
    for x,label in columns:text(fig,x,6.9,label,18,MUTED)
    values=[['1','1','1','0','2','0'],['2','10⁻²⁰','0','10⁻²⁰','10⁻²⁰','1']]
    for row,items in enumerate(values):
        y=5.65-row*1.25
        for (x,_),value in zip(columns,items,strict=True):text(fig,x,y,value,25,ORANGE if row==1 and x==12.7 else INK)
    line(fig,[.6,13.4],[6.4,6.4],LINE,1)
    rect(fig,.3,.6,6.7,2.75,fill='#EDF3FF',color='#C1D1F5');rect(fig,7,.6,13.7,2.75,fill='#FFF4E9',color='#EDCCA8')
    text(fig,.6,2.28,'aggregate：max(error) / max(scale)',18,BLUE,anchor='left')
    text(fig,3.5,1.35,'5 × 10⁻²¹',31,BLUE)
    text(fig,7.3,2.28,'逐 RHS 的 componentwise 診斷',18,ORANGE,anchor='left')
    text(fig,10.3,1.35,'[1]',31,ORANGE)
    text(fig,.4,.15,'第二分量未被近似解保留。componentwise 是診斷，不取代原本 aggregate／τ 判定。',15,MUTED,anchor='left')
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1],help='Repository root containing the public tests fixtures')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'figures')
    parser.add_argument('--preview-output',type=Path,help='Optional private primary-figure PNG and renderer closure record')
    parser.add_argument('--chrome',type=Path,default=Path('/opt/google/chrome/chrome'),help='Chrome binary; use the binary rather than a shell wrapper that redirects the DevTools pipe')
    args=parser.parse_args();facts,citations,commit=source_facts(args.source_root/'tests')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    scratch=Path(tempfile.mkdtemp(prefix='dfig-',dir='/tmp'))
    os.environ['TMPDIR']=str(scratch);tempfile.tempdir=None
    # Chrome's Unix-domain socket paths need a short temporary root.
    global go
    import plotly
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.offline import get_plotlyjs
    pio.defaults.mathjax=None
    import kaleido
    js=get_plotlyjs().encode('utf-8')
    script_bytes=Path(__file__).stat().st_size
    def write(name,raw):
        target=output/name
        current=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        prior=target.stat().st_size if target.exists() else 0
        if current-prior+len(raw)+script_bytes>CAP:raise RuntimeError('assigned 8MB source/asset/log allowance would be exceeded')
        target.write_bytes(raw)
        return {'path':'tests/figures/'+name,'bytes':len(raw),'sha256':sha(raw)}
    write('plotly.min.js',js)
    distribution=importlib.metadata.distribution('plotly')
    license_file=next(f for f in distribution.files if str(f).endswith('.dist-info/licenses/LICENSE.txt'))
    write('plotly-LICENSE.txt',Path(distribution.locate_file(license_file)).read_bytes())
    specs=[('rlgc-inputs',rlgc_figure(facts['rlgc']),'rlgc','合成 Q2D CSV 的兩個訊號導體、ground 參考與四個每單位長度矩陣。這是測試輸入，不是 CAD 幾何或場求解結果。'),
           ('lc-networks',lc_figure(facts['lc'],facts['matched']),'lc','上圖是 80/95 fF 的耦合 LC；下圖是獨立的匹配 RLC 與 Port。R 與 Port 參考阻抗為兩個獨立的 50 Ω 宣告。'),
           ('affine-domain',affine_figure(facts['affine']),'affine','length 的 baseline 為 1 m，support 為 [0.99,1.01] m；1.05 m 位於 support 外，須有明確授權。較寬的最佳化 bounds 不等於 support。'),
           ('residual-metrics',diagnostics_figure(facts['diagnostics']),'diagnostics','A=I 的多尺度 fixture：同一殘差的 aggregate 為 5×10⁻²¹，而逐 RHS 的 componentwise 診斷為 [1]。這裡呈現指定近似解的算式，沒有重新求解。')]
    records=[]
    for slug,fig,_,_ in specs:
        for index,trace in enumerate(fig.data):trace.uid=f'{slug}-{index}'
        fig.update_layout(uirevision=slug)
    async def render_all():
        rendered=[];preview_png=None;process=None;process_record={}
        async with kaleido.Kaleido(n=1,path=str(args.chrome),tmp_dir='/tmp',
                plotlyjs=str(output/'plotly.min.js'),mathjax=False,timeout=None) as renderer:
            process=renderer.subprocess
            stat=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
            process_record={'pid':process.pid,'pgid':int(stat[2]),'startticks':stat[19]}
            for index,(_,fig,_,_) in enumerate(specs):
                rendered.append(await renderer.calc_fig(fig,opts={'format':'svg'}))
                if index==0 and args.preview_output:
                    preview_png=await renderer.calc_fig(fig,opts={'format':'png'})
        process_record.update(returncode=process.wait(),waited=True,
            absent=not Path(f'/proc/{process.pid}').exists())
        remaining=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():continue
            try:
                stat=(entry/'stat').read_text().rsplit(')',1)[1].split()
                if int(stat[2])==process_record['pgid']:
                    remaining.append({'pid':int(entry.name),'state':stat[0],'startticks':stat[19]})
            except (FileNotFoundError,ProcessLookupError):pass
        process_record['remaining_owned_group']=remaining
        return rendered,preview_png,process_record
    svgs,preview_png,process_record=asyncio.run(render_all())
    if args.preview_output:
        args.preview_output.mkdir(parents=True,exist_ok=True)
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())+len(preview_png)+script_bytes>CAP:
            raise RuntimeError('assigned preview/asset cap would be exceeded')
        (args.preview_output/'rlgc-inputs.png').write_bytes(preview_png)
        (args.preview_output/'GENERATION-PROCESS.json').write_text(json.dumps(process_record,indent=2)+'\n')
    for (slug,fig,key,caption),svg in zip(specs,svgs,strict=True):
        svg_ref=write(slug+'.svg',svg)
        chart=pio.to_html(fig,full_html=False,include_plotlyjs=False,include_mathjax=False,div_id=slug,
            config={'displaylogo':False,'responsive':True,'scrollZoom':True,'toImageButtonOptions':{'format':'svg','filename':slug}})
        page=('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
              '<title>'+html.escape(caption)+'</title>'
              '<script src="plotly.min.js"></script><main>'+chart+'<p>'+html.escape(caption)+'</p></main></html>')
        html_ref=write(slug+'.html',page.encode('utf-8'))
        records.append({'id':slug,'kind':'authored-input or static-fixture illustration, not a new solver observation','caption_zh_TW':caption,
            'source':citations[key],'input_facts':facts[key] if key!='lc' else {'advanced':facts['lc'],'matched':facts['matched']},
            'svg':svg_ref,'interactive_html':html_ref,'same_figure':'SVG and HTML exported from this exact same Plotly Figure instance','uses_shared_js':'plotly.min.js'})
    # Chrome creates small anonymous backing files outside its profile directory.
    # Remove only this invocation's scratch after the browser was waited and
    # its owned group was observed empty.
    if not process_record['absent'] or process_record['remaining_owned_group']:
        raise RuntimeError('renderer still owns processes; retain its scratch')
    shutil.rmtree(scratch)
    provenance={'source_commit':commit,'figures':records,
        'source_checks':{'q2d_literal_blocks':'AST literal extraction; exact four titles/shapes and equality to the illustrated manual matrices'},'shared_plotly':{'path':'tests/figures/plotly.min.js','bytes':len(js),'sha256':sha(js)},
        'maintenance_script':{'path':'tests/docs_figures.py','sha256':sha(Path(__file__).read_bytes())},
        'versions':{'python':sys.version.split()[0],'plotly':plotly.__version__,'kaleido':importlib.metadata.version('kaleido')},
        'generation':{'solver_calls':0,'SCNSim_JAX_imports':sorted(k for k in sys.modules if k=='scnsim' or k.startswith('scnsim.') or k=='jax' or k.startswith('jax.')),
                      'local_chrome':True,'renderer_returned':True,'browser_process_waited':process_record['waited'],'browser_process_absent':process_record['absent'],'browser_owned_group_empty':not process_record['remaining_owned_group'],'temporary_renderer_directory_closed':True,'asset_cap_bytes':CAP},
        'render_contract':'SVG/PNG/JPEG ImageViewer uses the SVG; HTML is a separate offline interactive link, not ImageViewer input. Logical topology has no CAD dimensions.'}
    if provenance['generation']['SCNSim_JAX_imports']:raise RuntimeError('forbidden producer or JAX import occurred')
    write('provenance.json',(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n').encode())
    total=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())+script_bytes
    if args.preview_output:total+=sum(p.stat().st_size for p in args.preview_output.iterdir() if p.is_file())
    print(json.dumps({'figures':len(records),'assets_and_script_bytes':total,'cap':CAP,'provenance_sha256':sha((output/'provenance.json').read_bytes()),'renderer_closed':True,'solver_calls':0}))

if __name__=='__main__':main()
