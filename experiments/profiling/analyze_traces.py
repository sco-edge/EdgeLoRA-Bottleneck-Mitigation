"""Read Nsight SQLite exports; correlate GPU activities with node NVTX ranges."""
import pathlib,sqlite3,json,bisect,statistics,collections,re
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('results', type=pathlib.Path, help='Folder containing main/*-profiled/trace.sqlite exports')
O=_parser.parse_args().results.resolve()
if not O.is_dir(): _parser.error('Results folder does not exist')
def union_ns(intervals):
    intervals=sorted(intervals)
    if not intervals:return 0
    total=0;a,b=intervals[0]
    for x,y in intervals[1:]:
        if x<=b:b=max(b,y)
        else:total+=b-a;a,b=x,y
    return total+b-a
def summary(values):
    return dict(n=len(values),median=statistics.median(values),mean=statistics.mean(values),values=values)
def analyze(folder):
    con=sqlite3.connect('file:'+str(folder/'trace.sqlite')+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    tables={r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    schema={t:[dict(r) for r in con.execute('pragma table_info('+t+')')] for t in sorted(tables)}
    strings={r['id']:r['value'] for r in con.execute('select * from StringIds')}
    def textrow(row):return row.get('text') or strings.get(row.get('textId'),'')
    nv=[]
    for rr in con.execute('select * from NVTX_EVENTS'):
        r=dict(rr);r['label']=textrow(r)
        if r.get('end') is not None and r['end']>=r['start']:nv.append(r)
    measures=[r for r in nv if r['label'].startswith('LCP/MEASURE/')]
    assert len(measures)==3,(folder,len(measures))
    nodes=collections.defaultdict(list)
    for r in nv:
        if r['label'].startswith('LCP/NODE/'):nodes[r['globalTid']].append(r)
    starts={}
    for tid,ls in nodes.items():ls.sort(key=lambda r:r['start']);starts[tid]=[r['start'] for r in ls]
    def node_for_api(r):
        if not r or r.get('globalTid') not in nodes:return None
        tid=r['globalTid'];i=bisect.bisect_right(starts[tid],r['start'])-1
        if i<0:return None
        n=nodes[tid][i]
        return n if n['start']<=r['start'] and r['end']<=n['end'] else None
    api_map=collections.defaultdict(list)
    for t in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
        if t in tables:
            for rr in con.execute('select * from '+t):
                r=dict(rr);r['api_table']=t;api_map[r['correlationId']].append(r)
    gpu=[]
    for t,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','memcpy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
        if t not in tables:continue
        for rr in con.execute('select * from '+t):
            r=dict(rr);r['activity_kind']=kind
            if kind=='kernel':r['kernel_name']=strings.get(r.get('demangledName'),strings.get(r.get('shortName'),''))
            cand=api_map.get(r.get('correlationId'),[])
            if r.get('globalPid') is not None:
                cand=[a for a in cand if (int(a.get('globalTid',0))>>24)==(int(r['globalPid'])>>24)]
            matched=[]
            for api in cand:
                n=node_for_api(api)
                if n is not None:matched.append((api,n))
            # Multiple runtime/driver API records may identify the same launch node.
            identities={(n['globalTid'],n['start'],n['end']) for _,n in matched}
            labels={n['label'] for _,n in matched}
            assert len(identities)<=1,(folder,r,identities)
            r['node_label']=next(iter(labels)) if labels else None
            r['node_identity']=next(iter(identities)) if identities else None
            r['category']=r['node_label'].split('|')[0].replace('LCP/NODE/','') if labels else 'UNSCOPED_'+kind.upper()
            gpu.append(r)
    results=[];raw=[]
    for m in sorted(measures,key=lambda r:r['start']):
        length=int(m['label'].split('/B')[-1]);a,b=m['start'],m['end'];wall=(b-a)/1e6
        acts=[r for r in gpu if a<=r['start'] and r['end']<=b]
        for r in acts:
            if r['node_identity']:assert a<=r['node_identity'][1]<=r['node_identity'][2]<=b,('node outside measurement',folder,r)
        crossings=[r for r in gpu if r['start']<b and r['end']>a and not(a<=r['start'] and r['end']<=b)]
        assert not crossings,('activity crosses synchronized measured interval',folder,length,len(crossings))
        ns=[r for ls in nodes.values() for r in ls if a<=r['start'] and r['end']<=b]
        cpu=[r for r in nv if r['label'].startswith('LCP/CPU/') and a<=r['start'] and r['end']<=b]
        cstats={}
        for label in sorted({r['label'] for r in cpu}):
            rr=[r for r in cpu if r['label']==label]
            cstats[label.replace('LCP/CPU/','')]=dict(count=len(rr),sum_ms=sum(r['end']-r['start'] for r in rr)/1e6)
        cats={}
        for cat in sorted({r['category'] for r in acts}|{r['label'].split('|')[0].replace('LCP/NODE/','') for r in ns}):
            rr=[r for r in acts if r['category']==cat]
            nn=[r for r in ns if r['label'].split('|')[0]=='LCP/NODE/'+cat]
            ks=[r for r in rr if r['activity_kind']=='kernel']
            cats[cat]=dict(node_count=len(nn),node_host_wall_ms=sum(r['end']-r['start'] for r in nn)/1e6,kernel_count=len(ks),kernel_sum_ms=sum(r['end']-r['start'] for r in ks)/1e6,gpu_activity_union_ms=union_ns([(r['start'],r['end']) for r in rr])/1e6,memcpy_count=sum(r['activity_kind']=='memcpy' for r in rr),memcpy_bytes=sum(r.get('bytes',0) for r in rr if r['activity_kind']=='memcpy'))
            if cat=='LORA_ACC':
                bs=[int(re.search(r'\|bytes=(\d+)',r['label']).group(1)) for r in nn]
                aliases=collections.Counter(re.search(r'\|alias=(\d+)',r['label']).group(1) for r in nn)
                cats[cat].update(sum_result_tensor_bytes=sum(bs),logical_src0_read_plus_dst_write_bytes=2*sum(bs),alias_counts=dict(aliases),result_size_histogram=dict(collections.Counter(bs)))
        kernels=[r for r in acts if r['activity_kind']=='kernel']
        assert len({r.get('globalPid') for r in kernels})==1
        assert len({r.get('contextId') for r in kernels})==1
        assert all(r['node_identity'] for r in kernels),('unscoped kernels',folder,length)
        assert cats['LORA_ACC']['node_count']==cats['LORA_ACC']['kernel_count'],('ACC node/kernel mismatch',folder,length)
        assert all('acc_f32' in r['kernel_name'] for r in kernels if r['category']=='LORA_ACC')
        kernel_sum=sum(r['end']-r['start'] for r in kernels)/1e6
        kernel_union=union_ns([(r['start'],r['end']) for r in kernels])/1e6
        lora_sum=sum(v['kernel_sum_ms'] for k,v in cats.items() if k.startswith('LORA_'))
        lora_union=union_ns([(r['start'],r['end']) for r in kernels if r['category'].startswith('LORA_')])/1e6
        busy=union_ns([(r['start'],r['end']) for r in acts])/1e6
        names=collections.defaultdict(lambda:dict(count=0,sum_ms=0))
        for r in kernels:
            d=names[r['kernel_name']];d['count']+=1;d['sum_ms']+=(r['end']-r['start'])/1e6
        row=dict(B_input=length,nvtx_wall_ms=wall,kernel_sum_ms=kernel_sum,kernel_union_ms=kernel_union,kernel_overlap_excess_ms=kernel_sum-kernel_union,gpu_activity_union_ms=busy,not_gpu_activity_ms=wall-busy,lora_kernel_sum_ms=lora_sum,lora_kernel_union_ms=lora_union,lora_union_div_wall=lora_union/wall,lora_fraction_of_kernel_sum=lora_sum/kernel_sum if kernel_sum else None,lora_kernel_sum_div_wall=lora_sum/wall,categories=cats,cpu_ranges=cstats,kernel_names=sorted([dict(name=k,**v) for k,v in names.items()],key=lambda d:-d['sum_ms']),kernel_count=len(kernels),unscoped_kernel_count=sum(r['category']=='UNSCOPED_KERNEL' for r in kernels),stream_ids=sorted({r.get('streamId',-1) for r in acts}),measurement_range=dict(start=a,end=b,globalTid=m['globalTid']),source=str(folder))
        results.append(row)
        raw.extend(dict(B_input=length,**r) for r in acts)
    (folder/'trace-analysis.json').write_text(json.dumps(dict(schema=schema,results=results),indent=2))
    with (folder/'gpu-activities.jsonl').open('w') as f:
        for r in raw:f.write(json.dumps(r)+'\n')
    con.close();return results

allrows=[]
for p in sorted((O/'main').glob('*-profiled')):
    if (p/'trace.sqlite').exists():
        for r in analyze(p):allrows.append(r)
groups=[]
for length in [16,128,256]:
    rr=[r for r in allrows if r['B_input']==length]
    if not rr:continue
    fields=['nvtx_wall_ms','kernel_sum_ms','kernel_union_ms','kernel_overlap_excess_ms','gpu_activity_union_ms','not_gpu_activity_ms','lora_kernel_sum_ms','lora_kernel_union_ms','lora_union_div_wall','lora_fraction_of_kernel_sum','lora_kernel_sum_div_wall','kernel_count','unscoped_kernel_count']
    group=dict(B_input=length,metrics={k:summary([r[k] for r in rr]) for k in fields},categories={},cpu_ranges={})
    for cat in sorted({k for r in rr for k in r['categories']}):
        group['categories'][cat]={k:summary([r['categories'].get(cat,{}).get(k,0) for r in rr]) for k in ['node_count','node_host_wall_ms','kernel_count','kernel_sum_ms','gpu_activity_union_ms','memcpy_count','memcpy_bytes']}
    for label in sorted({k for r in rr for k in r['cpu_ranges']}):group['cpu_ranges'][label]=summary([r['cpu_ranges'].get(label,{}).get('sum_ms',0) for r in rr])
    groups.append(group)
result=dict(groups=groups,rows=allrows,limits=['GPU sums and CPU host ranges overlap and are not additive wall-time components.','Logical ACC byte counts are algorithmic tensor accesses, not measured DRAM or NVMe traffic.','Nsight collection overhead must be assessed with analyze_runs.py; GPU percentages refer to traced runs.','No LoRA-disabled ablation or performance optimization is included.'])
result['complete']=len(allrows)==18 and len(groups)==3 and all(g['metrics']['nvtx_wall_ms']['n']==6 for g in groups)
result['verification']=dict(profiled_measurements=len(allrows),expected=18,all_kernels_attributed=True,acc_node_kernel_one_to_one=True,single_pid_context_per_measurement=True)
(O/'analysis-traces.json').write_text(json.dumps(result,indent=2))
for g in groups:print('B',g['B_input'],{k:round(v['median'],3) for k,v in g['metrics'].items()})
