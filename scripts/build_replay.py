"""Build replay.json from locally held retrieval traces; never downloads data."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

def build(traces,benchmark,locomo_data=None,baseline=None):
    speakers={}
    if benchmark=='locomo':
        if locomo_data is None:raise ValueError('LoCoMo requires the original locomo10.json to resolve speakers')
        for sample in json.loads(locomo_data.read_text()):
            c=sample['conversation'];speakers[str(sample['sample_id'])]=[c['speaker_a'],c['speaker_b']]
    entries=defaultdict(list)
    for path in sorted(traces.glob('*.json')):
        d=json.loads(path.read_text())
        for step in d['steps']:
            if 'retrieved_memory' not in step or step['retrieved_memory'] is None:
                raise ValueError(f'Missing recorded retrieval in {path.name}')
            query=step['query']
            if benchmark=='locomo':key=json.dumps([*speakers[str(d['user_id'])],query])
            else:key=json.dumps([query,str(step.get('qa_metadata',{}).get('question_date','') or '')])
            entries[key].append(step['retrieved_memory'])
    if not entries:raise ValueError('No retrieval entries found')
    result={'benchmark':benchmark,'source':'locally supplied retrieval traces','entries':dict(entries)}
    if baseline:result['baseline']=baseline
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--benchmark',choices=['locomo','longmemeval'],required=True)
    p.add_argument('--locomo-data',type=Path);p.add_argument('--baseline',choices=['simplemem','lightmem','amem','zep','mem0'])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=build(a.traces,a.benchmark,a.locomo_data,a.baseline)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False)
    print('Wrote',sum(map(len,result['entries'].values())),'retrieval entries. Keep this generated dataset local.')
