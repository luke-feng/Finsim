import json
import tqdm

base_path = '/home/chaofeng/Documents/vscode/finsim/data/kg/1/'
prefilename = 'b'
results = []

for i in range(1, 4):
    file = base_path+prefilename+str(i)+'.json'
    with open(file, 'r') as f:
        predata = json.load(f)
        results.append(predata)

print(len(results))
outputs= []
par = tqdm.tqdm(total=len(results[0]), ncols=100)
with open(base_path+prefilename+'fin.json', 'w') as f:
    for i in range(len(results[0])):
        par.update(1)
        pre = {}
        t = ''
        l = 'null'
        output = {}
        for j in range(0,3):
            term = results[j][i]
            t = term['term']
            l = term['label']
            p = term['predicted_labels'][0]
            if p not in pre:
                pre[p] = 1
            else:
                pre[p] += 1
        a = sorted(pre.items(), key=lambda x:x[1], reverse=True)
        p_n = [s[0] for s in a]
        print(p_n)
        output['term'] = t
        output['label'] = l
        output['predicted_labels'] = p_n
        outputs.append(output)
    json.dump(outputs, f, indent=4)

