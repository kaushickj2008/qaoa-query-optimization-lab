const $ = (id) => document.getElementById(id);
let problems = {}, current = null, busy = false;
const pct = (x) => `${(100*x).toFixed(2)}%`;
const number = (x) => x == null ? 'Not available' : Number(x).toFixed(3);

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'The request failed.');
  return data;
}
function settingsChanged() {
  const id = $('problem').value, p = problems[id];
  if (p) { $('description').textContent = p.description; $('qubits').textContent = `${p.qubits} qubits`; $('states').textContent = `${2**p.qubits} basis states`; }
  $('penalty-control').hidden = id === 'maxcut' || id.endsWith('_valid');
  $('budget-note').textContent = $('method').value === 'COBYLA' ? 'Function-evaluation budget. Not directly comparable to gradient-method iterations.' : $('method').value === 'ADAM' ? 'Maximum Adam updates; learning rate 0.05.' : 'Maximum iterations. Gradient and objective calls are counted separately.';
}
function chart(history) {
  const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 760 235'); svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `Energy across ${history.length} objective evaluations, starting ${number(history[0])}, best ${number(Math.min(...history))}`);
  function element(tag, attrs, text) { const node = document.createElementNS(ns, tag); for (const [k,v] of Object.entries(attrs)) node.setAttribute(k,v); if(text!==undefined) node.textContent=text; svg.append(node); return node; }
  let lo = Math.min(...history), hi = Math.max(...history); const pad = (hi-lo || 1)*.12; lo-=pad; hi+=pad;
  const x = (i) => 62 + i / Math.max(1,history.length-1)*680, y = (v) => 190-(v-lo)/(hi-lo)*170;
  for(let i=0;i<4;i++){ const value=lo+(hi-lo)*i/3; element('line',{x1:62,x2:742,y1:y(value),y2:y(value),stroke:'#e3ebee'}); element('text',{x:52,y:y(value)+4,'text-anchor':'end',fill:'#526970','font-size':12},value.toFixed(2)); }
  let best=Infinity; const minimum=history.map(v=>best=Math.min(best,v));
  for(const [data,color,width] of [[history,'#91a7b4',2],[minimum,'#006d68',3]]) element('polyline',{points:data.map((v,i)=>`${x(i)},${y(v)}`).join(' '),fill:'none',stroke:color,'stroke-width':width});
  element('circle',{cx:x(history.length-1),cy:y(minimum.at(-1)),r:4,fill:'#006d68'});
  element('text',{x:62,y:216,fill:'#526970','font-size':12},'1');
  element('text',{x:742,y:216,fill:'#526970','font-size':12,'text-anchor':'end'},String(history.length));
  $('energy-chart').replaceChildren(svg);
}
function render(result) {
  current=result;
  $('empty').hidden=true; $('output').hidden=false; $('download').disabled=false;
  $('result-title').textContent=result.problem_name;
  $('run-meta').textContent=`p = ${result.config.p} · ${result.config.method} · seed ${result.config.seed} · ${result.config.shots.toLocaleString()} simulated samples · ${result.elapsed_seconds.toFixed(1)} s`;
  const m=result.metrics, t=result.training;
  $('optimal').textContent=pct(m.optimal_probability); $('valid').textContent=pct(m.valid_probability);
  $('optimal-samples').textContent=`${m.optimal_samples} optimal samples out of ${result.config.shots}`;
  $('expected-label').textContent=m.expected_label; $('expected').textContent=number(m.expected_value);
  $('exact').textContent=`Exact ${result.config.problem==='maxcut'?'maximum cut':'minimum retained cost'}: ${m.exact_value}`;
  chart(t.energy_history);
  $('calls').textContent=`${t.energy_history.length} objective · ${t.gradient_evaluations} gradient calls`;
  const rows=[...result.distribution].sort((a,b)=>b.probability-a.probability).slice(0,12).map(row=>{
    const tr=document.createElement('tr');
    [row.bits,pct(row.probability),row.count,row.optimal?'Optimal':row.valid?'Valid':'Invalid',row.valid?number(row.value):'—'].forEach((value,i)=>{const td=document.createElement('td');td.textContent=String(value);if(i===0 && row.plan){const plan=document.createElement('div');plan.textContent=row.plan;plan.className='help';td.append(plan);}if(i===3)td.className=row.optimal?'optimal-state':row.valid?'':'invalid-state';tr.append(td);});return tr;
  });
  $('outcomes').replaceChildren(...rows);
  $('comparison').textContent=`Uniform selection among valid solutions gives ${pct(m.uniform_valid_optimal_probability)} optimal probability; this run gives ${pct(m.optimal_probability)} before filtering. This probability comparison does not establish quantum advantage.`;
  $('termination').textContent=`${t.success?'Optimizer stopping criterion reached.':'Optimizer did not converge within its stopping conditions.'} ${t.message} Neither status proves a global optimum.`;
  $('model-note').textContent=result.config.problem==='maxcut'?`Expected approximation ratio: ${pct(m.expected_approximation_ratio)}. Samples are drawn from ideal simulator probabilities.`:'Join costs are synthetic or from the paper example, not measured SQL execution times. Reported cost is conditional on a valid plan; root cost is omitted. Training uses penalty-normalized energy.';
  if(result.config.problem.endsWith('_valid')) $('model-note').textContent=`Experimental plan-index encoding with a Grover-style mixer. All 15 plans and costs were classically enumerated first; exact selection took ${(1000*m.exact_enumeration_seconds).toFixed(3)} ms including that preparation. Exact plan: ${m.exact_plan}. This preprocessing already solves the problem, so simulator timing is not a quantum speedup. Energy is divided by maximum plan cost.`;
  $('parameters').textContent=JSON.stringify({gamma:t.parameters[0],beta:t.parameters[1],variable_order:result.model.variables,cost_scale_divisor:result.model.scale},null,2);
}
$('setup').addEventListener('submit',async event=>{
  event.preventDefault(); if(busy)return;
  const config=Object.fromEntries(new FormData($('setup')));
  for(const key of ['p','seed','sample_seed','shots','maxiter'])config[key]=Number(config[key]);
  busy=true; $('fields').disabled=true; $('error').hidden=true; $('run').textContent='Running…';
  $('status-badge').textContent='RUNNING'; $('status').textContent='Training circuit parameters…';
  const started=Date.now();
  try {
    const job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});
    let snapshot;
    do { await new Promise(resolve=>setTimeout(resolve,900)); snapshot=await api(`/api/jobs/${job.id}`); $('status').textContent=`Training and evaluating · ${Math.round((Date.now()-started)/1000)} s elapsed${current?' · previous result shown below':''}`; } while(snapshot.status==='running');
    if(snapshot.status==='failed')throw new Error(snapshot.error);
    render(snapshot.result); $('status-badge').textContent='COMPLETE'; $('status').textContent='Results ready · ideal simulator';
  } catch(error) { $('error').textContent=error.message; $('error').hidden=false; $('status-badge').textContent='ERROR'; $('status').textContent='Could not complete the request. A disconnected run may still finish on the server.'; }
  finally {busy=false;$('fields').disabled=false;$('run').textContent='Run experiment';}
});
$('download').addEventListener('click',()=>{if(!current)return;const url=URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`qaoa-${current.config.problem}-p${current.config.p}-seed${current.config.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
for(const id of ['problem','method'])$(id).addEventListener('change',settingsChanged);
api('/api/problems').then(data=>{problems=data.problems;settingsChanged();}).catch(error=>{$('error').textContent=`Cannot connect to the local server: ${error.message}`;$('error').hidden=false;});
