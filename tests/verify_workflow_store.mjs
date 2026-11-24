import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
const frontend = fileURLToPath(new URL('../frontend/', import.meta.url));
const require = createRequire(path.join(frontend, 'package.json'));
const ts = require('typescript');
const root = path.join(frontend, 'src');
const clone=(v)=>JSON.parse(JSON.stringify(v));
function makeStore(initializer){
 let state;
 const get=()=>state;
 const set=(value)=>{const partial=typeof value==='function'?value(state):value;state={...state,...partial};};
 const store=(selector)=>selector(state);
 store.getState=get;store.setState=set;
 state=initializer(set,get);return store;
}
function loadTs(file,mocks){
 const source=fs.readFileSync(file,'utf8');
 const js=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
 const module={exports:{}};
 const localRequire=(id)=>{if(id in mocks)return mocks[id]; if(id==='zustand')return{create:makeStore}; if(id==='@xyflow/react')return{applyNodeChanges:(changes,nodes)=>nodes.filter(n=>!changes.some(c=>c.type==='remove'&&c.id===n.id)).map(n=>{const c=changes.find(c=>c.type==='select'&&c.id===n.id);return c?{...n,selected:c.selected}:n;}),applyEdgeChanges:(_,x)=>x};
  if(id.startsWith('.'))return loadTs(path.resolve(path.dirname(file),id)+'.ts',mocks);throw Error('Unexpected import '+id);};
 vm.runInNewContext(js,{module,exports:module.exports,require:localRequire,console,Date,Math,Set,Map,Promise,setTimeout,clearTimeout,window:{setInterval,clearInterval,setTimeout,clearTimeout},crypto:globalThis.crypto},{filename:file});
 return module.exports;
}
function script(){return{project_id:'mock-project',mode:'video_dubbing',source_lang:'ja',glossary:{key:'译词'},characters:{A:{display_name:'A',emotion_refs:{}}},media:{duration:5,source_video:'/workspace/mock/input.mp4',vocals_wav:'/workspace/mock/dialogue.wav'},segments:[{id:0,start:0,end:2,speaker:'A',orig_text:'どうしたんです？',target_text:'どうしたんです？',target_lang:'zh',words:[],alignment_needs_review:true,speaker_needs_review:false,segment_timing_needs_review:false,emotion:'neutral',keep_original:false,speed_factor:1,status:'pending',tts_wav:null}]};}
function fixture(overrides={}){
 const calls=[]; const initial=script();
 const project=makeStore((set)=>({script:initial,isDirty:false,backendStatus:'online',gpuBusy:false,isLoading:false,isSynthesizing:false,setScript:(s)=>set({script:s,isDirty:false}),bumpAudioVersion:()=>{}}));
 const api={
  saveProjectScript:async(s)=>{calls.push(['save',clone(s)]);return clone(s);},
  translateProject:async(id,options)=>{calls.push(['translate',id,clone(options)]);const s=clone(project.getState().script);s.segments[0].target_text='怎么了？';s.segments[0].target_lang=options.target_lang;s.segments[0].status='pending';return{script:s,translated_ids:[0],skipped_count:0,errors:[]};},
  parseProjectMedia:async()=>{calls.push(['parse']);return clone(project.getState().script);},
  synthesizeSegments:async(id,ids)=>{calls.push(['synthesize',id,clone(ids)]);const s=clone(project.getState().script);s.segments.forEach(x=>{x.status='ready';x.tts_wav='/workspace/mock/tts.wav';});return{script:s,synthesized_ids:ids,overflow_segments:[],errors:[]};},
  renderVideo:async(id)=>{calls.push(['render',id]);return{output_video:'/workspace/mock/output.mp4',duration:5};},
  fetchParseStatus:async()=>({active:false,logs:[],log_count:0}),subscribeParseStatus:()=>()=>{},
  ...overrides,
 };
 const workflow=loadTs(root+'/store/useWorkflowStore.ts',{'../services/api':api,'./useProjectStore':{useProjectStore:project},'../utils/synthesisIssues':{synthesisIssues:()=>[]}}).useWorkflowStore;
 workflow.getState().selectNode('banquet');
 return{calls,project,workflow,api};
}
const results=[];
async function check(name,fn){try{await fn();results.push({name,passed:true});}catch(error){results.push({name,passed:false,error:error.stack});}}
await check('Translate-only saves source, calls target-language translation, skips parse/TTS/render',async()=>{
 const f=fixture();f.project.setState({isDirty:true});f.project.getState().script.segments[0].orig_text='手动修改的原文';f.project.getState().script.glossary={Beacon:'信标'};f.project.getState().script.characters.A.display_name='已编辑角色';f.workflow.getState().selectNode('llm_tagger');f.workflow.getState().updatePipelineConfig({targetLang:'en'});await f.workflow.getState().runPipeline({parseOnly:true});
 assert.equal(f.calls[0][1].segments[0].orig_text,'手动修改的原文');assert.equal(f.calls[0][1].glossary.Beacon,'信标');assert.equal(f.calls[0][1].characters.A.display_name,'已编辑角色');
 assert.deepEqual(f.calls.map(x=>x[0]),['save','translate']);assert.equal(f.calls[1][2].target_lang,'en');assert.equal(f.project.getState().script.segments[0].target_text,'怎么了？');assert.equal(f.workflow.getState().isPipelineRunning,false);
 assert.notEqual(f.workflow.getState().nodes.find(n=>n.id==='banquet').data.status,'done');
});
await check('Translate full workflow calls only saved source translation then synth and render',async()=>{
 const f=fixture();f.workflow.getState().selectNode('llm_tagger');f.workflow.getState().updatePipelineConfig({targetLang:'zh'});await f.workflow.getState().runPipeline({parseOnly:false});
 const names=f.calls.map(x=>x[0]);assert(!names.includes('parse'));assert(names.indexOf('translate')<names.indexOf('synthesize'));assert(names.indexOf('synthesize')<names.indexOf('render'));assert.equal(f.workflow.getState().isPipelineRunning,false);
});
await check('Offline start is blocked without API or simulation',async()=>{
 const f=fixture();f.project.setState({backendStatus:'offline'});f.workflow.getState().selectNode('llm_tagger');await f.workflow.getState().runPipeline({parseOnly:true});assert.equal(f.calls.length,0);assert(!f.workflow.getState().nodes.some(n=>n.data.status==='done'));assert.equal(f.workflow.getState().isPipelineRunning,false);
});
await check('Synthesize rejects source-equal untranslated text for cross-language project',async()=>{
 const f=fixture();f.workflow.getState().selectNode('agent_tts');await f.workflow.getState().runPipeline({parseOnly:false});assert(!f.calls.some(x=>['parse','translate','synthesize','render'].includes(x[0])));assert.equal(f.workflow.getState().isPipelineRunning,false);
});
await check('Render rejects missing ready WAV instead of silent empty movie',async()=>{
 const f=fixture();f.workflow.getState().selectNode('ffmpeg_mixer');await f.workflow.getState().runPipeline({parseOnly:false});assert(!f.calls.some(x=>x[0]==='render'));assert.equal(f.workflow.getState().isPipelineRunning,false);
});
await check('Running workflow blocks config edits and a second launch',async()=>{
 let release;const gate=new Promise(resolve=>release=resolve);const f=fixture({translateProject:async(id,options)=>{f.calls.push(['translate',id,options]);await gate;const s=clone(f.project.getState().script);s.segments[0].target_text='译文';return{script:s,translated_ids:[0],errors:[],skipped_count:0};}});
 f.workflow.getState().selectNode('llm_tagger');f.workflow.getState().updatePipelineConfig({targetLang:'zh'});const first=f.workflow.getState().runPipeline({parseOnly:true});await new Promise(resolve=>setImmediate(resolve));
 assert.equal(f.workflow.getState().isPipelineRunning,true);f.workflow.getState().updatePipelineConfig({targetLang:'en'});assert.equal(f.workflow.getState().pipelineConfig.targetLang,'zh');await f.workflow.getState().runPipeline({parseOnly:true});release();await first;assert.equal(f.calls.filter(x=>x[0]==='translate').length,1);
});
await check('Render-only saves and exports without parsing, translation or synthesis', async () => {
 const f=fixture(); const s=clone(f.project.getState().script); s.segments[0].status='ready'; s.segments[0].tts_wav='ready.wav'; f.project.setState({script:s});
 f.workflow.getState().selectNode('ffmpeg_mixer'); await f.workflow.getState().runPipeline();
 assert.deepEqual(f.calls.map(x=>x[0]),['save','render']);
});
await check('Synthesize start uses existing translation and skips prior steps', async () => {
 const f=fixture(); f.project.getState().script.segments[0].target_text='怎么了？';
 f.workflow.getState().selectNode('agent_tts'); await f.workflow.getState().runPipeline();
 assert.deepEqual(f.calls.map(x=>x[0]),['save','synthesize','render']);
});
await check('Partial translation preserves successes and stops all downstream work', async () => {
 for (const parseOnly of [true,false]) {
  const f=fixture({translateProject:async()=>{f.calls.push(['translate']); const s=clone(f.project.getState().script);s.segments[0].target_text='成功译文';return {script:s,translated_ids:[0],skipped_count:1,errors:['missing row']};}});
  f.workflow.getState().selectNode('llm_tagger');await f.workflow.getState().runPipeline({parseOnly});
  assert.deepEqual(f.calls.map(x=>x[0]),['save','translate']);assert.equal(f.project.getState().script.segments[0].target_text,'成功译文');
  const state=f.workflow.getState(); assert.equal(state.isPipelineRunning,false);assert.equal(state.nodes.find(n=>n.id==='llm_tagger').data.status,'error');assert(!state.nodes.some(n=>n.data.status==='queued'));
  assert(!state.pipelineLogs.some(x=>x.level==='success' && x.message.includes('仅生成剧本已完成')));
 }
});
await check('Failed save stops before translation and resets running state', async () => {
 const f=fixture({saveProjectScript:async()=>{f.calls.push(['save']);throw Error('disk full');}});
 f.workflow.getState().selectNode('llm_tagger');await f.workflow.getState().runPipeline({parseOnly:true});
 assert.deepEqual(f.calls.map(x=>x[0]),['save']);assert.equal(f.workflow.getState().isPipelineRunning,false);assert(!f.workflow.getState().edges.some(x=>x.animated));
});
await check('Full parse rejects untranslated output before invoking TTS', async () => {
 const f=fixture();await f.workflow.getState().runPipeline();
 assert.deepEqual(f.calls.map(x=>x[0]),['save','parse']);assert.equal(f.workflow.getState().nodes.find(n=>n.id==='agent_tts').data.status,'error');
});
await check('Partial synthesis cannot export a movie with missing dialogue', async () => {
 const f=fixture({synthesizeSegments:async()=>{f.calls.push(['synthesize']);return {script:clone(f.project.getState().script),synthesized_ids:[],errors:['failed']};}});
 f.project.getState().script.segments[0].target_text='译文';f.workflow.getState().selectNode('agent_tts');await f.workflow.getState().runPipeline();
 assert.deepEqual(f.calls.map(x=>x[0]),['save','synthesize']);assert.equal(f.workflow.getState().nodes.find(n=>n.id==='agent_tts').data.status,'error');
});
await check('Translate result does not overwrite a different open project', async () => {
 let release; const gate=new Promise(resolve=>release=resolve); const f=fixture({translateProject:async()=>{const s=clone(f.project.getState().script);await gate;return {script:s,translated_ids:[0],skipped_count:0,errors:[]};}});
 f.workflow.getState().selectNode('llm_tagger');const pending=f.workflow.getState().runPipeline({parseOnly:true});await new Promise(resolve=>setImmediate(resolve));
 f.project.setState({script:{...script(),project_id:'other'}});release();await pending;assert.equal(f.project.getState().script.project_id,'other');
});
await check('Empty source and unsupported script-only start stop before API calls', async () => {
 for (const startStep of ['translate','synthesize','render']) {
  const f=fixture();if(startStep==='translate')f.project.getState().script.segments[0].orig_text='';
  f.workflow.getState().selectNode({translate:'llm_tagger',synthesize:'agent_tts',render:'ffmpeg_mixer'}[startStep]);await f.workflow.getState().runPipeline({parseOnly:true});assert.equal(f.calls.length,0);
 }
});
function deferred() { let resolve; const promise=new Promise(done=>resolve=done); return {promise,resolve}; }
const flush=()=>new Promise(resolve=>setImmediate(resolve));
await check('Pause waits for translation result, blocks TTS, and continues without retranslating',async()=>{
 const gate=deferred(); const entered=deferred();
 const f=fixture({translateProject:async()=>{f.calls.push(['translate']);entered.resolve();await gate.promise;const s=clone(f.project.getState().script);s.segments[0].target_text='译文';return{script:s,translated_ids:[0],skipped_count:0,errors:[]};}});
 f.workflow.getState().selectNode('llm_tagger');const run=f.workflow.getState().runPipeline();await entered.promise;
 f.workflow.getState().pausePipeline();assert.equal(f.workflow.getState().isPauseRequested,true);assert.equal(f.workflow.getState().isPipelinePaused,false);
 gate.resolve();await flush();const state=f.workflow.getState();assert.equal(state.isPipelinePaused,true);assert.equal(state.isPipelineRunning,true);assert.equal(f.project.getState().script.segments[0].target_text,'译文');
 assert.deepEqual(f.calls.map(x=>x[0]),['save','translate']);assert(!state.edges.some(x=>x.animated));
 state.selectNode('ffmpeg_mixer');state.updatePipelineConfig({targetLang:'en'});assert.equal(f.workflow.getState().pipelineConfig.targetLang,'zh');await state.runPipeline();
 state.resumePipeline();await run;assert.deepEqual(f.calls.map(x=>x[0]),['save','translate','synthesize','render']);assert.equal(f.workflow.getState().isPipelinePaused,false);assert.equal(f.workflow.getState().isPauseRequested,false);
});
await check('Ending a paused run keeps results and never starts downstream steps',async()=>{
 const gate=deferred(); const f=fixture({translateProject:async()=>{f.calls.push(['translate']);await gate.promise;const s=clone(f.project.getState().script);s.segments[0].target_text='已保存译文';return{script:s,translated_ids:[0],skipped_count:0,errors:[]};}});
 f.workflow.getState().selectNode('llm_tagger');const run=f.workflow.getState().runPipeline();await flush();f.workflow.getState().pausePipeline();gate.resolve();await flush();assert.equal(f.workflow.getState().isPipelinePaused,true);
 f.workflow.getState().stopPipeline();await run;assert.deepEqual(f.calls.map(x=>x[0]),['save','translate']);assert.equal(f.project.getState().script.segments[0].target_text,'已保存译文');
 assert.equal(f.workflow.getState().isPipelineRunning,false);assert.equal(f.workflow.getState().isStopRequested,false);assert(!f.workflow.getState().nodes.some(x=>x.data.status==='queued'));
});
await check('Pause during synthesis prevents export until resumed',async()=>{
 const gate=deferred();const f=fixture({synthesizeSegments:async()=>{f.calls.push(['synthesize']);await gate.promise;const s=clone(f.project.getState().script);s.segments[0].status='ready';s.segments[0].tts_wav='ready.wav';return{script:s,synthesized_ids:[0],errors:[]};}});
 f.project.getState().script.segments[0].target_text='译文';f.workflow.getState().selectNode('agent_tts');const run=f.workflow.getState().runPipeline();await flush();f.workflow.getState().pausePipeline();gate.resolve();await flush();assert.equal(f.workflow.getState().isPipelinePaused,true);assert(!f.calls.some(x=>x[0]==='render'));
 f.workflow.getState().resumePipeline();await run;assert.equal(f.calls.filter(x=>x[0]==='render').length,1);
});
await check('Pending pause can be cancelled while the backend step is running',async()=>{
 const gate=deferred();const f=fixture({translateProject:async()=>{f.calls.push(['translate']);await gate.promise;const s=clone(f.project.getState().script);s.segments[0].target_text='译文';return{script:s,translated_ids:[0],skipped_count:0,errors:[]};}});
 f.workflow.getState().selectNode('llm_tagger');const run=f.workflow.getState().runPipeline();await flush();f.workflow.getState().pausePipeline();f.workflow.getState().resumePipeline();gate.resolve();await run;assert(f.calls.some(x=>x[0]==='render'));assert.equal(f.workflow.getState().isPipelinePaused,false);
});
await check('Stop while backend is running waits for saved results then ends',async()=>{
 const gate=deferred();const f=fixture({translateProject:async()=>{f.calls.push(['translate']);await gate.promise;return{script:clone(f.project.getState().script),translated_ids:[0],skipped_count:0,errors:[]};}});
 f.workflow.getState().selectNode('llm_tagger');const run=f.workflow.getState().runPipeline();await flush();f.workflow.getState().stopPipeline();assert.equal(f.workflow.getState().isPipelineRunning,true);gate.resolve();await run;assert.deepEqual(f.calls.map(x=>x[0]),['save','translate']);assert.equal(f.workflow.getState().isPipelineRunning,false);
});
await check('Pause requested during failed step clears all control flags',async()=>{
 const gate=deferred();const f=fixture({translateProject:async()=>{await gate.promise;throw Error('LLM unavailable');}});
 f.workflow.getState().selectNode('llm_tagger');const run=f.workflow.getState().runPipeline();await flush();f.workflow.getState().pausePipeline();gate.resolve();await run;const state=f.workflow.getState();assert.equal(state.isPauseRequested,false);assert.equal(state.isPipelinePaused,false);assert.equal(state.isPipelineRunning,false);
});
await check('Pausing after the final script-only step completes without waiting forever',async()=>{
 const gate=deferred();const f=fixture({translateProject:async()=>{await gate.promise;return{script:clone(f.project.getState().script),translated_ids:[0],skipped_count:0,errors:[]};}});
 f.workflow.getState().selectNode('llm_tagger');const run=f.workflow.getState().runPipeline({parseOnly:true});await flush();f.workflow.getState().pausePipeline();gate.resolve();await run;assert.equal(f.workflow.getState().isPipelineRunning,false);assert.equal(f.workflow.getState().isPauseRequested,false);
});
await check('Pause during initial save prevents even the first backend stage from starting',async()=>{
 const gate=deferred();const f=fixture({saveProjectScript:async s=>{f.calls.push(['save']);await gate.promise;return clone(s);}});
 f.workflow.getState().selectNode('llm_tagger');const run=f.workflow.getState().runPipeline();f.workflow.getState().pausePipeline();gate.resolve();await flush();assert.equal(f.workflow.getState().isPipelinePaused,true);assert.deepEqual(f.calls.map(x=>x[0]),['save']);f.workflow.getState().stopPipeline();await run;
});
await check('Pause after media parsing preserves parsed script and waits before synthesis',async()=>{
 const gate=deferred();const f=fixture({parseProjectMedia:async()=>{f.calls.push(['parse']);await gate.promise;const s=clone(f.project.getState().script);s.segments[0].target_text='解析译文';return s;}});
 const run=f.workflow.getState().runPipeline();await flush();f.workflow.getState().pausePipeline();gate.resolve();await flush();assert.equal(f.workflow.getState().isPipelinePaused,true);assert.equal(f.project.getState().script.segments[0].target_text,'解析译文');assert.deepEqual(f.calls.map(x=>x[0]),['save','parse']);f.workflow.getState().resumePipeline();await run;assert.deepEqual(f.calls.map(x=>x[0]),['save','parse','synthesize','render']);
});
await check('New run works after ending a paused run with no stale pause continuation',async()=>{
 const gate=deferred();const f=fixture({saveProjectScript:async s=>{f.calls.push(['save']);await gate.promise;return clone(s);}});
 f.workflow.getState().selectNode('llm_tagger');const first=f.workflow.getState().runPipeline();f.workflow.getState().pausePipeline();gate.resolve();await flush();f.workflow.getState().stopPipeline();await first;
 await f.workflow.getState().runPipeline({parseOnly:true});assert.deepEqual(f.calls.map(x=>x[0]),['save','save','translate']);assert.equal(f.workflow.getState().isPipelineRunning,false);
});
await check('No selected node or unsupported node cannot launch backend work',async()=>{
 for(const nodeId of [null,'whisperx','virtual_stage_out','deleted-node']) {
  const f=fixture();f.workflow.getState().selectNode(nodeId);await f.workflow.getState().runPipeline();assert.equal(f.calls.length,0);assert.equal(f.workflow.getState().isPipelineRunning,false);
 }
});
await check('Selection highlights exactly one node and clearing selection removes the run origin',async()=>{
 const f=fixture();f.workflow.getState().selectNode('llm_tagger');assert.deepEqual(clone(f.workflow.getState().nodes.filter(n=>n.selected).map(n=>n.id)),['llm_tagger']);
 f.workflow.getState().selectNode('ffmpeg_mixer');assert.deepEqual(clone(f.workflow.getState().nodes.filter(n=>n.selected).map(n=>n.id)),['ffmpeg_mixer']);
 f.workflow.getState().resetGraphLayout();assert.deepEqual(clone(f.workflow.getState().nodes.filter(n=>n.selected).map(n=>n.id)),['ffmpeg_mixer']);
 f.workflow.getState().selectNode(null);assert(!f.workflow.getState().nodes.some(n=>n.selected));await f.workflow.getState().runPipeline();assert.equal(f.calls.length,0);
});
await check('Script hub selection starts synthesis using saved translation',async()=>{
 const f=fixture();f.project.getState().script.segments[0].target_text='译文';f.workflow.getState().selectNode('script_hub');await f.workflow.getState().runPipeline();assert.deepEqual(f.calls.map(x=>x[0]),['save','synthesize','render']);
});
await check('Media input selection runs parsing when only a script is requested',async()=>{
 const f=fixture();f.workflow.getState().selectNode('media_input');await f.workflow.getState().runPipeline({parseOnly:true});assert.deepEqual(f.calls.map(x=>x[0]),['save','parse']);
});
await check('Canvas keyboard selection and deselection update the launch origin',async()=>{
 const f=fixture();f.workflow.getState().onNodesChange([{type:'select',id:'banquet',selected:false},{type:'select',id:'llm_tagger',selected:true}]);
 assert.equal(f.workflow.getState().selectedNodeId,'llm_tagger');await f.workflow.getState().runPipeline({parseOnly:true});assert.deepEqual(f.calls.map(x=>x[0]),['save','translate']);
 f.workflow.getState().onNodesChange([{type:'select',id:'llm_tagger',selected:false}]);assert.equal(f.workflow.getState().selectedNodeId,null);
});
console.log(JSON.stringify(results,null,2));

if(results.some(x=>!x.passed))process.exitCode=1;
