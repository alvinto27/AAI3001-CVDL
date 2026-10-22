"""Opt-in weight-only starts and fixed low-bpp curriculum (approved C1-C6).

Training uses the original photo schedule, OTF seeds, augmentation and shuffle.
Only the embedding config changes. Exactly one quarter of scheduled tile draws
uses the original config, before worker sharding/shuffling (not a batch quota).
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import lightning as L
from lsb_torch import TrainingImages

STAGES = ((.35,1),(.25,2),(.15,3))


def stage_for_epoch(epoch, stage_epochs=5, mode='low_bpp'):
    if type(epoch) is not int or not 0<=epoch<3*stage_epochs:
        raise ValueError('epoch outside the three-stage schedule')
    stage=epoch//stage_epochs
    maximum,weight=STAGES[stage] if mode=='low_bpp' else (.5,1)
    if mode not in ('low_bpp','original'):raise ValueError('unknown embedding schedule')
    return dict(stage=stage+1,bpp_min=.05,bpp_max=maximum,k1_weight=weight,
                original_fraction=.25 if mode=='low_bpp' else 1.)


def original_positions(count, seed, epoch, mode):
    if count%4:raise ValueError('exact quarter needs tiles_per_epoch divisible by four')
    if mode=='original':return np.ones(count,dtype=bool)
    if mode!='low_bpp':raise ValueError('unknown embedding schedule')
    flags=np.zeros(count,dtype=bool)
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,0x43555252]))
    flags[rng.permutation(count)[:count//4]]=True
    return flags


class CurriculumImages(TrainingImages):
    def __init__(self,corpus,mode,stage_epochs,*,cell_supervision='off'):
        super().__init__(corpus)
        if cell_supervision not in ('off','changed_cells_v1'):
            raise ValueError('unknown cell supervision')
        if corpus.settings.paired_batches and mode != 'original':
            raise ValueError('paired training requires the original schedule')
        if cell_supervision != 'off' and (mode != 'original' or corpus.train_tile != 256 or corpus.settings.source_mix is None):
            raise ValueError('cell targets need original schedule, 256 mixed-source training')
        self.cell_supervision=cell_supervision
        from lsb_batch import read_config
        if corpus.config.as_dict()!=read_config(corpus.root/'configs/train-k1to4.yaml').as_dict():
            raise ValueError('curriculum base must match train-k1to4.yaml exactly')
        self.mode=mode;self.stage_epochs=stage_epochs

    def _items_mixed(self,groups,epoch):
        from lsb_torch import TileSpec,OTFSeed,make_sample
        settings=self.corpus.settings
        stage=stage_for_epoch(epoch,self.stage_epochs,self.mode)
        config=replace(self.corpus.config,size={'bpp':[.05,stage['bpp_max']]},
                       k_weights={1:stage['k1_weight'],2:1,3:1,4:1},bpp_mixture=None).normalized()
        # The parent shards whole photos. Reconstruct occurrence positions from
        # the global schedule, so repeated tiles and worker count cannot change
        # the 25/75 assignment. Group identity is unique by photo_id.
        full,_=self.corpus.train_tile_schedule(epoch)
        flags=original_positions(settings.tiles_per_epoch,settings.run_seed,epoch,self.mode)
        starts={};offset=0
        for record,tiles in full:
            starts[record['photo_id']]=offset;offset+=len(tiles)
        if offset!=len(flags):raise ValueError('tile schedule count changed')
        for record,tiles in groups:
            photo=self.corpus.photo(record);ys,xs=self.corpus.training_axes(record)
            for occurrence,(row,col,variant) in enumerate(tiles):
                original=bool(flags[starts[record['photo_id']]+occurrence])
                chosen=self.corpus.config if original else config
                spec=TileSpec(self.corpus.training_grid(record),self.corpus.train_tile,row,col)
                seed=OTFSeed(settings.run_seed,'train',epoch,variant,settings.fixed_variants)
                clean,_=make_sample(photo,record,spec,self.corpus.config,seed,'clean')
                result=make_sample(photo,record,spec,chosen,seed,'stego',
                    **({'return_cell_targets':True} if self.cell_supervision!='off' else {}))
                stego,meta=result[:2]
                pair_id=(self.corpus.offsets[record['photo_id']]+(row*len(xs)+col)*settings.variants_per_tile+variant-1)
                # Metadata is carried through shuffle/packing to the callback;
                # only actually consumed stegos enter the epoch counts.
                tag=(pair_id*2,int(original))
                if settings.paired_batches:
                    clean_tag,stego_tag=tag,(tag[0]+1,tag[1])
                    if self.cell_supervision!='off':
                        clean_tag=clean_tag+(np.zeros((8,8),dtype=np.uint16),np.ones((8,8),dtype=bool))
                        stego_tag=stego_tag+tuple(result[2])
                    yield clean,stego,(clean_tag,stego_tag),meta['embedding_seed_key']
                elif self.cell_supervision!='off':
                    yield clean,0,tag+(np.zeros((8,8),dtype=np.uint16),np.ones((8,8),dtype=bool)),meta['embedding_seed_key']
                    yield stego,1,(tag[0]+1,tag[1],*result[2]),meta['embedding_seed_key']
                else:
                    yield clean,0,tag,meta['embedding_seed_key']
                    yield stego,1,(tag[0]+1,tag[1]),meta['embedding_seed_key']
            del photo

    def _output(self,item):
        if self.corpus.settings.paired_batches:
            from lsb_torch import image_tensor
            clean,stego,tags,entropy=item
            dtype=getattr(torch,self.corpus.settings.tensor_dtype)
            if self.cell_supervision!='off':
                images=[];packed=[]
                for image,tag in zip((clean,stego),tags):
                    image,cells=image_tensor(image,entropy,1,self.corpus.settings.augment,dtype,cell_targets=tag[2:])
                    images.append(image)
                    packed.append(torch.cat((torch.tensor(tag[:2],dtype=torch.long),*(a.flatten() for a in cells))))
                return *images,torch.stack(packed)
            return (image_tensor(clean,entropy,1,self.corpus.settings.augment,dtype),
                    image_tensor(stego,entropy,1,self.corpus.settings.augment,dtype),torch.tensor(tags,dtype=torch.long))
        if self.cell_supervision!='off':
            from lsb_torch import image_tensor
            image,label,tag,entropy=item
            image,cells=image_tensor(image,entropy,label,self.corpus.settings.augment,
                getattr(torch,self.corpus.settings.tensor_dtype),cell_targets=tag[2:])
            return image,label,torch.cat((torch.tensor(tag[:2],dtype=torch.long),*(a.flatten() for a in cells)))
        image,label,tag=super()._output(item)
        return image,label,torch.tensor(tag,dtype=torch.long)


class CurriculumCounts(L.Callback):
    def __init__(self,path,mode,stage_epochs):
        self.path=Path(path);self.mode=mode;self.stage_epochs=stage_epochs
        self.epoch=-1;self.original=0;self.staged=0

    def state_dict(self):
        return dict(epoch=self.epoch,original=self.original,staged=self.staged)

    def load_state_dict(self,state):
        self.epoch=state['epoch'];self.original=state['original'];self.staged=state['staged']
        # Validation can save last.ckpt before on_train_epoch_end writes counts.
        if self.epoch>=0:self._write(self.epoch)

    def on_train_epoch_start(self,trainer,module):
        self.epoch=int(trainer.current_epoch);self.original=0;self.staged=0
        stage=stage_for_epoch(self.epoch,self.stage_epochs,self.mode)
        print('CURRICULUM STAGE '+json.dumps(dict(epoch=int(trainer.current_epoch),**stage)),flush=True)

    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        _,labels,tags=batch
        if getattr(module,'hparams',{}).get('paired_batches',False):
            # One stego tag per pair; batch[1] contains pixels, not labels.
            flags=tags[:,1,1]
            self.original+=int((flags==1).sum())
            self.staged+=int((flags==0).sum())
        else:
            stego=labels==1
            self.original+=int(((tags[:,1]==1)&stego).sum())
            self.staged+=int(((tags[:,1]==0)&stego).sum())

    def on_train_epoch_end(self,trainer,module):
        self._write(int(trainer.current_epoch))

    def _write(self,epoch):
        record=dict(epoch=epoch,**stage_for_epoch(epoch,self.stage_epochs,self.mode),
                    original_stegos=self.original,stage_stegos=self.staged,
                    consumed_stegos=self.original+self.staged)
        # Replace an epoch on replay, rather than silently double count it.
        previous=[json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        previous=[r for r in previous if r['epoch']!=epoch]+[record]
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in sorted(previous,key=lambda r:r['epoch'])))
        print('CURRICULUM COUNTS '+json.dumps(record),flush=True)


def weights_only(module,path,expected_sha):
    """Validate before copying. Never load an optimiser, scheduler or epoch."""
    path=Path(path)
    content=path.read_bytes();actual=hashlib.sha256(content).hexdigest()
    if actual!=expected_sha:raise ValueError('initial checkpoint SHA-256 mismatch')
    import io
    saved=torch.load(io.BytesIO(content),map_location='cpu',weights_only=False)
    h=saved['hyper_parameters']; current=module.hparams
    defaults=dict(model='xunet',width=4,front_end='kv',norm='none',clip_t=3.)
    for key,default in defaults.items():
        if h.get(key,default)!=current.get(key,default):
            raise ValueError('initial checkpoint architecture setting mismatch: '+key)
    state={key.removeprefix('net.'):value for key,value in saved['state_dict'].items()}
    target=module.net.state_dict()
    head_keys={'cell_head.weight','cell_head.bias'} if current.get('cell_supervision','off')=='changed_cells_v1' else set()
    if (set(target)-set(state)!=head_keys or set(state)-set(target) or
            any(state[k].shape!=target[k].shape for k in state)):
        raise ValueError('initial checkpoint state architecture mismatch')
    # Only this explicitly declared new head stays initialized. All other keys
    # have been checked; the final copy remains strict (also for feature off).
    module.net.load_state_dict({**state,**{k:target[k] for k in head_keys}},strict=True)
    return dict(checkpoint=str(path),checkpoint_sha256=actual,weights_only=True,
                optimizer_restored=False,scheduler_restored=False,source_epoch=int(saved.get('epoch',-1)))


def start_identity(settings):
    from .settings import SCHEMA,plain
    # Machine controls may change on resume; experiment/data/seed/LR/start
    # settings may not. This also prevents extending the stage table by accident.
    identity={k:plain(v) for k,v in settings.values.items() if not SCHEMA[k].machine}
    if settings.train_yaml:
        identity['train_yaml_sha256']=hashlib.sha256(Path(settings.train_yaml).read_bytes()).hexdigest()
    if getattr(settings,'embedding_schedule',None) is not None:
        identity['stage_table']=[list(s) for s in STAGES]
    return identity


def check_start(settings,folder):
    """Fail before run records are overwritten. Default training never calls this."""
    path=Path(folder)/'training_start.json';resume=(Path(folder)/'last.ckpt').exists()
    identity=start_identity(settings)
    if path.exists():
        prior=json.loads(path.read_text())
        if prior['settings']!=identity:raise ValueError('start/resume settings or initialization fingerprint changed')
        if not resume:raise ValueError('existing initialized run has no last.ckpt; use a new run')
    elif resume:raise ValueError('resume checkpoint has no initialization identity record')
    return resume


def initialize(settings,module,folder,resume):
    path=Path(folder)/'training_start.json'
    if resume:
        print('WEIGHTS INIT skipped: resume restores this run last.ckpt',flush=True)
        return json.loads(path.read_text())
    checkpoint=Path(settings.init_checkpoint)
    if not checkpoint.is_absolute():checkpoint=Path(settings.root)/checkpoint
    record=weights_only(module,checkpoint,settings.init_checkpoint_sha256)
    record['settings']=start_identity(settings)
    path.write_text(json.dumps(record,indent=2)+'\n')
    print('WEIGHTS INIT '+json.dumps({k:v for k,v in record.items() if k!='settings'}),flush=True)
    return record
