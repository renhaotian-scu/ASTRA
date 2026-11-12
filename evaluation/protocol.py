

from collections import defaultdict
import math
import numpy as np
from scipy import ndimage
from . import geometry as original

MASK_GRID = (.1, .2, .3, .5, .7, .9)
PROTOCOL = {'version': 'WP2_mask_to_box_v2', 'mask_grid': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9], 'binary_test': 'probability >= mask_threshold', 'component': '8-connected; no morphology, component-area filter, manual splitting or merging', 'score': 'float64 arithmetic mean of original restored float32 probabilities at actual labeled CC foreground pixels; not bbox pixels', 'box': 'tight raster support xywh, x/y plus frozen dataset offset; extent=max-min+1', 'operating_point': 'all CC boxes, score ordered one-to-one IoU>=.5; no confidence threshold or top-K cut', 'selection': 'maximum VAL Box-F1, tie lower VAL Fa per all images, remaining tie lower mask; tolerance1e-12', 'public_scope': 'P, Pd=Recall, F1 in percent; Fa=FP/all images; no AP or model inference', 'ASTRA_working_point': 'retain its original VAL-selected native query confidence; do not count all queries as detections'}

def component_boxes(probability,image_id,original_size,mask_threshold,category_id,xy_offset=(0,0)):
 if not 0<mask_threshold<=1:raise ValueError('Invalid threshold')
 prob=original.restore_probability(probability,original_size)
 labels,n=ndimage.label(prob>=mask_threshold,structure=np.ones((3,3),dtype=np.uint8))
 if n==0:return []
 indices=np.arange(1,n+1);means=ndimage.mean(prob,labels,index=indices)
 maxima=ndimage.maximum(prob,labels,index=indices);areas=ndimage.sum(np.ones_like(prob,dtype=np.uint8),labels,index=indices)
 detections=[]
 for k,slices in enumerate(ndimage.find_objects(labels)):
  if slices is None:continue
  ys,xs=slices
  detections.append({'image_id':image_id,'category_id':category_id,
   'bbox':[float(xs.start+xy_offset[0]),float(ys.start+xy_offset[1]),float(xs.stop-xs.start),float(ys.stop-ys.start)],
   'score':float(means[k]),'component_id':k+1,'component_area_pixels':int(areas[k]),'component_max_probability':float(maxima[k])})
 return sorted(detections,key=lambda d:-d['score'])

def convert(manifest,split,maps,mask):
 samples=original.samples_for_split(manifest,split)
 if set(maps)!={s['sample_id'] for s in samples}:raise ValueError('Exact split maps required')
 offset=manifest.get('mask_box_xy_offset',[0,0]);category=original.category_id({'categories':manifest['categories']})
 return [d for s in samples for d in component_boxes(maps[s['sample_id']],s['image_id'],(s['height'],s['width']),mask,category,offset)]

def checked_all(gt,dets):
 images,annotations=original._validate_gt(gt);category=original.category_id(gt);grouped=defaultdict(list)
 for d in dets:
  if d['image_id'] not in images or d['category_id']!=category:raise ValueError('Unknown image/category')
  if len(d['bbox'])!=4 or min(d['bbox'][2:])<=0 or not all(math.isfinite(float(v)) for v in [*d['bbox'],d['score']]) or not 0<=d['score']<=1:raise ValueError('Invalid box/score')
  grouped[d['image_id']].append(d)
 return images,annotations,{iid:sorted(grouped.get(iid,[]),key=lambda d:-d['score']) for iid in images}

def match_all(gt,dets):
 images,annotations,by_image=checked_all(gt,dets);tp=fp=fn=0;records=[]
 for iid in images:
  truth=annotations[iid];used=set();preds=[]
  for index,d in enumerate(by_image[iid]):
   best=.5;match=None
   for j,a in enumerate(truth):
    overlap=original._box_iou(d['bbox'],a['bbox'])
    if j not in used and overlap>=best:best,match=overlap,j
   row={'prediction_index':index,**d,'matched_gt_index':match,'matched_gt_id':truth[match]['id'] if match is not None else None,
        'matched_IoU':best if match is not None else None,'status':'TP' if match is not None else 'FP'}
   preds.append(row)
   if match is None:fp+=1
   else:tp+=1;used.add(match)
  missing=[{'gt_index':j,'gt_id':a['id'],'bbox':a['bbox']} for j,a in enumerate(truth) if j not in used];fn+=len(missing)
  records.append({'image_id':iid,'file_name':images[iid].get('file_name'),'n_GT':len(truth),'n_predictions':len(preds),
    'TP':len(used),'FP':len(preds)-len(used),'FN':len(missing),'ground_truth':truth,'predictions':preds,'false_negatives':missing})
 assert tp+fp==len(dets),'All CC boxes must contribute to the PRF operating point'
 p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
 return {'TP':tp,'FP':fp,'FN':fn,'P':100*p,'R':100*r,'F1':200*p*r/(p+r) if p+r else 0.,
   'Fa_per_image':fp/len(images) if images else 0.,'images':len(images),'targets':tp+fn},records

def better(candidate,current):
 if current is None:return True
 cm,om=candidate['metrics'],current['metrics']
 if cm['F1']>om['F1']+1e-12:return True
 if abs(cm['F1']-om['F1'])<=1e-12:
  if cm['Fa_per_image']<om['Fa_per_image']-1e-12:return True
  if abs(cm['Fa_per_image']-om['Fa_per_image'])<=1e-12:return candidate['mask_threshold']<current['mask_threshold']
 return False
