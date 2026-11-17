#!/usr/bin/env python3
"""Build and run one experiment from this folder; standard library only."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODES = {'mono':'mono_visual.json','mono-fused':'mono_fused.json','stereo':'stereo_sample_pose.json','stereo-disparity':'stereo_sample_disparity.json'}
def validate(cfg):
    required=[cfg['yolo']['model_path'],cfg['strategies']['board_template']['template_path']]
    dual=cfg['strategies']['dual_roi']
    required += [dual['class0_points_path'],dual['class1_points_path']]
    image=cfg['input_system']['image']
    if image['type']=='video':required += [image['video_path'],image['timestamp_path']]
    elif image['type']=='directory':
        directory=ROOT/image['directory_path']
        left=sorted(p for p in directory.glob(image['left_pattern']+'*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'})
        if not left:raise ValueError('No stereo pairs: '+str(directory))
        missing=[p.name for p in left if not p.with_name(image['right_pattern']+p.name[len(image['left_pattern']):]).is_file()]
        right=list(p for p in directory.glob(image['right_pattern']+'*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'})
        if missing or len(left)!=len(right):raise ValueError('Unpaired stereo images: '+str(missing[:10]))
        print('Stereo pairs:',len(left),'(synthetic sample; see README)')
    for sensor in ['imu','altimeter']:
        node=cfg['input_system'].get(sensor,{})
        if node.get('enabled'):required.append(node['path'])
    missing=[p for p in required if not (ROOT/p).is_file()]
    if missing:raise ValueError('Missing required files: '+str(missing))
    print('Required files found:',len(required))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=MODES)
    ap.add_argument('--backend',choices=['onnx','rknn'],default='onnx')
    ap.add_argument('--max-frames',type=int)
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if args.max_frames is not None and args.max_frames<0:ap.error('--max-frames must be >= 0')
    if args.jobs<1:ap.error('--jobs must be >= 1')
    os.chdir(ROOT)
    cfg=json.loads((ROOT/'config'/MODES[args.mode]).read_text(encoding='utf-8'))
    if args.backend=='rknn':
        cfg['yolo']['model_path']={'yolo_onnx/best.onnx':'yolo_onnx/best_rk3588_fp16.rknn','yolo_onnx/yolov8n_p2.onnx':'yolo_onnx/yolov8n_p2.rknn'}[cfg['yolo']['model_path']]
    if args.max_frames is not None:cfg['input_system']['max_frames']=args.max_frames
    validate(cfg)
    if args.check_only:return
    if not sys.platform.startswith('linux'):raise RuntimeError('Run this entry in Linux/WSL/Docker/RK3588 Linux; this is not a Windows-native executable.')
    build=ROOT/('build-'+args.backend)
    command=['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DGPNP_WITH_RKNN='+('ON' if args.backend=='rknn' else 'OFF')]
    subprocess.run(command,check=True)
    subprocess.run(['cmake','--build',str(build),'--target','GPNP','-j',str(args.jobs)],check=True)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    result=ROOT/'results'/(args.mode+'_'+stamp)
    result.mkdir(parents=True,exist_ok=False)
    cfg['output']['directory']=result.relative_to(ROOT).as_posix()
    config=result/'config_used.json'
    config.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Results:',result,flush=True)
    env=os.environ.copy();env.setdefault('QT_QPA_PLATFORM','offscreen')
    subprocess.run([str(build/'GPNP'),str(config)],env=env,check=True)

if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as e:
        print('Run failed:',e,file=sys.stderr);sys.exit(1)
