"""实验 3 v4 入口：实际夹爪物理闭合、收敛验证、模型输入记录。"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_b90_controlled_v3 import ConvergedEpisode

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--group',choices=['B','C'],required=True)
    ap.add_argument('--snap',type=Path,required=True)
    ap.add_argument('--policy',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--state-only',action='store_true')
    ap.add_argument('--max-ticks-b',type=int,default=260)
    ap.add_argument('--checkpoint-sha',default='34a6e308')
    ap.add_argument('--commit',default='e9cb54c')
    a=ap.parse_args();snap=json.loads(a.snap.read_text())
    ep=ConvergedEpisode(a,snap,'frozen',a.group,'continuous');ep.run()
if __name__=='__main__':main()
