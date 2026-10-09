import cv2,glob,numpy as np
from multiprocessing import Pool
def load(f): return cv2.resize(cv2.imread(f),(480,270),interpolation=cv2.INTER_AREA)
if __name__=='__main__':
    for g in ('game1','game2'):
        fs=sorted(glob.glob(f"dataset/{g}/*/*.png"))
        with Pool(6) as p: arr=np.stack(p.map(load,fs,chunksize=8))
        np.save(f"dataset/cache/{g}_480.npy",arr)
        with open(f"dataset/cache/{g}_times.txt","w") as o: o.write("\n".join(f.split('/')[-1][:-4] for f in fs))
        print(g,arr.shape,flush=True)
