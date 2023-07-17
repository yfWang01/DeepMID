import os

import airPLS
import nmrglue as ng
import numpy as np
from tqdm import tqdm


def read_bruker_h_base(nmr_path, bRaw=False, bMinMaxScale=False):
    nmr_path = os.path.normpath(nmr_path)
    if bRaw:
        dic, fid = ng.fileio.bruker.read(f'{nmr_path}/1')
        zero_fill_size = dic['acqus']['TD']
        fid = ng.bruker.remove_digital_filter(dic, fid)
        fid = ng.proc_base.zf_size(fid, zero_fill_size)
        fid = ng.proc_base.fft(fid)

    else:
        dic, fid = ng.fileio.bruker.read_pdata(f'{nmr_path}/1/pdata/1')
        zero_fill_size = dic['acqus']['TD']
        offset = (float(dic['acqus']['SW']) / 2) - (float(dic['acqus']['O1']) / float(dic['acqus']['BF1']))
        start = float(dic['acqus']['SW']) - offset
        end = -offset
        step = float(dic['acqus']['SW']) / zero_fill_size
        ppms = np.arange(start, end, -step)[:zero_fill_size]

        # baseline
        baseline = airPLS(fid, lambda_=100, porder=1, itermax=15)
        fid = fid - baseline

        # set solvent peaks to zero
        q = np.min(np.where(np.round(ppms, 3) == 5.365))
        w = np.min(np.where(np.round(ppms, 3) == 5.100))
        t = np.min(np.where(np.round(ppms, 3) == 4.160))
        y = np.min(np.where(np.round(ppms, 3) == 3.800))
        u = np.min(np.where(np.round(ppms, 3) == 1.060))
        i = np.min(np.where(np.round(ppms, 3) == 1.030))
        o = np.min(np.where(np.round(ppms, 3) == 1.180))
        p = np.min(np.where(np.round(ppms, 3) == 1.105))
        a = np.min(np.where(np.round(ppms, 3) == 3.530))
        s = np.min(np.where(np.round(ppms, 3) == 3.190))

        fid[q:w] = 0
        fid[t:y] = 0
        fid[u:i] = 0
        fid[o:p] = 0
        fid[a:s] = 0

    # Normalization
    if bMinMaxScale:
        fid = fid / np.max(fid)
    v = np.max(np.where(np.round(ppms, 3) == 10.700))
    b = v + 32724  # b = np.max(np.where(np.round(ppms,3)==0.300))

    return {'name': nmr_path.split(os.sep)[-1], 'ppm': ppms[v:b], 'fid': fid[v:b], 'bRaw': bRaw}


def read_bruker_hs_base(data_folder, bRaw, bMinMaxScale, bDict):
    if bDict:
        spectra = {}
    else:
        spectra = []
    for name in tqdm(os.listdir(data_folder), desc="Read Bruker H-NMR files"):
        nmr_path = os.path.normpath(os.path.join(data_folder, name))
        s = read_bruker_h_base(nmr_path, bRaw, bMinMaxScale)
        if bDict:
            spectra[s['name']] = s
        else:
            spectra.append(s)
    return spectra
