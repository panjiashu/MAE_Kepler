import numpy as np
import pandas as pd

def getclosest(num,collection):
    '''Given a number and a list, get closest number in the list to number given.'''
    return min(collection,key=lambda x:abs(x-num))

def width_radius(radius):
  width = 0.92+23.03*np.exp(-0.27*radius)
  return width
ref_radius = np.linspace(0,40,1000)
ref_width = np.round(1/(width_radius(ref_radius)/1e6)/60/29.4)

def sigclip(x,y,subs,sig):
    keep = np.zeros_like(x)
    start=0
    end=subs
    nsubs=int((len(x)/subs)+1) if len(x)%subs!=0 else int(len(x)/subs)
    for i in range(0,nsubs):        
        me=np.mean(y[start:end])
        sd=np.std(y[start:end])
        good=np.where((y[start:end] > me-sig*sd) & (y[start:end] < me+sig*sd))[0]
        keep[start:end][good]=1
        start=start+subs
        end=end+subs
    return keep
    
def excel_read(start, excel_root):
  df = pd.read_excel(excel_root)
  kids = []
  loggs = []
  radiis = []
  es = []
  for i in range(start,len(df)):
    choose = df.loc[i][0].split()
    kid = eval(choose[0])
    radius = eval(choose[3])
    logg = eval(choose[6])
    e1 = eval(choose[7])
    e2 = eval(choose[8])
    e = (abs(e1)+abs(e2))
    kids.append(kid)
    loggs.append(logg)
    radiis.append(radius)
    es.append(e)
  return kids, loggs, radiis, es
  
def preprocess(data):
  flux0 = data['PDCSAP_FLUX']
  time0 = data['TIME']
  qual = data['SAP_QUALITY']
  qual[np.isnan(flux0)] = 1
  good=(qual == 0)
  if len(good) == 0:
    return None
  time=time0[good]
  flux=flux0[good]
  res=sigclip(time,flux,50,3)
  good=(res == 1)
  time=time[good]
  flux=flux[good]
  
  time_interp = np.arange(time[0],time[-1],30./(60.*24.))
  flux_interp = np.interp(time_interp, time, flux)
  flux_interp /= np.median(flux_interp)
  return flux_interp