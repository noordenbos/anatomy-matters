from pathlib import Path
import sys, os
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
root=Path(__file__).resolve().parents[2]
os.environ['MPLBACKEND']='Agg';os.environ['MPLCONFIGDIR']=str(root/'.matplotlib');os.environ['IPYTHONDIR']=str(root/'.ipython')
p=root/'notebooks'/sys.argv[1]
nb=nbformat.read(p,as_version=4)
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv[0]=sys.executable
try:
    NotebookClient(nb,timeout=14400,kernel_name='python3',km=km,resources={'metadata':{'path':str(root)}}).execute()
finally:
    nbformat.write(nb,p)
