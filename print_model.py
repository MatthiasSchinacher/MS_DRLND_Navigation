# * ---------------- *
#
#   ** Deep Reinforcement Learning Nano Degree **
#   project: Navigation
#   author:  Matthias Schinacher
#
# small helper script, to output a model
# * ---------------- *

# * ---------------- *
#    importing the packages we need
# * ---------------- *
import os.path
import sys
import torch

# * ---------------- *
#   command line arguments:
#    we expect exactly 2, the actual script name and the model-file-name
# * ---------------- *
if len(sys.argv) != 2:
    print('usage:')
    print('   python {} model-file-name'.format(sys.argv[0]))
    quit()

if not os.path.isfile(sys.argv[1]):
    print('usage:')
    print('   python {} model-file-name'.format(sys.argv[0]))
    print('[error] "{}" file not found or not a file!'.format(sys.argv[1]))
    quit()

modelQ = torch.load(sys.argv[1])
print('model from file "{}:"\n'.format(sys.argv[1]))
#print(modelQ)
for x in modelQ.modules():
    #print(type(x))
    print(x)
    if isinstance(x, torch.nn.modules.linear.Linear):
        print('weight:',x.weight)
        print('bias:',x.bias)
