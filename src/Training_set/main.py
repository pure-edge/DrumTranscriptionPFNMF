import os
from generateFlam import generateFlam
from generateDrag import generateDrag

strikePath = './Strike_MN'
flamFolder = './Flam/'
dragFolder = './Drag/'
generateFlam(strikePath, flamFolder)
generateDrag(strikePath, dragFolder)
