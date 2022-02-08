import numpy as np
import nrrd
import matplotlib.pyplot as plt
import h5py

# The laplacian file tells you the distance from the cortical surface and white matter surface for every point in the isocortex mask
# we don't need this to generate the volume, but we need it to generate a volume that is in percentage steps (see below)
#path_file = 'ccfpaths/laplacian_10.nrrd'
#paths, paths_meta = nrrd.read(path_file)

# The paths matrix is an N x 200 matrix which gives the linear index from every gray matter voxel to every white matter voxel, it takes a varying number of steps
# to get to the white matter, so once you arrive the remaining 200-x values are just zero
# The lookup matrix is AP x ML*2 and maps the position in ap/ml space to the index in the paths matrix
f1 = h5py.File('ccfpaths/dorsal_flatmap_paths_10.h5','r+')    
dorsal_paths = f1['paths'][:]
dorsal_lookup = f1['view lookup'][:]
f1.close()

# To compute a 3D volume which is APxML*2x200 we just copy over the paths data into this volume
dorsal_view_3D = np.zeros((1360,2720,200))
for api in np.arange(0,dorsal_view_3D.shape[0]):
    for mli in np.arange(0,dorsal_view_3D.shape[1]):
        for depth in np.arange(0,dorsal_view_3D.shape[2]):
            dorsal_view_3D[api,mli,depth] = dorsal_paths[dorsal_lookup[api,mli],depth]

nrrd.write('dorsal_flatmap_3D.nrrd',dorsal_view_3D)

# Note that we could also make a volume that is APxML*2 x depth percentage by using the laplacian volume to tell what percentage of the way from the
# gray matter to white matter we are, so e.g. we could go in 10% steps and end up with a volume that is AP x ML*2 x 11 for steps 0:0.1:1
# [TODO]

# Also save just the top slice for immediate use
top_slice = dorsal_view_3D[:,:,0]
np.save('dorsal_flatmap.npy',top_slice)

# And for clarity, load the ccf data and the dorsal flatmap and plot to show how to convert the coordinates into CCF space
ccf10, meta = nrrd.read('annotation_10.nrrd')
dorsal_view_3D, dorsal_meta = nrrd.read('dorsal_flatmap_3D.nrrd')
ccf10_flat = ccf10.flatten()
plt.imshow(np.take(ccf10_flat,np.int64(dorsal_view_3D[:,:,1])))
plt.clim(0,1000)