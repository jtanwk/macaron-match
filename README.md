# macaron-match

One of the key steps in baking macarons is to match top and bottom shells to each other. This is usually done by eye. I would like to do this by app.

Shells should be matched by shape and size. 

## Overview

1. Take an overhead picture of all shells.
2. Extract the shell contours with OpenCV
3. Extract the area of each shell's bounding box and perform unidimensional clustering by area
    - Other potential clustering metrics: bbox aspect ratio, contour area
    - Use Jenks natural breaks
4. Within each cluster, extract each contour's pairwise shape difference score using OpenCV
5. Within each cluster, perform maximimum bipartite matching to find pairs of similar size by shape
6. Label and output image with pair assignments