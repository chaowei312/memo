## DiUnet: Difference-based Input U-Net for Video Processing

### Key Idea: Sparse Feature Extraction via Frame Differences

The core innovation of this video processing neural network lies in its input feature representation strategy:

#### Traditional Approach vs. Proposed Method
- **Traditional**: Uses consecutive frame pairs `(f_i, f_{i-1})` as input
  - Requires processing entire previous frame features
  - High computational overhead due to redundant information between frames
  - Dense feature representation even for minimal motion regions

- **Proposed DiUnet Approach**: Uses current frame and forward difference `(f_i, f_{i+1} - f_i)` as input
  - `f_i`: Current frame providing spatial context
  - `f_{i+1} - f_i`: Forward temporal difference capturing motion dynamics
  
#### Sparse Feature Activation Strategy
The pixel difference regions undergo thresholded activation:
```
sparse_features = activate(|f_{i+1} - f_i| > threshold)
```

