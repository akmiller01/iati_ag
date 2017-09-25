# IATI Agriculture Sectors Machine Learning Test

## Parameters
 1. Classes: 8
 2. Training observations: 14,553
 3. Testing observations: 1,624
 4. Hidden layer units: 1,000
 5. Hidden layers: 1
 6. Learning rate: 0.1
 7. Overall accuracy: 52% (Compared with 12.5% random chance)
 
## Steps
 1. preprocess_aiddata.R
 2. process_data.py
 3. word2vec.lua
 4. classify_vector.lua
 5. stitch_and_average.py
 
## Results table
 
| Rounded confidence | Average accuracy | Count | Cumulative sum | Cumulative percent |
|--------------------|------------------|-------|----------------|--------------------|
| 0.8                | 100.00%          | 7     | 7              | 0.43%              |
| 0.7                | 92.31%           | 26    | 33             | 2.03%              |
| 0.6                | 87.88%           | 66    | 99             | 6.10%              |
| 0.5                | 80.00%           | 160   | 259            | 15.95%             |
| 0.4                | 65.17%           | 178   | 437            | 26.91%             |
| 0.3                | 61.22%           | 263   | 700            | 43.10%             |
| 0.2                | 46.48%           | 327   | 1027           | 63.24%             |
| 0.1                | 36.91%           | 382   | 1409           | 86.76%             |
| 0                  | 29.30%           | 215   | 1624           | 100.00%            |
| Total              | 52.34%           | 1624  | 1624           | 100.00%            |