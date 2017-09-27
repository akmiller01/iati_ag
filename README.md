# IATI Agriculture Sectors Machine Learning Test

## Word2Vec vectorization

### Parameters
 1. Classes: 8
 2. Training observations: 14,553
 3. Testing observations: 1,624
 4. Hidden layer units: 1,000
 5. Hidden layers: 1
 6. Learning rate: 0.1
 7. Overall accuracy: 52% (Compared with 12.5% random chance)
 8. Run time on CPU: ~2 minutes
 
### Steps
 1. preprocess_aiddata.R
 2. process_data.py
 3. word2vec.lua
 4. classify_vector.lua
 5. stitch_and_average.py
 
### Results table
 
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

## TFIDF vectorization

### Parameters
 1. Classes: 8
 2. Training observations: 14,553
 3. Testing observations: 1,624
 4. Hidden layer units: 1,000
 5. Hidden layers: 1
 6. Learning rate: 0.1
 7. Overall accuracy: 45% (Compared with 12.5% random chance)
 8. Run time on CPU: ~11 hours
 
### Steps
 1. preprocess_aiddata.R
 2. process_data.py
 3. tfidftrain.lua
 4. classify_tfidf.lua
 5. stitch_and_average.py
 
### Results table
 
| Rounded confidence | Average accuracy | Count | Cumulative sum | Cumulative percent |
|--------------------|------------------|-------|----------------|--------------------|
| 0.8                | 100.00%          | 1     | 1              | 0.06%              |
| 0.7                | 81.82%           | 11    | 12             | 0.74%              |
| 0.6                | 70.45%           | 44    | 56             | 3.45%              |
| 0.5                | 64.06%           | 192   | 248            | 15.27%             |
| 0.4                | 57.87%           | 254   | 502            | 30.91%             |
| 0.3                | 52.82%           | 301   | 803            | 49.45%             |
| 0.2                | 38.93%           | 298   | 1101           | 67.80%             |
| 0.1                | 30.14%           | 345   | 1446           | 89.04%             |
| 0                  | 25.28%           | 178   | 1624           | 100.00%            |
| Total              | 45.26%           | 1624  | 1624           | 100.00%            |