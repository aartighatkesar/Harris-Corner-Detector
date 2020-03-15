# Harris-Corner-Detector
Harris Corner Detector and correspondence

#### Description:
This project detects corners in a given image. Also, finds corresponding corners in two images based on Normalized Cross Correlation (NCC) and Sum of Squared Differences (SSD) scores
```
0. Build Haar filters for calculating derivatives in x and y direction. The size of the filter controls the amount of smoothing.
1. Compute derivatives in x and y direction
2. Build corner response matrix
3. Find locations where the corner response is greater than threshold
4. Apply local non maximum suppression to retain locations where corner response is a local maximum
5. In order to find corresponding corners, use NCC or SSD metrics
```


#### Dependencies

- OpenCV
- NumPy
- SciPy

## Scripts

- [**detect_harris_corners.py**](./detect_harris_corners.py):
```python
python detect_harris_corners.py
```

Script to detect corners in an image. Also implements Non-Maximal Supression. The parameters "sigma" and 

Edit the "images" variable to pass the path to the images whose corners need to be detected. 
Also, feel free to edit the "results_dir" variable to set the results directory path. Will add parser flags in future. It was just easier to test this way. 
- [**correspondence_measures.py**](./correspondence_measures.py): 
```python
python correspondence_measures.py

```
## Results

- [Input set 1 - click click](./input_imgs)
- [Results for Input set 1 - more clicks](./results)
    
##### Inputs

<img src="https://github.com/aartighatkesar/Harris-Corner-Detector/blob/master/input_imgs/pair1/1.jpg" alt="1.jpg" width="400" height="300" />  

##### Corners for various values of Smoothing

_Sigma = 0.707 : Num of corners detected = 368_
<br/>
<img src="https://github.com/aartighatkesar/Harris-Corner-Detector/blob/master/results/res_1/sigma_0.707_no_c_368.jpg" alt="1.jpg" width="400" height="300" />  
<br/>
<br/>
_Sigma = 1.0 : Num of corners detected = 320_
<br/>
<img src="https://github.com/aartighatkesar/Harris-Corner-Detector/blob/master/results/res_1/sigma_1_no_c_320.jpg" alt="1.jpg" width="400" height="300" />  
<br/>
<br/>
_Sigma = 1.414 : Num of corners detected = 282_
<br/>
<img src="https://github.com/aartighatkesar/Harris-Corner-Detector/blob/master/results/res_1/sigma_1.414_no_c_283.jpg" alt="1.jpg" width="400" height="300" />  
<br/>
<br/>
_Sigma = 2 : Num of corners detected = 225_
<br/>
<img src="https://github.com/aartighatkesar/Harris-Corner-Detector/blob/master/results/res_1/sigma_2_no_c_225.jpg" alt="1.jpg" width="400" height="300" />  
<br/>

#### Correspondence

_All corner points correspondence. Sigma = 2.0. Metric : Normalized Cross Correlation_
<br/>
<img src="https://github.com/aartighatkesar/Harris-Corner-Detector/blob/master/results/correspondence_all/1_2_sig_2_m_ncc.jpg" alt="1.jpg" width="1000" height="500" />  
<br/>
_Showing top 100 corner points correspondence. Sigma = 2.0. Metric : Normalized Cross Correlation_
<br/>
<img src="https://github.com/aartighatkesar/Harris-Corner-Detector/blob/master/results/correspondence_top_100/1_2_sig_2_m_ncc.jpg" alt="1.jpg" width="1000" height="500" />  
<br/>
<br/>
_All corner points correspondence. Sigma = 2.0. Metric : Sum of Squared Differences_
<br/>
<img src="https://github.com/aartighatkesar/Harris-Corner-Detector/blob/master/results/correspondence_all/1_2_sig_2_m_ssd.jpg" alt="1.jpg" width="1000" height="500" />  
<br/>
_Showing top 100 corner points correspondence. Sigma = 2.0. Metric : Sum of Squared Differences_
<br/>
<img src="https://github.com/aartighatkesar/Harris-Corner-Detector/blob/master/results/correspondence_top_100/1_2_sig_2_m_ssd.jpg" alt="1.jpg" width="1000" height="500" />  
<br/>








