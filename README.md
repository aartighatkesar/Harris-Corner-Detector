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
    
###### Inputs

<img src="https://github.com/aartighatkesar/Harris-Corner-Detector/blob/work-in-progress/input_imgs/pair1/1.jpg" alt="1.jpg" width="400" height="300" />  

<img src="https://github.com/aartighatkesar/Harris-Corner-Detector/blob/work-in-progress/input_imgs/pair2/1.jpg" alt="1.jpg" width="400" height="300" />  

###### Intermediate results

* All SIFT feature correspondence for img 2 and 3
<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/results/2_3_sift_corr.jpg" width="468" height="416" />


* Inliers from RANSAC for img 2 and 3
<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/results/2_3_inliers.jpg" width="468" height="416" />

* Outliers from RANSAC for img 2 and 3
<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/results/2_3outliers.jpg" width="468" height="416" />
       

Go [here](./input/p2/results) for more intermediate results for all image pairs

#### Final Image Mosaic

<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/results/panorama_4.jpg" width="602" height="482" />

        

- [Input set 2 - click click](./input/p4)
    - [Results for Input set 2 - more clicks](./input/p4/results)
    
###### Inputs
- 1.jpg
<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p4/1.jpg" alt="1.jpg" width="416" height="312" />

- 2.jpg
<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p4/2.jpg" alt="2.jpg" width="416" height="312" /> 

- 3.jpg
<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p4/3.jpg" alt="3.jpg" width="416" height="312" /> 

- 4.jpg
<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p4/4.jpg" alt="4.jpg" width="416" height="312" />

- 5.jpg
<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p4/5.jpg" alt="5.jpg" width="416" height="312" />


###### Intermediate results

* All SIFT feature correspondence for img 2 and 3
<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p4/results/2_3_sift_corr.jpg" width="832" height="312" />


* Inliers from RANSAC for img 2 and 3
<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p4/results/2_3_inliers.jpg" width="832" height="312" />

* Outliers from RANSAC for img 2 and 3
<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p4/results/2_3outliers.jpg" width="832" height="312" />
       

Go [here](./input/p4/results) for more intermediate results for all image pairs

#### Final Image Mosaic

<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p4/results/panorama_4.jpg" width="720" height="390" />










