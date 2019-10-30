# Harris-Corner-Detector
Harris Corner Detector and correspondence

#### Description:
This project detects corners in a given image. Also, finds corresponding corners in two images based on Normalized Cross Correlation (NCC) and Sum of Squared Differences (SSD) scores
```
0. Click pictures such that adjacent pictures have atleast 40% overlap. Order the pictures from left to right
1. Generate SIFT features among images
2. Establish correspondence between features
3. Apply RANSAC (Random Sampling and Consensus) to get rid of outliers
4. Generate initial estimate of Homography using inlier correspondence points obtained in step 4
5. Refine the Homography estimate using Levenberg- Marquardt optimization
6. Repeat the above steps for each of the adjacent image pairs 
7. After obtaining Homographies for each of the picture pairs, get the homographies with respect to the central image
8. Project all images (using inverse warping) on to a blank canvas. Use bilinear interpolation
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
Edit the "img_path" variable to 


###### Supporting scripts
- [**match_features.py**](./match_features.py): Script to generate SIFT features and establish correspondence between features of two images
- [**ransac.py**](./ransac.py): RANSAC algorithm to obtain inliers and discard outliers among correspondences
- [**optimize_fcn.py**](./optimize_fcn.py): Levenberg-Marquardt algorithm for optimization. Generic script and can be used for any Non-Linear Least Squares Estimation
- [**estimate_homography.py**](./estimate_homography.py): Helper functions which help in bilinear interpolation and projecting images to a canvas using Homography matrix

## Results

- [Input set 1 - click click](./input/p2)
    - [Results for Input set 1 - more clicks](./input/p2/results)
    
###### Inputs

|<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/1.jpg" alt="1.jpg" width="234" height="416" /> 1.jpg  |<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/2.jpg" alt="2.jpg" width="234" height="416" />  2.jpg 
|<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/3.jpg" alt="3.jpg" width="234" height="416" /> 3.jpg  |<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/4.jpg" alt="4.jpg" width="234" height="416" />  4.jpg
|<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/5.jpg" alt="5.jpg" width="234" height="416" /> 5.jpg  |


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










