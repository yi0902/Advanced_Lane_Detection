<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## Advanced Lane Detection Project
###### Udacity Self Driving Car Engineer Nanodegree - Project 4 - Yi CHEN - 02/02/2017
--

The goal of this project is to develop a pipeline to identify following things on a video taken from a forward-facing center camera mounted on the front of a car:

- The lane boundaries positions
- The vehicle position relative to the center of the lane
- The radius of lane curvature

The pipeline of processing frame image of the video is:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use gradients and color transforms to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image and output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1_undistorted.png
[image2]: ./output_images/test1_undistorted.png
[image3]: ./output_images/test5_masked.png
[image4]: ./output_images/pers_transform_straight.png
[image5]: ./output_images/pers_transform_test.png
[image6]: ./output_images/test5_lane_detection.png
[image7]: ./output_images/test5_warp_back.png
[video1]: ./project.mp4


### 1. Camera Calibration

#### In this step, I computed the camera matrix and distortion coefficients.

The code for this step is contained in the `get_cablibration_points_from_chessboard` function and `calibrate_undistort` function of the jupyter notebook **advanced_lane_detection.ipynb** located under root foler.

I started by preparing `objpoints`, which are the (x, y, z) coordinates of the chessboard corners in the world, and `imgpoints`, which are the pixel position of each of the corners in the image plane. This was done thanks to the `get_cablibration_points_from_chessboard` function.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the chessboard images using the `cv2.undistort()` function to check if the calibration was well done: 
![alt text][image1]

### 2. Distortion Correction

#### Here I applied the distortion correction to test images.
I applied the function `calibrate_undistort` (which takes an image, the `objpoints` and `imgpoints` as input and returned an undistorted image) to all test images and an example result is shown below:
![alt text][image2]
If we compare these two images, we could observe obvious differences between the original and undistorted image, especially around the edges, indicating that distortion has been removed from the original image.

### 3. Threshold Binary Image

#### In this step, I used gradients and color transforms to create a thresholded binary image.

For preparing, I computed the gradients on x and y directions by using `abs_sobel_thresh` function. I also computed the magnitude of gradients through `mag_thresh` function, as well as the gradient direction by `dir_thresh` function.

Then I looked into the HLS color space, and I found the S channel with threshold of `(150, 255)` was pretty stable for detecting lines in various conditions, while still brought in some noise when there were shadows. To remove the impact of detected shadows, I found the H channel of threshold `(0, 70)` could do a good job. This part of code is in function `hls_select`.

After multiple tries, I used following combination of gradient and color thresholds to generate the binary image (thresholding steps are in the function `create_threshold_image`):

- set pixel binary = 1 if meet threshold `(20, 100)` on both x gradient and y gradient
- set pixel binary = 1 if meet threshold `(30, 100)` on gradient magnitude and threshold `(0.7, 1.2)` on gradient direction
- set pixel binary = 1 if meet threshold `(150, 255)` on S channel and threshold `(0, 70)` on H channel in HLS color space

Here's an example of my output for this step. 
![alt text][image3]

### 4. Perspective Transform

#### Here I performed perspective transform to get birds-eye view of the lane.
 
The code for my perspective transform includes a function called `warper()`, which appears in the 13rd code cell of the jupyter notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I used the function `set_perspective_transform_points` to set the source and destination points with possibility to slightly adapt those points with `offset` attribute.

```
src = np.float32(
            [[((img_size[0] / 6) - 10), img_size[1]],
            [(img_size[0] / 2) - 60 - offset, img_size[1] / 2 + 100],
            [(img_size[0] / 2 + 60 + offset), img_size[1] / 2 + 100],
            [(img_size[0] * 5 / 6) + 60 - offset, img_size[1]]])
dst = np.float32(
            [[(img_size[0] / 4), img_size[1]],
            [(img_size[0] / 4), 0],
            [(img_size[0] * 3 / 4), 0],
            [(img_size[0] * 3 / 4), img_size[1]]])

```
I used an offset of 4 which resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 203, 720      | 320, 720      | 
| 576, 460      | 320, 0        |
| 704, 460      | 960, 0        |
| 1122, 720     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto straight lines images and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

I then applied this perspective transform to my previously rectified binary images. Two examples are shown below:

![alt text][image5]

### 5. Detect Lane Pixels & Fit Polynomial

#### In this step, I identified lane-line pixels and fit their positions with a 2nd order polynomial. 

The code of this part is in `fit_and_plot_lane` function.

In order to determine location of lane lines, I first identified peaks in the histogram of the image. Then I identified all non zero pixels around histogram peaks using the numpy function `numpy.nonzero()` and isolated those of lane lines by implementing the **sliding windows** technique. I finally fitted a polynomial to each lane using the numpy function `numpy.polyfit()`.

An example of lane lines detection can be seen below:

![alt text][image6]

### 6. Lane Curvature & Vehicle Position

#### In this step, I calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function `compute_lane_curvature_and_car_position`.

To compute the lane curvature, I used the formula R = ![equation](http://www.sciweavers.org/upload/Tex2Img_1486071436/render.png). 

As I wanted to measure the radius of curvature closest to the vehicle, I evaluated the above formula at the y value corresponding to the bottom of the image.

To calculate the position of the vehicle with respect to center:

- I calculated the average of the x intercepts from each of the two polynomials `cr_center = (leftx_yval + rightx_yval)/2`
- Then I calculated the distance from center by substracting the halfway point along the horizontal axis `diff = (cr_center - (img_size[0]/2))`
- Finally, the distance from center was converted from pixels to meters by multiplying **3.7/700**.

If the distance value to center is positive, it means the vehicle is at right of the center while negative value meaning the vehicle is at left of the center.           

### 7. Warp Back & Display

#### Here I plotted back the detected lane section to original image and displayed the lane curvature as well as the vehicle position relative to center.

I implemented this step in the function `warp_back`. As I detected and plotted lanes in warped view, I used inverse perspective trasform to unwarp the image from birds eye back to its original perspective, and then printted the distance from center and radius of curvature on to the final annotated image.

Here is an example of my result on a test image:

![alt text][image7]

---

###Pipeline (video)

The pipeline seemed to work well on static images, now it's time to expand the pipeline to treat video stream frame-by-frame.

To accomplish this, I created a `Lane` class to store features of the lane including information of dection on the last frame. 

If the lane was detected in the last frame, `fit_and_plot_lane` function would only checks for lane pixels in close proximity to the polynomial calculated in the last frame. This way, the pipeline didn't need to scan the entire image, and the pixels detected have a high confidence of belonging to the lane line. If at any time, the pipeline failed to detect lane pixels based on the the last frame, it would go back into blind search mode and scan the entire binary image for nonzero pixels to represent the lanes.

I also chose to weighted average the coefficients of the current fitted polynomials with the last fitted polynomals in order to make the output smoother.

Here's a [link to my video result][video1]

---

###Discussion 

I tried the above pipeline with project video and challenge video. The pipeline worked well on project video, but failed on challenge video as some fake lines presented on the lane. I found the current pipeline focused too much on color saturation, but not enough on color hue and brightness, thus it took the shadow of the left boundary wall as lane line during several frames. Further improvements to make the pipeline more robust could be trying more color models and channels to create a better binary thresholding process.

Also I could imagine, the current pipeline may fail in the night or when the lighting is very limited, or left/right lane line is not visible due to steep turn, or when the vehicule is shiftting lane, exiting the highway etc. In addition to test and improve the binary thresholding process in varied environements, another way to make the pipeline more robust is to smooth the lane detection over multiple frames, such that when we loose the lane on a frame, we could leverage on previous detections to approximate the current one.