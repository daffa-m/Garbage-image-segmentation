# Garbage-image-segmentation

This is a deep learning learning application (in this case, Pyramid Scene Parsing Network or PSPNet) for segmenting garbage image with web interface
It can obtain over 87 percent f1-score across various scenarios:
1. Using global-thresholding-image-output as input to feed the PSPNet model
2. Using otsu-thresholding-image-output as input to feed the PSPNet model
3. Using adaptive-mean-thresholding-image-output as input to feed the PSPNet model
4. Using combination of three-of-those-thresholdings-image-output to feed the PSPNet model
5. Using original image as input to feed the PSPNet model
6. Using ResNet50 as part of the PSPNet model
7. Using ResNet18 as part of the PSPNet model

To use this program :
1. Run command prompt on terminal on your computer
2. Run Code.py by using command "python Code.py"
3. Open your web browser
4. Go to "http://127.0.0.1:5000/" on your web browser
5. Choose a garbage image image to be segmented
6. Click on "Submit & predict" button
7. The segmentation results will be displayed immidiately

Screenshoots :
![image](https://user-images.githubusercontent.com/77146831/110207447-1d8f2480-7eb6-11eb-8927-d672ded3b295.png)
![image](https://user-images.githubusercontent.com/77146831/110207415-e91b6880-7eb5-11eb-900c-9d87b579ac92.png)

