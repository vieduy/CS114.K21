##### Môn: CS114.K21
# **Machine Learning Capstone: Phát hiện biển báo giao thông phổ biến trong Làng Đại học**
###
### Tổng quan
Mục tiêu của đồ án này là xây dựng một mô hình có thể phát hiện được các loại biển báo phổ biến trong Làng Đại học.
Ở đây ta sẽ phân loại 6 loại biển báo phổ biến nhất trong Làng Đại học.

### Mô tả  bài toán:
1. Input: Một bức ảnh có chứa biển báo bất kỳ.
2. Output: Vị trí biển báo trong bức ảnh và tên biển báo đó.

### Để xây dựng được mô hình như yêu cầu bài toán đó, ta cần phải xây dựng 2 mô hình:
1. Phát hiện biển báo(*).
2. Phân loại biển báo(* *).

###
---
###
## Mô hình phát hiện biển báo:
1. Quét bức ảnh input bằng một cửa sổ trượt từ trái sang phải và từ trên xuống dưới.
2. Trích xuất đặc trưng ở mỗi vùng scan qua trên hình.
3.	Sử dụng model phân loại biển báo(* *) để dựng đoán xem vùng đó có chứa biển báo hay không.
4.	Tổng hợp lại các vùng có chứa biển báo thỏa mãn để có một vùng duy nhất (Final bounding boxes) 

### Xây dựng Scanner
Để có thể tìm được vật thể chúng ta cần tìm, ta cần phải quét toàn bộ trên bức hình. Vì kích thước của vật thể có thể nằm bất cứ đâu trên hình và có kích thước ngẫu nhiên. Cho nên chúng ta cần xây dụng “scanner” theo 2 tiêu chí sau:
- Kích thước ảnh quét: Ta cần phải quét trên bức ảnh với nhiều kích thước khác nhau để có thể tìm ra được vật thể. Gọi tắt là kỹ thuật “Image Pyramid”.
- Phạm vi quét: Ta cần phải quét phạm vi toàn bức ảnh. Cho nên cần xây dựng một cửa sổ trượt để quét lần lượt toàn bộ bức ảnh theo chiều từ trên xuống và trái sang phải.(Sliding window)

### Trích xuất đặc trưng ở mỗi vùng scan trên hình
-	Ta sẽ sử dụng HOG((histogram of oriented gradients) để trích xuất đặc trưng trên những vùng mà cửa sổ trượt qua.
-	HOG là một feature descriptor được sử dụng trong computer vision và xử lý hình ảnh, dùng để detec một đối tượng. Các khái niệm về HOG được nêu ra từ năm 1986 tuy nhiên cho đến năm 2005 HOG mới được sử dụng rộng rãi sau khi Navneet Dalal và Bill Triggs công bố những bổ sung về HOG. HOG tương tự như các biểu đồ edge orientation, scale-invariant feature transform descriptors (như sift, surf ,..), shape contexts nhưnghog được tính toán trên một lưới dày đặc các cell và chuẩn hóa sự tương phản giữa các block để nâng cao độ chính xác. HOG được sử dụng chủ yếu để mô tả hình dạng và sự xuất hiện của một object trong ảnh.

### Dự đoán đối tượng trong cửa sổ trượt
-	Ta sẽ sử dụng một model đã được train về các loại biển báo phổ biến trong làng đại học để dự đoán xem có biển báo trong cửa sổ hay không. 
-  Sau khi dự đoán, nếu có đối tượng biển báo trong hình thì ta sẽ tiến hành trả về tọa độ vị trí của đối tượng.

### Tổng hợp lại các khung viền
-  Trong lúc trượt cửa sổ, tùy thuộc vào bước nhảy, sẽ có nhiều cửa sổ thỏa mãn điều kiện có chứa biển báo. Cho nên chúng ta cần phải chọn ra một cửa sổ tối ưu nhất.
-  Để làm được điều đó chúng ta sẽ sử dụng kỹ thuật Non-maximum Suppression (NMS)

#### Non-maximum Suppression (NMS)
Input: Một danh sách B là các cửa sổ thỏa mãn, cùng với các xác suất dự đoán tương ứng và cuối cùng là ngưỡng overlap N.
Output: Danh sách D các cửa sổ tối ưu cuối cùng.
Các bước thực hiện: 
1.	Chọn cửa sổ có xác suất dự đoán cao nhất. Xóa nó khỏi B và đưa nó vào D. 
2.	Tính giá trị IOU(Intersection over Union) của cửa sổ mới được chọn với những cửa sổ còn lại. Nếu giá trị IOU lớn hơn ngưỡng N thì ta sẽ xóa nó khỏi lớp B
3.	Tiếp tục chọn cửa sổ có xác suất dự đoán cao nhất còn lại. Quay về bước 2
4.	Lặp cho tới khi không còn giá trị nào trong B
Giá trị IOU được sử dụng để tính toán sự trùng lặp của 2 khung cửa sổ

###
---
###
## Mô hình phân loại biển báo

Mô hình này mục đích là để đưa ra kết quả dự đoán xem trong cửa sổ trượt đó có biển báo hay không và chúng thuộc loại nào.

### Các bước xây dựng:
1. Thu thập dữ liệu
2. Xử lý dữ liệu
3. Phân chia dữ liệu Training và Testing
4. Chọn model và training
5. Đánh giá mô hình và nhận xét

### Thu thập dữ liệu:
Dữ liệu là những bức ảnh biển báo giao thông tự chụp bằng điện thoại. Tùy thuộc vào tần suất xuất hiện nên số ảnh ở mỗi lớp có sự chênh lệch
Ảnh không chứa biển báo: SceneClass13 gồm 3000 tấm

#### Số lượng
Bao gồm 6 classes và 1 class ảnh ngoại cảnh(ảnh không chứa biển báo)
Training:
- Biển Speed limit (40km/h): 109 tấm
- Biển W.207b sign: 108 tấm
- Biển Pedestrians:88 tấm 
- Biển No entry: 92 tấm
- Biển Keep right: 66 tấm
- Biển Roundabout mandatory: 41 tấm
- SceneClass13: 3000 tấm(chỉ dùng 1000 tấm)

Testing:
- Biển Speed limit (40km/h): 18 tấm
- Biển W.207b sign: 24 tấm
- Biển Pedestrians: 27 tấm 
- Biển No entry: 11 tấm
- Biển Keep right: 8 tấm
- Biển Roundabout mandatory: 6 tấm
- SceneClass13: 156 tấm

### Xử lý dữ liệu
Xử lý dữ liệu bao gồm các bước như sau:
1.	Cắt vùng có chứa biển báo ra khỏi ảnh ban đầu bằng công cụ “Crop” trong Image trên Windows 10.
2.	Tăng độ sáng của những bức ảnh chụp bị ngược sáng hoặc độ sáng thấp.
3.	Chuyển ảnh từ ảnh màu RGB sang ảnh xám.
4.	Resize bức ảnh về chung một kích thước duy nhất là 64x64.
5.	Sử dụng HOG để trích xuất đặc trưng cho bức ảnh. Chuẩn bị cho bước training

### Chọn model và training
Dùng model SVM và KNN để training. Model được import từ scikit-learn
Ở SVM cần quan tâm tới các hyperparameters như sau:
1.	C: 0.01
2.	Probability=true
3.	Random_state=42
4.	Kernel=”linear”

Gọi phương thức model.fit để thực hiện training. Sau khi training xong ta sẽ lưu model lại cho những lần dự đoán tiếp theo

### Đánh giá mô hình
- Ta sẽ đánh giá model của chúng ta bằng tập ảnh “Test”. 
- Tập này bao gồm 20% số ảnh đã chụp được của các lớp. 
- Tiến hành xử lý ảnh trên tập ảnh “Test” như ở tập training. Sau đó gọi phương thức model.predict ta được kết quả như sau:

<img src='images/acc.jpg'>

---
## Data Preprocessing
Given the issues identified above, I decided to explore the following preprocessing operations (in addition to the standard practice of _normalization_):

* __Normalization__ (standard)
* __Contrast enhancement__ (done as part of normalization process)
  * I used this Scikit [histogram equalization function](http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist), which not only normalizes the images, but also enhances local contrast details in regions that are darker or lighter than most of the image. You can see from the image sample below this also inherently increases the brightness of the image. [(link to source code)](Traffic_Sign_Classifier_final_v5.py#L231)

   <img src='images/writeup/orig_vs_norm.jpg' width="25%"/>

* __Augmentation__
  * __Increase total number of images__, so that the model has more training examples to learn from.
  * __Create an equal distribution of images__ (i.e., same number of images per class) so that the model has a sufficient number of training examples in each class. I initially tested models on sets of 4k images per class, and found that models performed better with more images. I ultimately generated a set of 6k images per class for the final model.
  * __Apply affine transformations__. Used to generate images with various sets of perturbations. Specifically: rotation, shift, shearing, and zoom. But, I decided _not_ to apply horizontal/vertical flipping as this didn't seem pertinent to real-life use cases.
  * __Apply ZCA whitening__ to accentuate edges.
  * __Apply color transformations__
    * _Color channel shifts_ -- This was done to create slight color derivations to help prevent the model from overfitting on specific color shades. This intuitively seemed like a better strategy than grayscaling.
    * _Grayscaling_ -- This was performed separately _after_ all of the above transformations. Due to the high darkness and low contrast issues, applying grayscale before the other transformations didn't make sense. It would only make the contrast issue worse. So, I decided to test the grayscale versions as a separate data set to see if it boosted performance (spoiler alert: it didn't).

Here is the snippet of code that takes the already normalized images (with contrast enhanced) and applies the other transformations listed above. It outputs a new training set with 6k images per class, including the set of normalized training images.

[(link to source code)](Traffic_Sign_Classifier_final_v5.py#L285)

<img src='images/writeup/keras-aug-function.jpg' width="60%"/>

<img src='images/writeup/aug-function.jpg' width="96%"/>

<img src='images/writeup/aug-count.jpg' width="62%"/>


### Augmented Image Samples
Here is a sample of a traffic sign images after the complete set of **normalization, contrast enhancement, and augmentation** listed above.

<img src='images/writeup/augmented-sample.jpg' width="100%"/>


### Grayscaling
Here is a sample of images with **grayscaling** then applied. At first glance, it doesn't appear that grayscaling improves the images in any meaningful way. So, my hypothesis was that the grayscaled versions would perform the same or worse than the augmented images (this turned out to be correct).

<img src='images/writeup/grayscale-sample.jpg' width="85%"/>


---
## Model Architecture

I tested a variety of models (more than 25 different combinations). Ultimately, I settled on a relatively small and simple architecture that was easy to train and still delivered great performance. My final model consisted of the following layers:

<img src='images/writeup/architecture-diagram.png' width="60%"/>


###
Here is a snapshot of the code. You can see that I use: (a) a relu activation on every layer, (b) maxpooling on alternating convolutional layers with a 5x5 filter, and (c) dropouts on the two fully connected layers with a 0.5 keep probability.

[(link to source code)](Traffic_Sign_Classifier_final_v5.py#L633)

<img src='images/writeup/final-model-code.jpg' width="100%"/>


###
Here are my **training and loss functions**. You can see that I use AdamOptimizer to take advantage of its built-in hyperparameter tuning, which varies the learning rate based on moving averages (momentum) to help the model converge faster, without having to manually tune it myself. You'll notice that I also use L2 regularization to help prevent overfitting.

[(link to source code)](Traffic_Sign_Classifier_final_v5.py#L713)

<img src='images/writeup/training-and-loss-functions.jpg' width="100%"/>


###
Here are the **hyperparameters** I used. My goal was to get the model to converge in less than 50 epochs. Essentially, given time constraints, I didn't want to spend more than two hours training the model. Everything else is pretty standard. Although, I did decrease my L2 decay rate (i.e. lower penalty on weights) during the tuning process, which yielded a small lift in performance.  

[(link to source code)](Traffic_Sign_Classifier_final_v5.py#L545)

<img src='images/writeup/hyperparams.jpg' width="37%"/>


###
Here is the output when I construct the graph. I use print statements to verify that the model structure matches my expectations. I find this very useful as it's easy to get confused when you're tweaking and testing lots of different models. Especially at 3am.  =)

[(link to source code)](Traffic_Sign_Classifier_final_v5.py#L772)

<img src='images/writeup/final-graph-output.jpg' width="50%"/>


###
### Final Model Results:
* training set accuracy of **100%**
* validation set accuracy of **99.4%**
* test set accuracy of **98.2%**

###
### Model Iteration & Tuning
Here I'll try to summarize the approach I took to find a solution that exceeded the benchmark validation set accuracy of 0.93. Although some of the details got lost in the fog of war. I battled with these models for too many days. If you're curious, you can view a fairly exhaustive list of the models I tested [here](model-performance-summary-v2.xlsx).

#### Phase 1
The first steps were to get the most basic version of the LeNet CNN running and begin tuning it. I got 83% validation accuracy without any modifications to the model or preprocessing of the training data. Adding regularization and tuning the hyperparameters made the performance worse. So, I started to explore different types of architectures.

#### Phase 2
This is where I started making mistakes that cost me a lot of time (although I learned a lot in the process). In hindsight, I should have done two simple things at this point: (1) start applying some basic preprocessing to the data and testing the performance impact, and (2) keep iterating on the LeNet architecture by incrementally adding and deepening the layers.

Instead, I started explore different architectures such as [DenseNets](https://arxiv.org/abs/1608.06993). Just look at this diagram from [the paper](); how hard could it be, right?

Wrong.

<img src='images/writeup/densenet.jpg' width="90%"/>

DenseNets didn't seem overly complex at the time, and I probably could have gotten them working if I'd just focused on this. However, in parallel, I tried to get Tensorboard working. Trying to tackle both of these at once was a disaster. In short, creating DenseNets requires a lot of nested functions to create all of the various blocks of convolutional layers. Getting the Tensorboard namespaces to work, getting all of your variables to initialize properly, and getting all of the data to flow in and out of these blocks was a huge challenge. After a bunch of research and trial and error (and coffee), I ultimately abandoned this path. ¯\_(ツ)_/¯

I then tried to implement the (much simpler) inception framework discussed by Vincent [during the lectures](https://www.youtube.com/watch?v=SlTm03bEOxA). After some trial and error, I got an inception network running. But, I couldn't get it to perform better than 80% validation accuracy, so I abandoned this path as well. I believe this approach could have worked, but by this point, I wanted to get back to the basics. So, I decided to focus on data preprocessing and iterating on the original LeNet architecture (which I should have done from the beginning! Arg.)

#### Phase 3
After a day of sleep, yoga, and a few dozen ohms to clear my head...I got back to work.

I started by applying basic transformations to the data and testing simple adjustments to the LeNet architecture. Model performance started to improve, but I still had a bias problem. In the beginning, my models were consistently overfitting the training data and therefore my training accuracy was high but my validation accuracy was still low.

This is a summary of the tactics I deployed to improve performance.

| Model			        |     Validation Accuracy	        					|
|:---------------------|:----------------------------------------------:|
| Basic LeNet      		                                    | 82.6%   	|
| LeNet + bias init =0 (instead of 0.01)    			         | 85.2%		|
| LeNet + bias init + contrast enhancement					   | 92.9%		|
| LeNet + bias init + contrast + augmentation v1 	         | 94.9%		|
| LeNet + bias init + contrast + aug_v1 + deeper layers		| 97.5%     |
| LeNet + bias init + contrast + aug_v1 + more layers	+ regularization		| 98.1%     |
| LeNet + bias init + contrast + aug_v2 + more layers	+ reg tuning		| 99.0%   |
| LeNet + bias init + contrast + aug_v2 + more layers	+ reg tuning + grayscale		| 95.8%  |
| LeNet + bias init + contrast + aug_v2 + more layers	+ reg tuning + more training images	+ more epochs	| 99.4%  |
###
###
Here are more details regarding the tactics above (in order of greatest impact on the model):

* __Contrast enhancement__ &mdash; Pound for pound, this tactic had the greatest impact on performance. It was easy to implement and my validation accuracy immediately jumped more than 7%. I only wish I'd done it sooner. As discussed in my initial exploration of the data, I predicted that the low contrast of many of the original images would make it difficult for the model to recognize the distinct characteristics of each sign. This is obvious even to the human eye! But for some reason I didn't implement this tactic until halfway through the project. **Lesson learned: design and test your pipeline around simple observations and intuitions BEFORE you pursue more complicated strategies.** Keep it simple stupid!
* __Augmentation v1 vs v2__ &mdash; The first iteration of my augmentation function boosted performance by 2% (which was great!). However, my range settings for the affine and color transformations were a little too aggressive. This made the training images overly distorted (this was obvious with the naked eye). Because of these distortions, the model kept overfitting (i.e., it achieved high training accuracy but wasn't able to generalize to the validation set).

   In v2 of my augmentation function, I dialed down the range settings and got a 1% performance boost. Then I added ZCA whitening to improve edge detection and got another 1% lift. In my very last optimization, I then increased the number of images being produced by this function so that there were 6k images per class (instead of 4k). This tactic combined with longer training time yielded the final (and elusive!) 0.4% lift to bring the final validation accuracy to 99.4%. Then I slept.
* __More layers and deeper layers__ &mdash; Surprisingly, and after many iterations, I learned that it doesn't take a high number of layers or incredibly deep layers to achieve human level performance. That said, some modest increases in the size of the model were critical to breaking the 95% accuracy plateau. You can see from the [model diagram](images/writeup/architecture-diagram.png) that I ended up with seven convolutional layers (five more than LeNet) and that my convolutional and fully connected layers are deeper than LeNet as well. Of course, to mitigate this extra learning power, I had to employ regularization tactics.
* __Regularization__ &mdash; Both dropout and L2 regularization proved critical. I made an initial mistake of adding these to the model too early in the process, or had them set too high, which caused the model to underfit. I then removed them altogether until I had a model that was starting to fit and generate high training accuracies. At that point, I added regularization back into the model and started to increase it whenever my model was overfitting (i.e., higher dropout and L2 decay values). After a few overcorrections, I ultimately landed on a dropout of 0.5 and decay of 0.0003.
* __Bias initialization__ &mdash; Initially, I was initializing my biases at 0.01 (using tf.constant). Once I started initializing the biases at zero, my accuracy jumped more than 2%. This was a big surprise. Even after doing more research on the issue, I'm still not exactly sure why this small bias initialization negatively affected the model. My best guess is even this small amount of bias was not self correcting enough during back propagation, and given that the data was normalized, that extra bias was causing additional overfitting in the model. [(link to source code)](Traffic_Sign_Classifier_final_v5.py#L576)
* __Grayscale__ &mdash; Just for shits and giggles, I ran a test on a grayscaled version of the augmented image set. The grayscale set still performed well with a validation accuracy of 95.8%. But, this test turned out to be more trouble than it's worth. The big problem was that there are a bunch of tools out there to help you convert RGB images to grayscale, and none of them (as far as I can tell) provide the correct shape. To feed grayscale images into the network, they need to be rank 4 `(batch_size, 32, 32, 1)`. So, you have to convert each RGB image from `(32, 32, 3)` to `(32, 32, 1)`. Seems simple, right? But all of the scripts I tested strip out the third dimension, yielding an image with shape `(32, 32)`. And, there wasn't much help for this issue on StackOverflow, etc. After lots of troubleshooting, I finally discovered the underlying problem and used a simple matrix multiplication to apply the grayscale conversion while maintaining the right shape. [(link to source code)](Traffic_Sign_Classifier_final_v5.py#L442)

<img src='images/writeup/grayscale-function.jpg' width="70%"/>


###
---
## Test the Model with New Images

### Test Set
I gathered a set of **30 new images** for this phase of testing: 11 of the images were pulled from the internet, and 19 of the images I shot myself around the streets of Prague, which uses the same traffic signs as Germany. Overall, I made the new image set quite challenging in order to learn about the strengths and weaknesses of the model.

Here is the complete set of new images and their corresponding originals.

[(link to source code)](Traffic_Sign_Classifier_final_v5.py#L916)

<img src='images/writeup/test-signs.png' width="90%"/>

### Challenges
Within the new image set, the ones below pose distinct challenges for the model. My hypothesis was that the model would get less than 50% of these correct, while scoring above 80% on the other "normal" new images. In particular, some of the signs I found on the streets of Prague seem particularly challenging. How would the model react when it sees two standard signs combined into a single custom sign? Keep reading to find out!

1. **Large Vehicles Prohibited** &mdash; like many signs that I encountered on the streets of Prague, a single traffic sign includes a combination of two or more signs/symbols.
<img src='images/new-signs/challenging/16-large_vehicles_prohibited_prg_a.jpg' width="10%"/>

2. **No Trucks or Motorcycles** &mdash; again, what are normally two signs are incorporated into one
<img src='images/new-signs/challenging/16-no_trucks_or_motorcycles.jpg' width="10%"/>

3. **Yield** &mdash; yet again, the image includes two signs (this one is from the internet)
<img src='images/new-signs/challenging/13-yield.jpg' width="10%"/>

4. **No Entry** &mdash; the bracket holding up this sign is damaged, so the sign is heavily tilted
<img src='images/new-signs/challenging/17-no_entry_tilt_prg.jpg' width="10%"/>

5. **Turn Right** &mdash; this sign is partially occluded by a very pink van
<img src='images/new-signs/challenging/33-turn_right_occluded_prg.jpg' width="10%"/>

6. **50 km/h** &mdash; the viewing angle makes the image heavily sheared
<img src='images/new-signs/challenging/02-a-50kmh_shear.jpg' width="10%"/>

7. **No Entry** &mdash; this sign has graffiti on it. Punks!
<img src='images/new-signs/challenging/17-b-no_entry_graffiti.jpg' width="10%"/>

8. **Ahead Only** &mdash; this sign is only partially visible
<img src='images/new-signs/challenging/35-ahead_only_occluded.jpg' width="10%"/>


### New Image Test Results
The overall accuracy dropped considerably to 77%, although the model performed pretty well on the new images of "normal" difficulty with 91% accuracy. However, this is still well below the 98.2% accuracy achieved on the original test set. This indicates just how quickly accuracy can drop off when a model encounters new patterns it hasn't yet seen in the training set.

| Image difficulty level|   Correct    |   Out of	 |     Accuracy	  	|
|:---------------------:|:------------:|:-----------:|:-----------------:|
| normal     	         |     20   		|		22     |			91%	      |
| hard     			      |     3			|	  	8		 |        38%        |
| **total**			   	|     **23**	|	**30**    |	    **77%**       |

###
### Top 5 Predictions
Below you can see the top 5 predictions and the corresponding softmax probabilities for each test image.  

[(link to source code)](Traffic_Sign_Classifier_final_v5.py#L1078)

<img src='images/notebook-outputs/output_72_1.png' width="100%"/>
<img src='images/notebook-outputs/output_72_2.png' width="100%"/>
<img src='images/notebook-outputs/output_72_3.png' width="100%"/>
<img src='images/notebook-outputs/output_72_4.png' width="100%"/>
<img src='images/notebook-outputs/output_72_5.png' width="100%"/>
<img src='images/notebook-outputs/output_72_6.png' width="100%"/>
<img src='images/notebook-outputs/output_72_7.png' width="100%"/>
<img src='images/notebook-outputs/output_72_8.png' width="100%"/>
<img src='images/notebook-outputs/output_72_9.png' width="100%"/>
<img src='images/notebook-outputs/output_72_10.png' width="100%"/>
<img src='images/notebook-outputs/output_72_11.png' width="100%"/>
<img src='images/notebook-outputs/output_72_12.png' width="100%"/>
<img src='images/notebook-outputs/output_72_13.png' width="100%"/>
<img src='images/notebook-outputs/output_72_14.png' width="100%"/>
<img src='images/notebook-outputs/output_72_15.png' width="100%"/>
<img src='images/notebook-outputs/output_72_16.png' width="100%"/>
<img src='images/notebook-outputs/output_72_17.png' width="100%"/>
<img src='images/notebook-outputs/output_72_18.png' width="100%"/>
<img src='images/notebook-outputs/output_72_19.png' width="100%"/>
<img src='images/notebook-outputs/output_72_20.png' width="100%"/>
<img src='images/notebook-outputs/output_72_21.png' width="100%"/>
<img src='images/notebook-outputs/output_72_22.png' width="100%"/>
<img src='images/notebook-outputs/output_72_23.png' width="100%"/>
<img src='images/notebook-outputs/output_72_24.png' width="100%"/>
<img src='images/notebook-outputs/output_72_25.png' width="100%"/>
<img src='images/notebook-outputs/output_72_26.png' width="100%"/>
<img src='images/notebook-outputs/output_72_27.png' width="100%"/>
<img src='images/notebook-outputs/output_72_28.png' width="100%"/>
<img src='images/notebook-outputs/output_72_29.png' width="100%"/>
<img src='images/notebook-outputs/output_72_30.png' width="100%"/>


### Precision & Recall &mdash; Original Test Images
Listed below are the precision, recall, and F1 scores for the original set of test images.

[(link to source code)](Traffic_Sign_Classifier_final_v5.py#L1136)

<img src='images/writeup/precision-recall-results.jpg' width="60%"/>

<img src='images/writeup/confusion-matrix.jpg' width="40%"/>


Here are the worst performing classes among the **original test images**.

| Class ID|   Sign Label           |   Precision	 |     Recall	  	|   F1 Score     |
|:-------:|:----------------------|:--------------:|:-----------:|:-----------------:|         
 |    27  |Pedestrians              |     0.61     |     0.52   |   0.56     |   
 |   24   |Road narrows on the right|    0.57       |   0.86   |   0.68     |   
 |    21  |Double curve             |     0.73     |    0.73   |   0.73     |   
 |    37  |Go straight or left      |     0.59      |    1.00   |   0.74     |   
|      0  |Speed limit (20km/h)     |     0.62      |    0.95   |   0.75     |   
 |   29   |Bicycles crossing        |    0.64      |    0.96   |   0.77     |   


### Precision & Recall &mdash; New Images
Here are the worst performing classes for the **new image set**. Not surprisingly, the worst performing class from the original test set (label `27: Pedestrians`) is also one of the poorest performers in the new image set.

| Class ID|   Sign Label                      | Precision| Recall	|  F1 Score  |  Count |
|:-------:|:----------------------------------|:--------:|:------:|:----------:|:------:|  
|  13|   Yield                                |  0.00   |  0.00    |  0.00     |    1   |
|  15|   No vehicles                          |  0.00   |   0.00   |   0.00    |     1  |
|  16|Vehicles over 3.5 metric tons prohibited|  1.00   |   0.33   |   0.50    |     3  |
|   2|   Speed limit (50km/h)                 |  1.00   |   0.50   |   0.67    |     2  |
|  27|   Pedestrians                          |  1.00   |   0.50   |   0.67    |     2  |
|  11|Right-of-way at next intersection       |  0.50   |   1.00   |   0.67    |     1  |
|  14|   Stop                                 |  0.50   |   1.00   |   0.67    |     1  |

There are two thing in particular I want to call out here:

1. If we look at the images from six of the worst performing classes between the two sets, we can see that they all look quite similar. This would help explain the high occurrence of false positives (low precision) and false negatives (low recall). This may also be a case where the transformations done during preprocessing overly distort the images, especially when they're applied to low resolution images. The additional loss in fidelity can make it hard to distinguish some of the symbols from each other.

   Given this, one future improvement to our pipeline would be to review how each transformation affects the various classes, and if needed, create a custom set of transformations to be applied on a class-by-class basis.

<img src='images/writeup/low-precision-recall.jpg' width="60%"/>

###
###
2. I think the most interesting insight from the precision/recall data is the misclassification of label `15: No Vehicles`. If we look at image samples from this class (below), it is arguably the simplest sign and should be one of the easiest to recognize. But upon further inspection, we can see that the contrast boosting function that boosted performance in other classes actually hurts us in this case. This is because any minor spots or shadows on the central white portion of the sign get exacerbated by the contrast enhancement function. These dark spots can then resemble symbols to the model. This is another example of how class-specific preprocessing tactics could improve the pipeline.

<img src='images/writeup/15-no-vehicles.jpg' width="100%"/>


##### Precision Recall Reference Diagram
<img src='images/writeup/precision-recall.png' width="40%"/>
https://en.wikipedia.org/wiki/Precision_and_recall


###
###
---
---
###
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
_Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?_

Given time constraints, I did not complete this optional exercise prior to my project submission.
