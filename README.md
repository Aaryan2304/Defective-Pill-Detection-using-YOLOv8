# Defective-Pill-Detection using YOLO-V8

**Aim : The objective of the project is to create a Defective Pill Detection System using YOLOv8, a state-of-the-art deep learning algorithm for object detection. This system aims to enhance quality control in pharmaceutical manufacturing by automatically identifying defective pills during production processes.** <br>

**Selected Pre-trained Model : YoLo V8 nano** <br>

**Dataset : https://docs.ultralytics.com/#yolo-licenses-how-is-ultralytics-yolo-licensed** <br>

## Steps :
1) A pretrained model that is recommended for training was loaded -YOLOV8n.pt

2) Then the model was trained on the dataset with 50 epochs and image size 640

3) We performed validation on the model using our dataset and settings that were previously remembered by the model. During validation, the model evaluates its performance on a validation dataset, typically calculating metrics such as mean average precision (mAP) or other relevant metrics for object detection tasks.

4) We did prediction using a pre-trained YOLOv8n (YOLO version 8n) model on a set of test images, saving the predicted results, with a specified image size of 640 pixels and a confidence threshold of 0.2.

5) Then we exported our YOLO model, into the TensorFlow Lite (tflite) format, enabling deployment on mobile and embedded devices.

## Key Learnings :
1)	Effective Utilization of Hardware Acceleration: Using the NVIDIA RTX 3050 GPU significantly enhanced the performance of the YOLOv8 model by accelerating both the training and inference phases. This underscores the importance of choosing powerful and appropriate hardware to optimize computational efficiency in resource-intensive machine learning tasks.

2) Integration and Compatibility in Software Development: The use of Android Studio Jellyfish proved beneficial for its stability and robust features, enabling smooth integration of the YOLOv8 model into an Android application. This highlights how advanced IDEs can support complex functionalities like real-time image processing and ensure compatibility with the latest operating system updates.

3) Use of Jetson Nano would have increased the processing power.
