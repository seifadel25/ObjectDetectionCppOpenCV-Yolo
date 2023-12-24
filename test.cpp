////#include <opencv2/core/core.hpp>
////#include <opencv2/highgui/highgui.hpp>
////#include <opencv2/imgproc.hpp>
////#include <iostream>
////using namespace cv;
////using namespace std;
////int main() {
////	string img_path = "C:\\Users\\Administrator\\Pictures\\Screenshots\\test.png";
////	Mat img = imread(img_path, IMREAD_COLOR);
////	resize(img, img, { 500,500 }, 0, 0, cv::INTER_NEAREST);
////	imshow("Image", img);
////	waitKey(0);
////	return 0;
////}

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace dnn;


int main() {
    // Load names of classes
    string classesFile = "I:\\test\\coco.names";
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    float confThreshold = 0.1; // Confidence threshold
    float nmsThreshold = 0.3;  // Non-maximum suppression threshold
    int inpWidth = 416;        // Width of network's input image
    int inpHeight = 416;       // Height of network's input image


    while (getline(ifs, line)) classes.push_back(line);

    // Load the neural network
    dnn::Net net; // Declare 'net' outside of the try block
    std::ifstream testFile("I:\\test\\yolov7-tiny.cfg");
    if (testFile.good()) {
        std::cout << "File exists and is accessible" << std::endl;
    }
    else {
        std::cout << "File not found or not accessible" << std::endl;
    }
    testFile.close();
    try {
        net = cv::dnn::readNet("I:\\test\\yolov7-tiny.weights",
            "I:\\test\\yolov7-tiny.cfg");
    }
    catch (const cv::Exception& e) {
        cerr << "Error loading YOLOv7 model: " << e.what() << endl;
        return -1;
    }
    // Open a video file or an image file or a camera stream.
    VideoCapture cap(1);

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        cout << "Error opening video stream" << endl;
        return -1;
    }

    Mat frame, blob;

    while (true) {
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, 0.00392, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        // Remove the bounding boxes with low confidence and perform non-maximum suppression
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;
        for (size_t i = 0; i < outs.size(); ++i) {
            // Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];
            int top = max(box.y, 0);
            rectangle(frame, box, Scalar(0, 255, 0), 2); // Draw the bounding box

            // Get the label for the class name and its confidence
            string label = format("%.2f", confidences[idx]);
            if (!classes.empty()) {
                CV_Assert(classIds[idx] < (int)classes.size());
                label = classes[classIds[idx]] + ": " + label; // Class name and confidence
            }

            // Display the label at the top of the bounding box
            int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            top = max(top, labelSize.height);
            rectangle(frame, Point(box.x, top - round(1.5 * labelSize.height)), Point(box.x + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
            putText(frame, label, Point(box.x, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }


        // Display the resulting frame
        imshow("Frame", frame);

        // Press 'q' on keyboard to exit the program
        char c = (char)waitKey(25);
        if (c == 'q')
            break;
    }

    // When everything is done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

    return 0;
}

//#include <opencv2/opencv.hpp>
//#include <iostream>
//using namespace cv;
//
//int main() {
//    // Open the default camera (camera index 0)
//    cv::VideoCapture cap(1);
//
//    // Check if the camera opened successfully
//    if (!cap.isOpened()) {
//        std::cerr << "Error opening camera!" << std::endl;
//        return -1;
//    }
//
//    // Load the pre-trained object detection model (adjust the path accordingly)
//    cv::CascadeClassifier cascade;
//    if (!cascade.load("I:\\test\\haarcascade_frontalface_default.xml")) {
//        std::cerr << "Error loading face cascade!" << std::endl;
//        return -1;
//    }
//
//    cv::Mat frame;
//    while (true) {
//        // Capture a frame from the camera
//        cap >> frame;
//
//        // Check if the frame is empty
//        if (frame.empty()) {
//            std::cerr << "Error capturing frame!" << std::endl;
//            break;
//        }
//
//        // Convert the frame to grayscale for face detection
//        cv::Mat gray;
//        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
//
//        // Detect faces in the frame
//        std::vector<cv::Rect> faces;
//        cascade.detectMultiScale(gray, faces, 1.3, 5);
//
//        // Draw rectangles around the detected faces
//        for (const auto& face : faces) {
//            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
//        }
//
//        // Display the resulting frame
//        cv::imshow("Object Detection", frame);
//
//        // Break the loop if 'Esc' key is pressed
//        if (cv::waitKey(1) == 27) {
//            break;
//        }
//    }
//
//    // Release the camera and close the window
//    cap.release();
//    cv::destroyAllWindows();
//
//    return 0;
//}