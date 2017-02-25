//by Matthijs van der Jagt 3895963 and Joep Hamersma 5571995

#include "opencv2\core.hpp"
#include "opencv2\imgcodecs.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\calib3d.hpp"
#include "opencv2\video.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

// size of the squares on the board
const float calibrationSquareDimension = 0.115f; //meters
// amount of inside crossings on the pattern
const Size chessboardDimensions = Size(8, 6);



//true for calibration mode, false for using predefined calibrated values
const bool calibrationMode = false;
const bool videotest = true;
const bool backgroundMaking = false;




//predefined calibrated values
const Mat CalibratedValues = (Mat_<float>(3, 3) << 495.231
	,0
	,331.735
	,0
	,496.27
	,250.625
	,0
	,0
	,1);
const Mat DistanceCalibrated = (Mat_<float>(5, 1) << -0.331566
	,0.179457
	,- 0.0044484
	,0.00591031
	,- 0.0797654);

//given a board create the positions of all crossings
void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{

	for (int i = 0; i < boardSize.height; i++)
	{
		for (int j = 0; j < boardSize.width;j++)
		{
			corners.push_back(Point3f(j * squareEdgeLength, i*squareEdgeLength, 0.0f));
		}
	}
}

//get the chessboard corners for a bunch of images
void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, vector<Mat>& outputImages, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(8,6), pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		
		if (found)
		{
			allFoundCorners.push_back(pointBuf);
		}
		
		if (showResults)
		{
			drawChessboardCorners(*iter, Size(8, 6), pointBuf, found);
			imshow("looking for corners", *iter);
			char c = waitKey();

			//accept frame or not
			switch (c)
			{
			case 'y': //y key to accept frame
				outputImages.push_back(*iter);
				break;
			case 'n': //n key to ignore frame
				break;
			}
		}
	}

}

//calibrate the camera based on a set of images, the board dimensions and size of a square
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients)
{

	vector<Mat> extra;
	vector<vector<Point2f>> checkerBoardImageSpacePoints;
	getChessboardCorners(calibrationImages, checkerBoardImageSpacePoints, extra, false);

	vector<vector<Point3f>> worldSpaceCornerPoints(1);

	createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(checkerBoardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

	vector<Mat> rVectors, tVectors;
	distanceCoefficients = Mat::zeros(8, 1, CV_64F);

	Size imageSize = calibrationImages[0].size();

	calibrateCamera(worldSpaceCornerPoints, checkerBoardImageSpacePoints, imageSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);
}

//saving the values of the calibration in a simple file
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
	ofstream outStream(name);
	if (outStream)
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = cameraMatrix.at<double>(r, c);
				outStream << value << endl;
			}
		}

		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = distanceCoefficients.at<double>(r, c);
				outStream << value << endl;
			}
		}
		outStream.close();
		return true;
	}

	return false;
}

//get the coordinates for the axes and the cube to draw
vector<Point3f> getCubeAndAxesPoints(float squareSize)
{
	vector<Point3f> temp;
	temp.push_back(Point3f(0, 0, 0)); //origin
	temp.push_back(Point3f(4 * squareSize, 0, 0)); //x-axis
	temp.push_back(Point3f(0, 4 * squareSize, 0)); //y-axis
	temp.push_back(Point3f(0, 0, -4 * squareSize)); //z-axis
	//cube corners, z = 0 plane first
	temp.push_back(Point3f(2 * squareSize, 0, 0)); 
	temp.push_back(Point3f(2 * squareSize, 2 * squareSize, 0));
	temp.push_back(Point3f(0, 2 * squareSize, 0));
	temp.push_back(Point3f(0, 0, -2 * squareSize));
	temp.push_back(Point3f(2 * squareSize, 0, -2 * squareSize));
	temp.push_back(Point3f(2 * squareSize, 2 * squareSize, -2 * squareSize));
	temp.push_back(Point3f(0, 2 * squareSize, -2 * squareSize));
	return temp;
}

//actually draw the axes and cube onto image
void drawCubeAndAxes(Mat img, vector<Point2f> coordinates)
{
	arrowedLine(img, coordinates.at(0), coordinates.at(1), CV_RGB(255, 0, 0), 1, 8, 0); //x-axis
	arrowedLine(img, coordinates.at(0), coordinates.at(2), CV_RGB(0, 255, 0), 1, 8, 0); //y-axis
	arrowedLine(img, coordinates.at(0), coordinates.at(3), CV_RGB(0, 0, 255), 1, 8, 0); //z-axis
	//draw bottom 4 lines of cube
	line(img, coordinates.at(0), coordinates.at(4), CV_RGB(0, 255, 255), 1, 8, 0);
	line(img, coordinates.at(4), coordinates.at(5), CV_RGB(0, 255, 255), 1, 8, 0);
	line(img, coordinates.at(5), coordinates.at(6), CV_RGB(0, 255, 255), 1, 8, 0);
	line(img, coordinates.at(6), coordinates.at(0), CV_RGB(0, 255, 255), 1, 8, 0);
	//draw top 4 lines of cube
	line(img, coordinates.at(7), coordinates.at(8), CV_RGB(0, 255, 255), 1, 8, 0);
	line(img, coordinates.at(8), coordinates.at(9), CV_RGB(0, 255, 255), 1, 8, 0);
	line(img, coordinates.at(9), coordinates.at(10), CV_RGB(0, 255, 255), 1, 8, 0);
	line(img, coordinates.at(10), coordinates.at(7), CV_RGB(0, 255, 255), 1, 8, 0);
	//draw connecting lines
	line(img, coordinates.at(0), coordinates.at(7), CV_RGB(0, 255, 255), 1, 8, 0);
	line(img, coordinates.at(4), coordinates.at(8), CV_RGB(0, 255, 255), 1, 8, 0);
	line(img, coordinates.at(5), coordinates.at(9), CV_RGB(0, 255, 255), 1, 8, 0);
	line(img, coordinates.at(6), coordinates.at(10), CV_RGB(0, 255, 255), 1, 8, 0);

}

//main function to start webcam and do the calibration and drawings
int main(int argv, char** argc, int mode)
{
	if (videotest)
	{
		//number of images
		int Count = 270;
		vector<Mat> cam1Images;

		//obtain all images from folder
		for (int i = 0; i < Count; i++)
		{
			//change per camera
			string name = format("C:\\Users\\joep\\Desktop\\ComputerVision1\\data\\cam1images\\scene%05d.png", i*10 + 1);
			Mat img = imread(name);
			if (img.empty())
			{
				cerr << name << "can't be loaded" << endl;
				continue;
			}
			Mat frame;
			undistort(img, frame, CalibratedValues, DistanceCalibrated);
			imshow("undistorted", frame);
			imshow("distorted", img);
			waitKey();
			//cam1Images.push_back(img);
		}


		vector<vector<Point2f>> foundCorners;
		vector<Mat> acceptedImages;
		//modified getchessboardcorners so you can say y if you want to use a frame for calibration and n if you want to ignore it
		getChessboardCorners(cam1Images, foundCorners, acceptedImages, true);
		

		Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
		Mat distanceCoefficients;
		cameraCalibration(acceptedImages, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distanceCoefficients);
		
		//change file name per camera
		saveCameraCalibration("camera1calibration", cameraMatrix, distanceCoefficients);


	}
	else if (backgroundMaking)
	{
		int Count = 11;
		vector<Mat> backgroundImages;
		
		Mat imgRef = imread("C:\\Users\\joep\\Desktop\\ComputerVision1\\data\\cam1background\\scene00001.png");
		Mat backgroundPNG = Mat::zeros(imgRef.size(),CV_32FC3);


		//obtain all images from folder
		for (int i = 0; i < Count; i++)
		{
			//change per camera
			string name = format("C:\\Users\\joep\\Desktop\\ComputerVision1\\data\\cam4background\\scene%05d.png", i * 10 + 1);
			Mat img = imread(name);
			Mat img2;
			img.convertTo(img2, CV_32FC3);
			backgroundPNG += img2;
		}
		backgroundPNG /= Count;
		int channels = backgroundPNG.channels();
		imwrite("C:\\Users\\joep\\Desktop\\ComputerVision1\\data\\cam4background\\backgroundcam4.png", backgroundPNG);

	}
	else
	{
		//possible scenes
		Mat frame;
		Mat drawToFrame;
		Mat undistortFrame0;
		Mat undistortFrame1;
		Mat rvec, tvec;

		//lists of points
		vector<Point3f> pointsToDraw;
		vector<Point2f> resultingPoints;
		vector<Point3f> points;
		vector<Point2f> foundPoints;

		//camera matrix and distance coefficients
		Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
		Mat distanceCoefficients;
		Mat rotationMatrix;

		//images to use for calibration
		vector<Mat> savedImages;

		//found corners
		vector<vector<Point2f>> markerCorners, rejectedCandidates;
		Mat corners;

		//booleans
		bool calibrated = false;
		bool found = false;

		VideoCapture vid(0);

		if (!vid.isOpened())
		{
			return 0;
		}

		//fps
		int framesPerSecond = 20;

		namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

		//start the process
		while (true)
		{
			if (!vid.read(frame))
			{
				break;
			}
			else
			{
				//find the crossings on the checkerboard if present

				found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
				//copy frame and draw the crossings
				frame.copyTo(drawToFrame);
				drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);

				//adjust for distortions
				undistort(frame, undistortFrame0, cameraMatrix, distanceCoefficients);


				//calibration mode
				if (calibrationMode)
				{
					//if calibrated remove distortions
					if (calibrated)
					{
						imshow("Webcam", undistortFrame0);
					}
					//else show cornerfindings
					else if (found)
					{
						imshow("Webcam", drawToFrame);
					}
					//otherwise just show frame
					else
					{
						imshow("Webcam", frame);
					}

					char character = waitKey(1000 / framesPerSecond);

					switch (character)
					{
					case ' ': //space key
					//saving image for calibration
						if (found)
						{
							Mat temp;
							frame.copyTo(temp);
							savedImages.push_back(temp);
						}
						break;
					case 13: //enter key
					//start calibration if enough samples gathered
						if (savedImages.size() > 15)
						{
							cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distanceCoefficients);
							saveCameraCalibration("CameraCalibration3", cameraMatrix, distanceCoefficients);
							bool calibrated = true;
						}
						break;

					case 27: //esc key
					//exit
						return 0;
						break;

					case 't':
						//toggle distortion
						calibrated = !calibrated;
						break;
					}
				}
				else  //use predefined camera matrix and distance coefficients to draw axes and cube if possible
				{
					if (found)
					{
						//find rvec and tvec using known 3d and 2d locations
						createKnownBoardPosition(chessboardDimensions, calibrationSquareDimension, points);
						points.resize(foundPoints.size());

						solvePnP(points, foundPoints, CalibratedValues, DistanceCalibrated, rvec, tvec, false, SOLVEPNP_ITERATIVE);

						pointsToDraw = getCubeAndAxesPoints(calibrationSquareDimension);
						projectPoints(pointsToDraw, rvec, tvec, CalibratedValues, DistanceCalibrated, resultingPoints);
						drawCubeAndAxes(frame, resultingPoints);
					}
					undistort(frame, undistortFrame1, CalibratedValues, DistanceCalibrated);
					imshow("Webcam", undistortFrame1);
					char character = waitKey(1000 / framesPerSecond);
				}
			}
		}

		return 0;
	}
}	