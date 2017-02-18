#include "opencv2\core.hpp"
#include "opencv2\imgcodecs.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\calib3d.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

// size of the squares on the board
const float calibrationSquareDimension = 0.024f; //meters
// amount of inside crossings on the pattern
const Size chessboardDimensions = Size(6, 9);

vector<Point3f> pointsToDraw;
vector<Point2f> resultingPoints;
Mat rotationMatrix;

vector<Point3f> points;
vector<Vec2f> foundPoints;
bool found = false;

//true for calibration mode, false for using predefined calibrated values
const bool calibrationMode = false;
//predefined calibrated values
const Mat CalibratedValues = (Mat_<double>(3, 3) << 703.235, 0, 3, 0, 526.117, 4.5, 0, 0, 1);
const Mat DistanceCalibrated = (Mat_<double>(5, 1) << -0.0397106, 0.000333702, 0.00542373, 0.00241272, -5.83182e-07);

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
void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, Size(9,6), pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			allFoundCorners.push_back(pointBuf);
		}
		
		if (showResults)
		{
			drawChessboardCorners(*iter, Size(9, 6), pointBuf, found);
			imshow("looking for corners", *iter);
			waitKey(0);
		}
	}

}

//calibrate the camera based on a set of images, the board dimensions and size of a square
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	vector<vector<Point2f>> checkerBoardImageSpacePoints;
	getChessboardCorners(calibrationImages, checkerBoardImageSpacePoints, false);

	vector<vector<Point3f>> worldSpaceCornerPoints(1);

	createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(checkerBoardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

	vector<Mat> rVectors, tVectors;
	distanceCoefficients = Mat::zeros(8, 1, CV_64F);

	calibrateCamera(worldSpaceCornerPoints, checkerBoardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);
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


//main function to start webcam and do the calibration and drawings
int main(int argv, char** argc, int mode)
{
	//possible scenes
	Mat frame;
	Mat drawToFrame;
	Mat undistortFrame0;
	Mat undistortFrame1;
	Mat rvec, tvec;

	//camera matrix and distance coefficients
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	Mat distanceCoefficients;

	//images to use for calibration
	vector<Mat> savedImages;

	//found corners
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	Mat corners;
	
	VideoCapture vid(0);

	//bool to know if calibration is done
	bool calibrated = false;

	
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
					
					createKnownBoardPosition(chessboardDimensions, calibrationSquareDimension, points);
					points.resize(foundPoints.size());
					

					solvePnP(points, foundPoints, CalibratedValues, DistanceCalibrated, rvec, tvec);
					//Rodrigues(rvec, rotationMatrix);

					
					pointsToDraw.push_back(Point3f(0, 0, 0));
					pointsToDraw.push_back(Point3f(4*calibrationSquareDimension, 0, 0));
					pointsToDraw.push_back(Point3f(0, 4*calibrationSquareDimension, 0));
					pointsToDraw.push_back(Point3f(0, 0, 4*calibrationSquareDimension));
					pointsToDraw.push_back(Point3f(2 * calibrationSquareDimension, 0, 0));
					pointsToDraw.push_back(Point3f(2 * calibrationSquareDimension, 2 * calibrationSquareDimension, 0));
					pointsToDraw.push_back(Point3f(0, 2 * calibrationSquareDimension, 0));
					pointsToDraw.push_back(Point3f(0, 0, 2 * calibrationSquareDimension));
					pointsToDraw.push_back(Point3f(2 * calibrationSquareDimension, 0, 2 * calibrationSquareDimension));
					pointsToDraw.push_back(Point3f(2 * calibrationSquareDimension, 2 * calibrationSquareDimension, 2 * calibrationSquareDimension));
					pointsToDraw.push_back(Point3f(0, 2 * calibrationSquareDimension, 2 * calibrationSquareDimension));
					
					projectPoints(pointsToDraw, rvec, tvec, CalibratedValues, DistanceCalibrated, resultingPoints);
					line(frame, resultingPoints.at(0), resultingPoints.at(1), (255, 0, 0), 1, 8, 0);
					line(frame, resultingPoints.at(0), resultingPoints.at(2), (0, 255, 0), 1, 8, 0);
					line(frame, resultingPoints.at(0), resultingPoints.at(3), (0 , 0, 255), 1, 8, 0);
					line(frame, resultingPoints.at(0), resultingPoints.at(4), (0, 255, 255), 1, 8, 0);
					line(frame, resultingPoints.at(4), resultingPoints.at(5), (0, 255, 255), 1, 8, 0);
					line(frame, resultingPoints.at(5), resultingPoints.at(6), (0, 255, 255), 1, 8, 0);
					line(frame, resultingPoints.at(6), resultingPoints.at(0), (0, 255, 255), 1, 8, 0);
					line(frame, resultingPoints.at(7), resultingPoints.at(8), (0, 255, 255), 1, 8, 0);
					line(frame, resultingPoints.at(8), resultingPoints.at(9), (0, 255, 255), 1, 8, 0);
					line(frame, resultingPoints.at(9), resultingPoints.at(10), (0, 255, 255), 1, 8, 0);
					line(frame, resultingPoints.at(10), resultingPoints.at(7), (0, 255, 255), 1, 8, 0);
					line(frame, resultingPoints.at(0), resultingPoints.at(7), (0, 255, 255), 1, 8, 0);
					line(frame, resultingPoints.at(4), resultingPoints.at(8), (0, 255, 255), 1, 8, 0);
					line(frame, resultingPoints.at(5), resultingPoints.at(9), (0, 255, 255), 1, 8, 0);
					line(frame, resultingPoints.at(6), resultingPoints.at(10), (0, 255, 255), 1, 8, 0);
				}
			undistort(frame, undistortFrame1, CalibratedValues, DistanceCalibrated);
			imshow("Webcam", undistortFrame1);
			char character = waitKey(1000 / framesPerSecond);
			}
		}
	}

	return 0;
}	