
#include "UCPSCLS.h"
cv::Mat clsUCPSD1::computeNormals(std::vector<cv::Mat> camImages,
	cv::Mat Mask = cv::Mat())
	{
	int height = camImages[0].rows;
	int width = camImages[0].cols;
	int numImgs = camImages.size();
	/* populate A */
	cv::Mat A(height * width, numImgs, CV_32FC1, cv::Scalar::all(0));

	for (int k = 0; k < numImgs; k++) {
		int idx = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				A.at<float>(idx++, k) = camImages[k].data[i * width + j] *
					sgn(Mask.at<uchar>(cv::Point(j, i)));
			}
		}
	}

	/* speeding up computation, SVD from A^TA instead of AA^T */
	cv::Mat U, S, Vt;
	cv::SVD::compute(A.t(), S, U, Vt, cv::SVD::MODIFY_A);
	cv::Mat EV = Vt.t();
	cv::Mat N(height, width, CV_32FC3, cv::Scalar::all(0));
	int idx = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (Mask.at<uchar>(cv::Point(j, i)) == 0) {
				N.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 255);
			}
			else {
				float rSxyz = 1.0f / sqrt(EV.at<float>(idx, 0) * EV.at<float>(idx, 0) +
					EV.at<float>(idx, 1) * EV.at<float>(idx, 1) +
					EV.at<float>(idx, 2) * EV.at<float>(idx, 2));
				/* V contains the eigenvectors of A^TA, which are as well the z,x,y
				 * components of the surface normals for each pixel	*/
				float sz = 127.5f +
					127.5f * sgn(EV.at<float>(idx, 0)) *
					fabs(EV.at<float>(idx, 0)) * rSxyz;
				float sx = 127.5f +
					127.5f * sgn(EV.at<float>(idx, 1)) *
					fabs(EV.at<float>(idx, 1)) * rSxyz;
				float sy = 127.5f +
					127.5f * sgn(EV.at<float>(idx, 2)) *
					fabs(EV.at<float>(idx, 2)) * rSxyz;
				N.at<cv::Vec3f>(i, j) = cv::Vec3f(sx, sy, sz);
			}
			idx++;
		}
	}
	return N;
}

void clsUCPSD1::updateHeights(cv::Mat &Normals, cv::Mat &Z, int iterations)
{
	//for (int k = 0; k < iterations; k++) {
	int k = 0;
	cv::Mat Z1;
	float err = FLT_MAX;
	do {
		Z.copyTo(Z1);
		for (int i = 1; i < Normals.rows - 1; i++) {
			for (int j = 1; j < Normals.cols - 1; j++) {
				float zU = Z.at<float>(cv::Point(j, i - 1));
				float zD = Z.at<float>(cv::Point(j, i + 1));
				float zL = Z.at<float>(cv::Point(j - 1, i));
				float zR = Z.at<float>(cv::Point(j + 1, i));
				float nxC = Normals.at<cv::Vec3f>(cv::Point(j, i))[0];
				float nyC = Normals.at<cv::Vec3f>(cv::Point(j, i))[1];
				float nxU = Normals.at<cv::Vec3f>(cv::Point(j, i - 1))[0];
				float nyU = Normals.at<cv::Vec3f>(cv::Point(j, i - 1))[1];
				float nxD = Normals.at<cv::Vec3f>(cv::Point(j, i + 1))[0];
				float nyD = Normals.at<cv::Vec3f>(cv::Point(j, i + 1))[1];
				float nxL = Normals.at<cv::Vec3f>(cv::Point(j - 1, i))[0];
				float nyL = Normals.at<cv::Vec3f>(cv::Point(j - 1, i))[1];
				float nxR = Normals.at<cv::Vec3f>(cv::Point(j + 1, i))[0];
				float nyR = Normals.at<cv::Vec3f>(cv::Point(j + 1, i))[1];
				bool up = fabs(nxU) > FLT_EPSILON && fabs(nyU) > FLT_EPSILON;
				bool down = fabs(nxD) > FLT_EPSILON  && fabs(nyD) > FLT_EPSILON;
				bool left = fabs(nxL) > FLT_EPSILON && fabs(nyL) > FLT_EPSILON;
				bool right = fabs(nxR) > FLT_EPSILON && fabs(nyR) > FLT_EPSILON;

				if (up && down && left && right) {
					Z.at<float>(cv::Point(j, i)) =
						1.0f / 4.0f * (zD + zU + zR + zL + nxU - nxC + nyL - nyC);
				}			  
			}
		}
		err = cv::norm(Z, Z1, cv::NORM_L2) / (Z.rows * Z.cols);
		k++;
	} while (err > 1e-3 && k < iterations);
}


cv::Mat clsUCPSD1::localHeightfield(cv::Mat Normals) {
	const int pyramidLevels = 4;
	const int iterations = 700;
	/* building image pyramid */
	std::vector<cv::Mat> pyrNormals;
	cv::Mat Normalmap = Normals.clone();
	pyrNormals.push_back(Normalmap);

	for (int i = 0; i < pyramidLevels; i++) {
		cv::pyrDown(Normalmap, Normalmap);
		pyrNormals.push_back(Normalmap.clone());
	}

	/* updating depth map along pyramid levels, starting with smallest level at
	 * top */
	cv::Mat Z(pyrNormals[pyramidLevels - 1].rows,
		pyrNormals[pyramidLevels - 1].cols, CV_32FC1, cv::Scalar::all(0));

	for (int i = pyramidLevels - 1; i > 0; i--) {
		updateHeights(pyrNormals[i], Z, iterations);
		cv::pyrUp(Z, Z);
	}

	/* linear transformation of matrix values from [min,max] -> [a,b] */
	double min, max;
	cv::minMaxIdx(Z, &min, &max);
	double a = 50.0, b = 100; // 128; // 150.0;

	for (int i = 0; i < Normals.rows; i++) {
		for (int j = 0; j < Normals.cols; j++) {
			Z.at<float>(cv::Point(j, i)) =
				(float)a +
				(b - a) * ((Z.at<float>(cv::Point(j, i)) - min) / (max - min));
		}
	}
	return Z;
}


std::string clsUCPS::type2str(int type)
{
	/*
			+--------+----+----+----+----+------+------+------+------+
			|        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
			+--------+----+----+----+----+------+------+------+------+
			| CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
			| CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
			| CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
			| CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
			| CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
			| CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
			| CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
			+--------+----+----+----+----+------+------+------+------+
	*/
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}


void clsUCPS::exportOBJ(cv::Mat Depth, cv::Mat Normals, cv::Mat texture) {
	/* writing obj for export */
	std::ofstream objFile, mtlFile;
	objFile.open("../output/export.obj");
	int width = Depth.cols;
	int height = Depth.rows;
	/* vertices, normals, texture coords */
	objFile << "mtllib export.mtl" << std::endl;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			objFile << "v " << x << " " << y << " "
				<< Depth.at<float>(cv::Point(x, y)) << std::endl;
			objFile << "vt " << x / (width - 1.0f) << " " << (1.0f - y) / height
				<< " "
				<< "0.0" << std::endl;
			objFile << "vn " << Normals.at<cv::Vec3f>(y, x)[0] << " "
				<< Normals.at<cv::Vec3f>(y, x)[1] << " "
				<< Normals.at<cv::Vec3f>(y, x)[2] << std::endl;
		}
	}

	/* faces */
	objFile << "usemtl picture" << std::endl;

	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			int f1 = x + y * width + 1;
			int f2 = x + y * width + 2;
			int f3 = x + (y + 1) * width + 1;
			int f4 = x + (y + 1) * width + 2;
			objFile << "f " << f1 << "/" << f1 << "/" << f1 << " ";
			objFile << f2 << "/" << f2 << "/" << f2 << " ";
			objFile << f3 << "/" << f3 << "/" << f3 << std::endl;
			objFile << "f " << f2 << "/" << f2 << "/" << f2 << " ";
			objFile << f4 << "/" << f4 << "/" << f4 << " ";
			objFile << f3 << "/" << f3 << "/" << f3 << std::endl;
		}
	}

	/* texture */
	cv::imwrite("../output/export.jpg", texture);
	mtlFile.open("../output/export.mtl");
	mtlFile << "newmtl picture" << std::endl;
	mtlFile << "map_Kd export.jpg" << std::endl;
	objFile.close();
	mtlFile.close();
}

cv::Mat clsUCPS::imageMask(std::vector<cv::Mat> camImages) {
	assert(camImages.size() > 0);
	cv::Mat image = camImages[0].clone();
	int quarter = image.cols / 4.0;
	int eighth = image.rows / 8.0;
	cv::Mat result, bgModel, fgModel;
	cv::Rect area(quarter, eighth, 3 * quarter, 7 * eighth);
	/* grabcut expects rgb images */
	cv::cvtColor(image, image, CV_GRAY2BGR);
	cv::grabCut(image, result, area, bgModel, fgModel, 1, cv::GC_INIT_WITH_RECT);
	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	return result;
}

/**
 * Returns (binary) light pattern img L_j {j = 1..N}
 */
cv::Mat clsUCPS::lightPattern(int width, int height, int j, int N) {
	cv::Mat img(height, width, CV_8UC1, cv::Scalar::all(0));

	for (int y = -(height / 2); y < height / 2; y++) {
		for (int x = -(width / 2); x < width / 2; x++) {
			if (sgn(x * cos(2 * CV_PI * j / N) +
				y * sin(2 * CV_PI * j / N)) == 1) {
				img.at<uchar>(y + height / 2, x + width / 2) = 255;
			}
		}
	}
	return img;
}

cv::Mat clsUCPS::cvtFloatToGrayscale(cv::Mat F, int limit) {
	double min, max;
	cv::minMaxIdx(F, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(F, adjMap, limit / max);
	cv::Mat dst;
	bitwise_not(adjMap, dst);
	return dst;
}

void clsUCPS::exportPLY(int width, int height, cv::Mat Z) {

	/* creating visualization pipeline which basically looks like this:
	 vtkPoints -> vtkPolyData -> vtkPolyDataMapper -> vtkActor -> vtkRenderer */
	vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkPolyDataMapper> modelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	vtkSmartPointer<vtkActor> modelActor = vtkSmartPointer<vtkActor>::New();
	vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
	vtkSmartPointer<vtkCellArray> vtkTriangles = vtkSmartPointer<vtkCellArray>::New();

	/* insert x,y,z coords */
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			points->InsertNextPoint(y, x, Z.at<float>(y, x));
		}
	}

	/* setup the connectivity between grid points */
	vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();
	triangle->GetPointIds()->SetNumberOfIds(3);
	for (int i = 0; i < height - 1; i++) {
		for (int j = 0; j < width - 1; j++) {
			triangle->GetPointIds()->SetId(0, j + (i*width));
			triangle->GetPointIds()->SetId(1, (i + 1)*width + j);
			triangle->GetPointIds()->SetId(2, j + (i*width) + 1);
			vtkTriangles->InsertNextCell(triangle);
			triangle->GetPointIds()->SetId(0, (i + 1)*width + j);
			triangle->GetPointIds()->SetId(1, (i + 1)*width + j + 1);
			triangle->GetPointIds()->SetId(2, j + (i*width) + 1);
			vtkTriangles->InsertNextCell(triangle);
		}
	}
	polyData->SetPoints(points);
	polyData->SetPolys(vtkTriangles);

	/* create two lights */
	vtkSmartPointer<vtkLight> light1 = vtkSmartPointer<vtkLight>::New();
	light1->SetPosition(-1, 1, 1);
	renderer->AddLight(light1);
	vtkSmartPointer<vtkLight> light2 = vtkSmartPointer<vtkLight>::New();
	light2->SetPosition(1, -1, -1);
	renderer->AddLight(light2);

	/* meshlab-ish background */
	modelMapper->SetInputData(polyData);
	renderer->SetBackground(.45, .45, .9);
	renderer->SetBackground2(.0, .0, .0);
	renderer->GradientBackgroundOn();
	vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	modelActor->SetMapper(modelMapper);

	/* setting some properties to make it look just right */
	modelActor->GetProperty()->SetSpecularColor(1, 1, 1);
	modelActor->GetProperty()->SetAmbient(0.2);
	modelActor->GetProperty()->SetDiffuse(0.2);
	modelActor->GetProperty()->SetInterpolationToPhong();
	modelActor->GetProperty()->SetSpecular(0.8);
	modelActor->GetProperty()->SetSpecularPower(8.0);

	renderer->AddActor(modelActor);
	vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	interactor->SetRenderWindow(renderWindow);

	/* export mesh */
	vtkSmartPointer<vtkPLYWriter> plyExporter = vtkSmartPointer<vtkPLYWriter>::New();
	plyExporter->SetInputData(polyData);
	plyExporter->SetFileName("../output/export.ply");
	plyExporter->SetColorModeToDefault();
	plyExporter->SetArrayName("Colors");
	plyExporter->Update();
	plyExporter->Write();

	/* render mesh */
	//renderWindow->Render();
	//interactor->Start();
}

