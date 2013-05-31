#include <iostream> 
#include "imageHandler.h"
#include <stdio.h> 
#include <sys/types.h> 
#include <dirent.h> 
#include "Eigen/Eigen/Dense" //For Matrices (CPU)
#include <stdlib.h> //srand, rand
#include <time.h> //time


using namespace std;
using namespace Eigen;

MatrixXf modifiedLaplacian(MatrixXf m, int dim); //function to generate (lambda*dim*Laplacian + identitymatrix)
MatrixXf compareImagesUponConfidenceMeasure(MatrixXf T, MatrixXf F, int dim, float threshold); //__to be implemented on gpu but with another function signature
MatrixXf symmetrize(MatrixXf F, int dim); //symmetrize confidence measure matrix
MatrixXf initExample(MatrixXf T); //init with constructed example for 10 images and hard coded knowledge


/* Nur ein wenig rumgespiele, evtl. kann man es ja brauchen. Wir müssen
   doch von einem Ordner mit Bildern ausgehen und anhand dessen dann den
   G_0 aufbauen usw. */
int main(int argc, char** argv) 
{

    if (argc != 2)
    {
        printf("Usage: initialize <directory>\n");
        return -1;
    }

	ImageHandler* test1 = new ImageHandler(argv[1]);
	printf("*** Imagehandler rdy ***\n");

	test1->sortImages();

	printf("total nr of images: %i\n", test1->getTotalNr());
	printf("image0: %s\n",test1->getImage(0));
	printf("image9: %s\n",test1->getImage(9));
	//printf("image39: %s\n",test1->getImage(39));

	int iterations = 5;
	//int dim = test1->getTotalNr();
	int dim = 10;

	MatrixXf T = MatrixXf::Zero(dim,dim);

	cout << T.rows() << "x" << T.cols() << endl;

	srand(time(NULL)); //initialize random seed (for random init T-matrix)

	for(int i = 0; i < iterations; i++)
	{
		if (i == 0) //G_0: random matching
		{
			//Initialize some random testing Matrix T_0
			//for testing purpose using some random -1s and 1s
			//but symmetrically

			/*for(int j = 0; j < dim; j++)
			{
				//generate random nrs
				int r1 = rand() % dim; //random nr between 0 and dimension-1
				int r2 = 0;
				int v  = (rand() % 2)-1;
				v = (v == 0 ? 1 : v); //random -1 or 1
				do {
					r2 = rand() % dim;
				} while (r1 == r2); //no diagonal elements matching

				//fill matrix
				T(r1,r2) = v;
				T(r2,r1) = v;
			}*/

			T = initExample(T);
			cout << "Init T:\n" << T << endl;
			
		}

		//generate modified laplacian to solve the energy function
		MatrixXf X = modifiedLaplacian(T, dim); //__to be generated on gpu
		//cout << "Intermediate X:\n" << X << endl;

		//solve X*F= T column by column
		MatrixXf F = MatrixXf::Zero(dim, dim);

		//__to be calculated parallel on gpu
		for (int j = 0; j < dim; j++) {
			F.col(j) = X.colPivHouseholderQr().solve(T.col(j));
		}

		//symmetrize F
		//__gpu time?
		F = symmetrize(F, dim);
		cout << "F_" << i << ":\n" << F << endl;

		//based on confidence measure matrix F, which pictures have to be compared
		//__to be done on gpu
		T = compareImagesUponConfidenceMeasure(T, F, dim, 0.1);
		cout << "T_" << i << ":\n" << T << endl;

	}

    return 0; 
}


MatrixXf modifiedLaplacian(MatrixXf m, int dim)
{
	int lambda = 1;	
	MatrixXf res = MatrixXf::Zero(dim,dim);

	//construct W
	for(int i = 0; i<dim;i++)
	{
		for(int j = 0; j < dim; j++)
		{
			if (m(i,j) > 0)
			{
				res(i,j) = m(i,j);
			}
		}
	}
	//cout << "W:\n" << res << endl;
	
	//TODO consider using Degree-Matrix D and Adjacency-Matrix A to compute laplacian as D - A.

	//construct laplacian (modified with *lambda*res+Identity
	for(int i = 0; i<dim;i++)
	{
		for(int j = 0; j < dim; j++)
		{
			if (i==j)
			{
				int neighbours = 0;
				for(int k = 0; k < dim; k++)
				{
					res(i,k) ? neighbours++ : neighbours;
				}
				res(i,j) = neighbours * lambda * dim + 1;
			} else {
				res(i,j) = (res(i,j)==0) ? 0 : -res(i,j);
				res(i,j) *= (lambda * dim);
			}
		}
	}

	//cout << "Laplacian:\n" << res << endl;

	return res;
}

MatrixXf compareImagesUponConfidenceMeasure(MatrixXf T, MatrixXf F, int dim, float threshold)
{
	//Assumption for testing purpose:
	//If confidence measure for 2 images > threshold => images similar.
	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			if (i != j && F(i,j) > threshold)
			{
				printf("confidence for (%d, %d) is %f\n", i, j, F(i,j));
				T(i,j) = 1;
			}
			//TODO we have to decide if just putting in 1 for similarity or a score how similar images are.
		}
	}
	return T;
}


MatrixXf symmetrize(MatrixXf F, int dim)
{
	//set diagonal to zero, dont compare to itself
	//FIXED: rather check above if i != j
	//for(int i = 0; i < dim; i++)
	//{
	//	F(i,i) = 0;
	//}
	return (F+F.transpose())/2;
}

MatrixXf initExample(MatrixXf T){
	T(3,7) = 1; T(7,3) = 1;
	T(3,8) = 1; T(8,3) = 1;

	T(1,6) = -1; T(6,1) = -1;
	T(0,6) = -1; T(6,0) = -1;
	T(5,6) = -1; T(6,5) = -1;

	T(0,2) = 1; T(2,0) = 1;
	T(4,5) = 1; T(5,4) = 1;
	T(2,5) = 1; T(5,2) = 1;

	T(1,9) = 1; T(9,1) = 1;
	return T;
}
