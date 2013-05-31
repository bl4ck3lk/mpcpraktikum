#include <map>

class ImageHandler
{
	public:
		ImageHandler(char* imgDir); //constructor
		//~ImageHandler(); //destructor
		char* getImage(int img); //get specific image
		int getTotalNr(); //get total nr of images
		void sortImages();

	private:
		std::map<int,char*> images;
		int nrImages; //total number of images
};
