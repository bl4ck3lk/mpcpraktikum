import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.*;
import org.opencv.highgui.Highgui;

public class SiftExtractorDemo {

	/**
	 * @param args
	 */
	
	Mat image;
	String imgFileName;
	String imgFilePath;
	public SiftExtractorDemo(String imgFile) {
		this.imgFilePath = imgFile;
		this.imgFileName = new File(imgFile).getName();
		this.image = Highgui.imread(imgFile);
	}
	
	public void run() {
	    System.out.println("\nRunning Sift-FeatureExtractor");

//	    Mat image = Highgui.imread(getClass().getResource("/resources/the-it-crowd.jpg").getPath());
	    MatOfKeyPoint kp = new MatOfKeyPoint();
	    FeatureDetector detector = FeatureDetector.create(FeatureDetector.SURF);

	    detector.detect(this.image, kp);
	    System.out.println(String.format("Number of features: %s", kp.toArray().length));
	    
	    // Save the visualized detection.
	    Mat outputImg = new Mat();
	    Features2d.drawKeypoints(this.image, kp, outputImg);
	    String outputFile = "output/" + this.imgFileName;
	    System.out.println(String.format("Writing features to %s", outputFile));
	    Highgui.imwrite(outputFile, outputImg);
	    
	    // TODO: write feature vectors as binary files
//	    System.out.println(kp.dump());

	}
	public static void main(String[] args) {

	    // Load the native library.
	    System.loadLibrary("opencv_java245");
	    File dir = new File(args[0]);
	    System.out.println(dir.exists());
	    for (File child : dir.listFiles()) {
		    new SiftExtractorDemo(child.getAbsolutePath()).run();
	    }
	}
}