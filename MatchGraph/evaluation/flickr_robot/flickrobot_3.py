import urllib
import urllib2
import re
import os
import Image

class Robot:
	def run(self, query, n, size, startpage, destination):
	
		query = query.replace( " ", "+" );
	
		suffix = ""
		if(size=="d"):
			suffix = "_d"
		elif(size=="t"):
			suffix = "_t"

		# Pages over 67 are not returned - this is a Flickr hard limit.
		endpage = int(n)+1
		if( endpage > 67 ):
			endpage = 67
			
		# Iterate over the n+1 pages of results
		for i in range(int(startpage),endpage):
			print("Page " + str(i))
			# Create the URL
			url = "http://www.flickr.com/search/?q=" + query + "&page=" + str(i)
			# Scan the page
			for line in urllib2.urlopen(url):
				# Search for image inclusion in the HTML code
				m = re.search(r"(img.*src=\"(.*jpg)\")", line)
				if(m):
					# Get the image location and remove the thumbnail tag _t
					location = m.group(2)
					location = location.replace("_t",suffix)
					#print location
					# Get the filename only
					name = location.rpartition("/")
					
					# Check if the directory with the name of the destination exists
					# And create it if necessary
					dir = "./"+destination+"/"
					d = os.path.dirname(dir)
					if not os.path.exists(d):
						os.makedirs(d)
					# Complete path
					path = dir+query+"_"+name[2]
					
					print("Retrieving " + path)
					if not os.path.exists(path):
						# Download the image
						try:
							image = urllib.urlretrieve(location,path)
						except:
							print( location + " error, skipping." );

						#response = urllib2.urlopen(location)
						#image = response.read();
						
						#file = open(path, "rb")
						#i = Image.open(path)
						#w = i.size[0]
						#h = i.size[1]
						#file.close()
						#if(w<200 or h<200):
						#    os.remove(path)
						#    print("Image too small, deleting " + path)
# Main
def main(query,n,size,startpage,destination):
	myRobot = Robot()
	myRobot.run(query,n,size,startpage,destination)

# Entry point
if __name__ == "__main__":
	import sys
	if(len(sys.argv)!=6):
		print("""Usage: python flickrobot.py query npage size startpage destination
					query: keyword
					npage: number of pages to scan
					size: m = medium, d = default, and t = thumbnail
					startpage: which page to start at (i.e., 1)
					destination: folder to store images
				""")
	else:
		main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])