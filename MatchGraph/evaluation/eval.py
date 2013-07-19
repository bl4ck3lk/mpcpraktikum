import fnmatch
import os
import sys
import getopt

def read_log_simple(logFile):
    f = open(logFile)
    similar = []
    dissimilar = []
    for line in f:
        if line[0].isalnum():
            l = line.split()
            filesPart = ":".join(l[0:2])  # tab space
            if l[2] == "-1":
                dissimilar.append(filesPart)
            else:
                similar.append(filesPart)
        else:
            print line
    return similar, dissimilar

def read_gold_simple(goldFile):
    f = open(goldFile)
    clusters = []
    for line in f:
        if line.startswith("/"):
            clusters.append([])  # new cluster
            #continue
        else:
            l = line.split()
            filesPart = ":".join(l[0:2])  # tab space
            clusters[-1].append(filesPart)
    return clusters


def get_gold(path, outputFile):
    """
    read gold file into data structures
    :rtype : dict
    """

    # scan through subdirectories of a given path
    files = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            files.append(os.path.join(root, filename))

    # stores the data into dict based on path:file
    datadict = {}
    for f in files:
        name = f.split("/")
        pathName = os.sep.join(name[:-1]).lower()
        fileName = name[-1].lower()
        datadict[pathName] = datadict.get(pathName, []) + [fileName]

    # writes the data from the dict into a file
    output = open(outputFile, 'w')
    for key in datadict.iterkeys():
        fs = datadict[key]
        output.write(key + "\n")
        for f in fs:
            for f2 in fs:
                if f != f2:
                    output.write(f + "\t" + f2 + "\n")
    output.close()
    print "Saved in:",outputFile
    return datadict


def evaluate(gold, system):
    """
    evaluate system output based on gold standard data
    """
    similar = system[0]
    dissimilar = system[1]

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    unk = 0

    # true positive, in cluster and system got it right, i.e. 1
    for s in similar:
        for cluster in gold:
            if s in cluster:
                tp += 1
                break  # once we found s in some cluster, we don't check other clusters

    # false negative, in cluster but system got -1
    for d in dissimilar:
        for cluster in gold:
            if d in cluster:
                fn += 1
                break

    # true negative, not in cluster and system got -1
    tn = len(dissimilar) - fn

    # false positive, not in cluster but system got 1
    fp = len(similar) - tp

    print "True Positives =", tp/2
    print "False Negatives =", fn/2
    print "True Negatives =", tn/2
    print "False Positives =", fp/2
    print "----------------------"
    print "Total compared =", (len(similar) + len(dissimilar))/2
    print "----------------------"

    print "Precision = %0.2f" %(tp/(tp+fp*1.0))
    print "Recall = %0.2f" %(tp/(tp+fn*1.0))
    print "Accuracy (Hit ratio) = %0.3f" %((tp+tn)/(tp+tn+fp+fn*1.0))
    #print "F-Measure", 2 * ((precision * recall)/(precision + recall))

    # sanity check
    if not ((len(similar) + len(dissimilar)) == tp + fn + tn + fp):
        raise Exception("Something gone wrong!")

def usage():
    usage = """
    -h --help       USAGE: 'python eval.py -g goldFile -s systemOutput' OR 'python eval.py -c clusterRoot'
    -g --goldlog    path/to/goldstandard/file
    -s --syslog     path/to/system/output/file
    -c --cluster    path/to/folder/containing/subfolders (use this param alone)
    """
    print usage

def main(argv):
    if len(argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "hg:s:c:", ["help", "goldlog", "outlog", "cluster"])
        except getopt.GetoptError, err:
            print str(err)
            usage()
            sys.exit(2)

        gold = None
        system = None

        for o, a in opts:
            if o == "-h":
                print usage()
            elif o in ("-g", "--goldlog"):
                gold = read_gold_simple(a)  # list(list(cluster))
            elif o in ("-s", "--outlog"):
                system = read_log_simple(a)  # tuple(list(similar), list(dissimilar))
            elif o in ("-c", "--cluster"):
                get_gold(a, a.split(os.sep)[-1]+".log")
            else:
                assert False, "Invalid input"

        if gold and system:
            evaluate(gold, system)
    else:
        print usage()

if __name__ == "__main__":
    main(sys.argv)


