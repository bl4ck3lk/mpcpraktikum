import fnmatch
import os

__author__ = 'rodrig'

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


# def read_log(logFile):
#
#     f = open(logFile)
#     column1 = []
#     column2 = []
#     result = []
#     system = {}
#     for line in f:
#         if not line.startswith("/"):
#             l = line.split()
#             column1.append(l[0])
#             column2.append(l[1])
#             result.append(l[2])
#     #print column1
#     #print column2
#     #print result
#     #print zip(column1, column2, result )
#     f.close()
#
#     c1 = zip(column1,result)
#     c2 = zip(column2,result)
#     for i in range(len(column1)):
#         if system.has_key(column1[i]):
#             system[column1[i]].append(c2[i])
#         else:
#             system[column1[i]] = [c2[i]]
#     for i in range(len(column2)):
#         if system.has_key(column2[i]):
#             system[column2[i]].append(c1[i])
#         else:
#             system[column2[i]] = [c1[i]]
#     #print system
#     return system
#     #for key, value in system.iteritems():
#     #    print key, value

def get_gold(path):
    """
    read gold file into data structures
    :rtype : dict
    """

    # scan through subdirectories of a given path
    files = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            files.append(os.path.join(root, filename))
    #print files

    # stores the data into dict based on path:file
    datadict = {}
    for f in files:
        name = f.split("/")
        pathName = os.sep.join(name[:-1]).lower()
        fileName = name[-1].lower()
        datadict[pathName] = datadict.get(pathName, []) + [fileName]

    # writes the data from the dict into a file
    output = open('gold.txt', 'w')
    for key in datadict.iterkeys():
        fs = datadict[key]
        #print "dir: " + key
        output.write(key + "\n")
        for f in fs:
            for f2 in fs:
                if f != f2:
                    output.write(f + "\t" + f2 + "\n")
    output.close()
    return datadict
    #for k in datadict.keys():
    #    print datadict[k]


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
                break

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

    print "True Positives", tp/2
    print "False Negatives", fn/2
    print "True Negatives", tn/2
    print "False Positives", fp/2

    # sanity check
    if not ((len(similar) + len(dissimilar)) == tp + fn + tn + fp):
        raise Exception("Something gone wrong!")



if __name__ == "__main__":
    #gold = get_gold("/Users/rodrigpro/Flickr/download_imgs/downloaded")
    #print gold
    #{'/.../thespire/00001': ['1504224139_f47bd48928_2181_11431482@n00.jpg',...],...}
    #system = read_log("log.txt")
    #{'IMG_9726.ppm': [('IMG_9730.ppm', '-1'),...}
    system = read_log_simple("matchGraphSandPics.log")  # tuple(list(similar), list(dissimilar))
    gold = read_gold_simple("goldSandPics.txt")  # list(list(cluster))
    #import time
    #t0 = time.clock()
    evaluate(gold, system)
    #print time.clock() - t0, "seconds process time"