import glob
import xml.etree.ElementTree as ET
import sys


def find_spefic(path, miss_labeled):
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            if member[0].text == miss_labeled:
                print(root.find('filename').text)

if __name__ == '__main__':
    path_name = sys.argv[1]
    miss_labeled = sys.argv[2]
    find_spefic(path_name, miss_labeled)
