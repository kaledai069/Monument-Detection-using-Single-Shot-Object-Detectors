import glob
import xml.etree.ElementTree as ET
import pandas as pd
import sys

def xml_to_df(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main(path,csv_name):
    dataframe = xml_to_df(f'{path}')
    new_dataframe = dataframe.groupby(['class']).count()
    new_dataframe[["filename"]].to_csv(f"./Report/{csv_name}.csv")

if __name__ == '__main__':
    path_name = sys.argv[1]
    csv_file_name = sys.argv[2]
    main(path_name, csv_file_name)