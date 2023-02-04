import pandas as pd

def csv_to_label_map(path, data_type):
    df = pd.read_csv(path)
    ignore_list = ['king statue', 'kotilingeshvara', 'til mahadev narayan temple']
    mistake_list = ['degutale temple', 'kritipur tower', 'degu tale']
    correct_list = ['degu tale temple_KDS', 'kirtipur tower']

    class_list = list(df['class'])
    final_class_list = list()

    for class_name in class_list:
        if class_name not in ignore_list and class_name not in correct_list:
            if class_name in mistake_list:
                if class_name == 'degutale temple' or class_name == 'degu tale':
                    class_name = correct_list[0]
                elif class_name == 'kritipur tower':
                    class_name = correct_list[1]
            final_class_list.append(class_name)
            
    class_index = 1
    res_dict = {}

    for class_label in final_class_list:
        res_dict[class_label] = class_index
        class_index += 1

    if data_type == 'list-type':
        return final_class_list
    elif data_type == 'dict-type':
        return res_dict