precision_list = []
recall_list = []
def PR_Curve():
    for iou_theshold in np.arange(0.5, 0.96, 0.05).tolist():
        image_no = 0
        data_list = []
        accumulated_TP = 0
        accumulated_FP = 0
        gt_ROIs = 0
        for index, ele in enumerate(test_data):
            gt_ind_ROI = 0
            image_name = 'Image ' + str(image_no)

            bounding_boxes = ele[1].numpy()
            class_labels = ele[2].numpy()

            predicted_pred_bboxes = pred_bboxes[index]
            pred_class_labels = pred_labels[index]
            pred_conf_scores = pred_scores[index]

            for _ in class_labels:
                gt_ind_ROI += 1
            gt_ROIs += gt_ind_ROI

            detected_count = 0
            for bbox in predicted_pred_bboxes:
                if bbox[0] > 0 or bbox[1] > 0 or bbox[2] > 0 or bbox[3] > 0:
                    detected_count += 1
                else:
                    break
            for i in range(detected_count):
                tp = 0
                fp = 0
                if pred_class_labels[i] in class_labels:
                    class_label_index = class_labels.tolist().index(pred_class_labels[i])
                    if IOU_score(predicted_pred_bboxes[i], bounding_boxes[class_label_index]) > iou_theshold:
                        tp = 1
                        accumulated_TP += 1
                    else:
                        fp = 1 
                        accumulated_FP += 1
                else:
                    fp = 1
                    accumulated_FP += 1
                precision = float(accumulated_TP) / (float(accumulated_TP) + float(accumulated_FP))
                recall = float(accumulated_TP) / 694.0
                data_list.append((image_name, labels[int(pred_class_labels[i])], pred_conf_scores[i] * 100, tp, fp, detected_count))

            image_no += 1

        column_name = ['Image', 'Detections', 'Confidence %', "TP", "FP", 'detected_count']
        df = pd.DataFrame(data_list, columns = column_name)

        df.sort_values(['Confidence %'], inplace = True, ignore_index=True, ascending = False)

        new_data_list = list()
        accumulated_TP = 0
        accumulated_FP = 0

        for row in df.iterrows():
            if row[1]['TP'] == 1:
                accumulated_TP += 1
            elif row[1]['FP'] == 1:
                accumulated_FP += 1
            precision = float(accumulated_TP) / (float(accumulated_TP) + float(accumulated_FP))
            recall = float(accumulated_TP) / gt_ROIs
            new_data_list.append((row[1]['Image'], row[1]['Detections'], row[1]['Confidence %'], row[1]['TP'], row[1]['FP'], accumulated_TP, accumulated_FP, precision, recall))

        column_name = ['Image', 'Detections', 'Confidence %', "TP", "FP", 'Acc TP', 'Acc FP', 'Precision', 'Recall']
        new_df = pd.DataFrame(new_data_list, columns = column_name)
        
        precision_list.append(new_df['Precision'])
        recall_list.append(new_df['Recall'])
        
#         plt.figure(figsize = (16, 8), dpi = 240)
#         plt.style.use('seaborn-whitegrid')
#         plt.plot(new_df['Recall'], new_df['Precision'])
#         plt.scatter(new_df['Recall'], new_df['Precision'], s = 10)
#         font = {'color': 'black', 'size': 16, 'weight': 'bold'}
#         plt.xlabel('Recall', fontdict = font)
#         plt.ylabel('Precision', fontdict = font)
#         plt.title(f'Precision x Recall Curve (IoU Threshold: {iou_theshold})', font)
#         plt.show()



eleven_point = np.arange(0.0, 1.1, 0.1).tolist()

output_AP_list = list()
for i in range(len(precision_list)):
    list_size = len(precision_list[i])
    
    points_precision_val = list()
    for interpolation_point in eleven_point:
        for val_index in range(list_size):
            if recall_list[i][val_index] > interpolation_point:
                if val_index == 0:
                    points_precision_val.append(precision_list[i][val_index])
                    break
                else:
                    recall_value = interpolation_point
                    high_p = precision_list[i][val_index]
                    low_p = precision_list[i][val_index - 1]
                    high_r = recall_list[i][val_index]
                    low_r = recall_list[i][val_index - 1]
                    points_precision_val.append(calculate_p(recall_value, high_p, low_p, high_r, low_r))
                    break
    output_AP_list.append(np.sum(np.array(points_precision_val)) / 11)